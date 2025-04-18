#!/usr/bin/env python3
import logging
import os
import time
from collections import deque

import numpy as np
import rclpy
from FLAIR_msg.msg import HumanControl, Adapted, Perturbation, SystemControl
from influxdb_client import InfluxDBClient, Point, WriteOptions
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
from vicon_msgs.msg import Position

ADAPTATION_ON = True

NUM_REP = 20

# If all are false, use default Ramp + Wind
CHICANE_MODE = False
WIND_ONLY_MODE = False
ROBUSTNESS_MODE = False
assert not(CHICANE_MODE and WIND_ONLY_MODE), "ERROR both CHICANE_MODE and WIND_ONLY_MODE set to True."
assert not(WIND_ONLY_MODE and ROBUSTNESS_MODE), "ERROR both WIND_ONLY_MODE and ROBUSTNESS_MODE set to True."
assert not(CHICANE_MODE and ROBUSTNESS_MODE), "ERROR both CHICANE_MODE and ROBUSTNESS_MODE set to True."

# Default Damage
DEFAULT_SCALING_SIDE = "right"
DEFAULT_SCALING_AMPLITUDE = 0.7

# Automatic driver  speed
MIN_DRIVER_SPEED = 0.3
MAX_DRIVER_SPEED = 0.5

# Activate/De-activate perturbation
USE_SCALING_BIG_LOOP = True
USE_WIND_SMALL_LOOP = True
USE_DYNAMIC_WAYPOINT_SCALING_CHICANE = False
USE_STATIC_SCALING_CHICANE = False

# Set number of laps on the chicane and wind
NUMBER_LAPS_CHICANE = 2
NUMBER_LAPS_WIND = 4

# Set number of point without damage in the wind
WIND_DELAY = 0

# Big loop on the ramp
BIG_LOOP_RAMP = [
    (-1703, 1700),
    (-1330, 583),
    (-432, 183),
    (839, 109),
    (2878, 973),
    (1934, 1680),
    (1447, 534),
    (2878, 973),
    (1934, 1680),
    (1232, 1500),
]

# Small loop on the floor
SMALL_LOOP_JUNCTION = [
    (424, 1190),
    (-738, 2252),
]
## Smaller Loop
SMALL_LOOP_LAP = [
     (-1667, 1901),
     (-2009, 978),
     (-1762, 331),
     (-1419, -8), 
     (-347, -234),
     (344, 203),
     (642, 1121),
     (97, 2244),
     (-821, 2566),
]
if WIND_ONLY_MODE:
    SMALL_LOOP = NUMBER_LAPS_WIND * SMALL_LOOP_LAP
else:
    SMALL_LOOP = SMALL_LOOP_JUNCTION + NUMBER_LAPS_WIND * SMALL_LOOP_LAP

# Chicane on the floor
CHICANE_LOOP_LAP = [
    (-1723, 405),
    (-1595, 2071),
    (-1286, 2265),
    (1966, 2066),
    (2296, 1613),
    (2328, 388),
    (1895, -12),
    (1495, 65),
    (1272, 298),
    (1229, 1040),
    (851, 1578),
    (347, 1473),
    (-188, 1036),
    (-163, 425),
    (-599, -98),
    (-1440, 14),
]
CHICANE_LOOP = CHICANE_LOOP_LAP * NUMBER_LAPS_CHICANE 

####################
# Common functions #


class MyStreamHandler(logging.StreamHandler):
    def flush(self) -> None:
        self.stream.flush()


def init_influx():
    """Init the Influx client."""

    influx_client = InfluxDBClient(
        url="http://influxdb:8086",
        bucket=os.environ["INFLUXDB_BUCKET"],
        token=os.environ["INFLUXDB_KEY"],
        org="FLAIR",
    )
    return influx_client.write_api(
        write_options=WriteOptions(
            batch_size=500,
            flush_interval=1000,
            jitter_interval=0,
            retry_interval=5000,
            max_retries=1,
            max_retry_delay=30000,
            max_close_wait=300000,
            exponential_base=2,
        )
    )


def writeEntry(
    influx_client,
    topic: str,
    fields: list,
    other_tag: str = None,
    timestamp: int = None,
) -> None:
    """Write on entry to Influx."""

    pt = Point(topic)

    if timestamp:
        pt.time(timestamp)
    else:
        pt.time(time.time_ns())

    if other_tag:
        pt.tag("other_tag", other_tag)
    for f in fields:
        pt = pt.field(f[0], f[1])
    try:
        _ = influx_client.write(record=pt, org="FLAIR", bucket=os.environ["INFLUXDB_BUCKET"])
    except BaseException:
        pass


######################
# Main Vicon class #


class Vicon(Node):

    def __init__(self, loglevel: str = "INFO") -> None:

        super().__init__("vicon")

        # Set up logging
        llevel = getattr(logging, loglevel.upper())
        if not isinstance(llevel, int):
            raise ValueError(f"Invalid log level: {loglevel}")

        formatter = logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d %(name)s %(message)s", datefmt="%s"
        )
        sh = MyStreamHandler()
        sh.setLevel(llevel)
        sh.setFormatter(formatter)
        self._logger = logging.getLogger("Vicon")
        self._logger.setLevel(llevel)
        self._logger.addHandler(sh)
        self._logger.info("Vicon: Starting")

        # Init Influx Client
        self.influx_client = init_influx()

        # Publish vicon ground truth
        self._pub_vel = self.create_publisher(
            Adapted, "/vicon/vicon_ground_truth", qos_profile_sensor_data
        )
        self._pub_vel_sent = False

        # Publish our requested system status
        self._pub_system_control = self.create_publisher(
            SystemControl, "/system_control", qos_profile_sensor_data
        )
        self._pub_system_control_sent = False

        # Publish our requested perturbation
        self._pub_perturbation = self.create_publisher(
            Perturbation, "/set_perturbation", qos_profile_sensor_data
        )
        self._pub_perturbation_sent = False

        # Publish our requested movement
        self._pub_human = self.create_publisher(
            HumanControl, "/cmd_vel_human", qos_profile_sensor_data
        )
        self._pub_human_sent = False

        # Getting Data From the Vicon Receiver Package
        self._sub_vicon = self.create_subscription(
            Position,
            "/vicon/GVRBot_IC/GVRBot_IC",
            self._vicon_callback,
            qos_profile_sensor_data,
        )
        self._sub_vicon_seen = False

        # Necessary attributes
        self.current_rep = 0
        self.lin_vels = deque(maxlen=3)
        self.ang_vels = deque(maxlen=3)

        # Set up the path
        self.big_loop_ramp = BIG_LOOP_RAMP
        self.small_loop = SMALL_LOOP
        self.chicane = CHICANE_LOOP

        # Set up the path
        if CHICANE_MODE:
            self.path = self.chicane + [self.chicane[0]]
        elif WIND_ONLY_MODE:
            self.path = self.small_loop + [self.big_loop_ramp[0]]
        elif ROBUSTNESS_MODE:
            self.big_loop_ramp = self.big_loop_ramp + self.small_loop[0:2]
            self.path = self.big_loop_ramp + [(-1556, 2020)]
        else:
            self.path = self.big_loop_ramp + self.small_loop + [self.big_loop_ramp[0]]

        # Set up perturbations (default no-perturbations)
        self._logger.warning("")
        self.scaling = [False for way_point in range (len(self.path))]
        self.scaling_side = [DEFAULT_SCALING_SIDE for way_point in range (len(self.path))]
        self.scaling_amplitude = [DEFAULT_SCALING_AMPLITUDE for way_point in range (len(self.path))]
        self.wind = [False for way_point in range (len(self.path))]

        # For Chicane
        if CHICANE_MODE:

            self._logger.warning("DRIVER: Running CHICANE driver.")

            assert not (
                USE_DYNAMIC_WAYPOINT_SCALING_CHICANE and USE_STATIC_SCALING_CHICANE
            ), "Both dynamic and static damage are on on the chicane, choose one only."

            # Set up static scaling
            if USE_STATIC_SCALING_CHICANE:
                self._logger.warning("DRIVER: Applying static scaling perturbation.")
                self.scaling = [False] + [True for way_point in range (len(self.path) - 1)]

            # Set up waypoint dynamic scaling
            elif USE_DYNAMIC_WAYPOINT_SCALING_CHICANE:
                self._logger.warning("DRIVER: Applying dynamic scaling perturbation.")
                self.scaling = [False] + [True for way_point in range (len(self.path) - 1)]
                self.scaling_amplitude =[DEFAULT_SCALING_AMPLITUDE]*5 + [DEFAULT_SCALING_AMPLITUDE-0.3]*12
            
                if NUMBER_LAPS_CHICANE > 1:
                    self.scaling_side = self.scaling_side + NUMBER_LAPS_CHICANE * self.scaling_side
                    self.scaling_amplitude = self.scaling_amplitude + NUMBER_LAPS_CHICANE*self.scaling_amplitude

            else:
                self._logger.warning("DRIVER: Applying no perturbation.")

        # For Wind only
        elif WIND_ONLY_MODE:

            self._logger.warning("DRIVER: Running WIND ONLY driver.")

            if USE_WIND_SMALL_LOOP:
                self._logger.warning("DRIVER: Applying wind perturbation in the small loop.")
                self.wind = (
                    [False for way_point in range(WIND_DELAY)] 
                    + [True for way_point in range (len(self.path) - WIND_DELAY)]
                )
            else:
                self._logger.warning("DRIVER: Applying no wind perturbation in the small loop.")

        # For Ramp only for robustness
        elif ROBUSTNESS_MODE:

            self._logger.warning("DRIVER: Running RAMP ONLY for ROBUSTNESS driver.")

            if USE_SCALING_BIG_LOOP:
                self._logger.warning("DRIVER: Applying static scaling perturbation on the ramp in the big loop.")
                self.scaling = [False] + [True for way_point in range (len(self.path) - 1)]
            else:
                self._logger.warning("DRIVER: Applying no perturbation on the ramp in the big loop.")

        # For Ramp and Wind
        else:

            self._logger.warning("DRIVER: Running RAMP and WIND driver.")

            # Set scaling on the ramp
            if USE_SCALING_BIG_LOOP:
                self._logger.warning("DRIVER: Applying static scaling perturbation on the ramp in the big loop.")
                self.scaling = [
                    way_point < len(self.big_loop_ramp)
                    for way_point in range(len(self.path))
                ]
            else:
                self._logger.warning("DRIVER: Applying no perturbation on the ramp in the big loop.")

            # Set wind on the floor
            if USE_WIND_SMALL_LOOP:
                self._logger.warning("DRIVER: Applying wind perturbation in the small loop.")
                self.wind = [
                    way_point >= (len(self.big_loop_ramp) + WIND_DELAY)
                    for way_point in range(len(self.path))
                ]
            else:
                self._logger.warning(
                    "DRIVER: Applying no wind perturbation in the small loop."
                )

        # SAFETY: initialise attributes for safety
        self.done_safety = False
        self.safety_mode = False
        self.safety_path = self.path[0]
        self.normal_path = self.path
        self._smaller_distance_so_far = np.inf

        self.reset_and_start_rep()

        self._logger.warning("")
        self._logger.info("Vicon: Done")


    def reset_and_start_rep(self):
        """Reset the robot and start the next replication."""

        # Increment rep counter
        self.current_rep = self.current_rep + 1
        if self.current_rep > NUM_REP:
            return

        # SAFETY: If caused by a safety stop, drive to initial position with nothing on
        if self.done_safety:
            self._logger.info(f"Bringing robot back to initial position after safety reset.")

            # Send to Influx
            try:
                writeEntry(
                    self.influx_client,
                    "/done_safety",
                    [
                        ("done_safety", True),
                        ("target", self.target),
                        ("smaller_distance", self._smaller_distance_so_far),
                    ],
                )
            except BaseException as e:
                self._logger.warning("Warning: Influx sending failed.")
                self._logger.warning(e)

            # Reset everything
            self.previous_position = None
            self.previous_rot = None
            self.target = 0
            self.done = False
            self.done_safety = False
            self._smaller_distance_so_far = np.inf
            self._send_perturbation(
                scaling=False,
                scaling_side="right",
                scaling_amplitude=1.0,
                dynamic_scaling=False,
                dynamic_scaling_amplitude=1.0,
                dynamic_scaling_interval=0.0,
                offset=False,
                wind=False,
                bernoulli=False,
            )

            # De-activate the current controller
            self._send_system_control(adaptation_reset=True, adaptation_on=False)
            self._send_system_control(adaptation_reset=True, adaptation_on=False)
            self._send_system_control(adaptation_reset=True, adaptation_on=False)

            # Set up the path to be just a reset path
            self._logger.info(f"Entering safety mode.")
            self.safety_mode = True
            self.path = self.safety_path

        # SAFETY: if done with a saetyf reset, put everything back to normal
        if self.safety_mode:
            self._logger.info(f"Exiting safety mode.")
            self.safety_mode = False
            self.path = self.normal_path

        # Reset attributes and perturbations
        self._logger.info(f"Reseting attributes and perturbation for rep {self.current_rep} / {NUM_REP}.")
        self.previous_position = None
        self.previous_rot = None
        self.target = 0
        self.done = False
        self.done_safety = False
        self._smaller_distance_so_far = np.inf
        self._send_perturbation(
            scaling=False,
            scaling_side="right",
            scaling_amplitude=1.0,
            dynamic_scaling=False,
            dynamic_scaling_amplitude=1.0,
            dynamic_scaling_interval=0.0,
            offset=False,
            wind=False,
            bernoulli=False,
        )


        # Putting the robot back one meter to ensure smooth reset
        if self.current_rep > 1:
            self._logger.info(f"Reseting robot for rep {self.current_rep} / {NUM_REP}.")
            new_msg = HumanControl()
            new_msg.joystick.linear.x = -0.3
            new_msg.joystick.angular.z = 0.0
            new_msg.flipper.angular.x = 0.0
            self._send_system_control(adaptation_reset=True)
            self._send_system_control(adaptation_reset=True)
            self._send_system_control(adaptation_reset=True)
            start_t = time.time()
            while time.time() - start_t < 1.0:
                self._send_human(new_msg)
                time.sleep(0.1)

        # Send configuration to Influx
        self._logger.info(f"Sending configuration for rep {self.current_rep} / {NUM_REP}.")
        try:
            # Send the part of the path as str so they can be any size
            writeEntry(
                self.influx_client,
                "/vicon_configuration",
                [
                    ('path', str(self.path)),
                    ('big_loop_ramp', str(self.big_loop_ramp)),
                    ('small_loop', str(self.small_loop)),
                    ('chicane', str(self.chicane)),
                    ('scaling', str(self.scaling)),
                    ('scaling_sides', str(self.scaling_side)),
                    ('scaling_amplitudes', str(self.scaling_amplitude)),
                    ('wind', str(self.wind)),
                    ('adaptation_on', ADAPTATION_ON),
                    ('min_speed', MIN_DRIVER_SPEED),
                    ('max_speed', MAX_DRIVER_SPEED),
                ],
            )
        except BaseException as e:
            self._logger.warning("Warning: Influx sending failed.")
            self._logger.warning(e)

        self._logger.info(f"Starting rep {self.current_rep} / {NUM_REP}.")

    def _vicon_callback(self, msg: Position) -> None:
        """Process vicon data, when receiving them through corresponding topic."""
        if not self._sub_vicon_seen:
            self._logger.info("Present: Vicon Data")
            self._sub_vicon_seen = True
            if self.target == 0:
                self._send_system_control(adaptation_reset=True)

        # Send vicon reading msg
        self._logger.debug("Sent: Velocity {}".format(msg))
        adapted_msg = Adapted()
        adapted_msg.stamp = self.get_clock().now().to_msg()
        linear_speed, angular_speed = self.get_speed(msg)
        adapted_msg.vx = linear_speed[0]
        adapted_msg.vy = linear_speed[1]
        adapted_msg.vz = linear_speed[2]
        adapted_msg.wx = angular_speed[0]
        adapted_msg.wy = angular_speed[1]
        adapted_msg.wz = angular_speed[2]
        quaternion = np.asarray([msg.x_rot, msg.y_rot, msg.z_rot, msg.w])
        try:
            r = R.from_quat(quaternion)
        except Exception:
            return
        current_rot = r.as_euler("XYZ", degrees=True)
        adapted_msg.roll = current_rot[0]
        adapted_msg.pitch = current_rot[1]
        adapted_msg.yaw = current_rot[2]
        adapted_msg.tx = msg.x_trans
        adapted_msg.ty = msg.y_trans
        adapted_msg.tz = msg.z_trans
        self._send_vicon(adapted_msg)

        # Send Path Tracking command
        current_way_point = self.target % len(self.path) # Works for random number of laps
        if current_way_point == len(self.path) - 1:
            self.error_threshold = 0.2
        else:
            self.error_threshold = 0.1
        human_msg = self.follow_path(msg)

        # Send SystemControl
        self._send_system_control()

        # Send Perturbation
        self._send_perturbation(
            scaling=self.scaling[current_way_point],
            scaling_side=self.scaling_side[current_way_point],
            scaling_amplitude=self.scaling_amplitude[current_way_point],
            dynamic_scaling=False,
            dynamic_scaling_amplitude=1.0,
            dynamic_scaling_interval=0.0,
            offset=False,
            wind=self.wind[current_way_point],
            bernoulli=False,
        )

        # Reset if done with this rep
        if not self.done:
            self._send_human(human_msg)
        else:
            self.reset_and_start_rep()

        # Command is at 10Hz
        time.sleep(0.1)


    def _send_vicon(self, movement: Adapted):
        """Send Vicon reading."""

        if not self._pub_vel_sent:
            self._logger.info("Sent: Velocity")
            self._pub_vel_sent = True

        self._logger.debug(f"Msg: {movement}")
        self._pub_vel.publish(movement)

        # Send to Influx
        try:
            writeEntry(
                self.influx_client,
                "/vicon_sensor",
                [
                    ("tx", movement.tx),
                    ("ty", movement.ty),
                    ("tz", movement.tz),
                    ("vx", movement.vx),
                    ("vy", movement.vy),
                    ("vz", movement.vz),
                    ("roll", movement.roll),
                    ("pitch", movement.pitch),
                    ("yaw", movement.yaw),
                    ("wx", movement.wx),
                    ("wy", movement.wy),
                    ("wz", movement.wz),
                ],
            )
        except BaseException as e:
            self._logger.warning("Warning: Influx sending failed.")
            self._logger.warning(e)

    def _send_human(self, movement: HumanControl):
        """Send human command."""

        if not self._pub_human_sent:
            self._logger.info("Sent: HumanWaypoint")
            self._pub_human_sent = True
        self._logger.debug(f"Human Msg: {movement}")
        self._pub_human.publish(movement)

        # Send to Influx
        try:
            writeEntry(
                self.influx_client,
                "/vicon_target",
                [
                    ("target_id", self.target),
                    ("target_tx", self.path[self.target][0]),
                    ("target_ty", self.path[self.target][1]),
                ],
            )
        except BaseException as e:
            self._logger.warning("Warning: Influx sending failed.")
            self._logger.warning(e)

    def _send_system_control(self, adaptation_reset=False, adaptation_on=ADAPTATION_ON):
        """Create and send a system control message."""

        if not self._pub_system_control_sent:
            self._logger.info("Sent: SystemControl")
            self._pub_system_control_sent = True

        if adaptation_reset:
            self._logger.info("Sending RESET signal to adaptation.")

        # Create message
        status_msg = SystemControl()
        status_msg.stamp = self.get_clock().now().to_msg()
        status_msg.adaptation_on = adaptation_on
        status_msg.adaptation_reset = adaptation_reset
        status_msg.max_human_vx = 0.5

        self._pub_system_control.publish(status_msg)

        # Send to Influx
        try:
            writeEntry(
                self.influx_client,
                "/system_control",
                [
                    ("adaptation_on", adaptation_on),
                    ("adaptation_reset", adaptation_reset),
                    ("max_human_vx", 0.5),
                ],
            )
        except BaseException as e:
            self._logger.warning("Warning: Influx sending failed.")
            self._logger.warning(e)

    def _send_perturbation(
        self,
        scaling: bool,
        scaling_side: str,
        scaling_amplitude: float,
        dynamic_scaling: bool,
        dynamic_scaling_amplitude: float,
        dynamic_scaling_interval: float,
        offset: bool,
        wind: bool,
        bernoulli: bool,
    ):
        """Create and send a perturbation message."""

        if not self._pub_perturbation_sent:
            self._logger.info("Sent: Perturbation")
            self._pub_perturbation_sent = True

        perturbation_msg = Perturbation()

        # Default values
        perturbation_msg.left_scale = 1.0
        perturbation_msg.right_scale = 1.0
        perturbation_msg.dynamic_scale_amplitude = 1.0
        perturbation_msg.dynamic_scale_interval = 0.0
        perturbation_msg.left_offset = 0.0
        perturbation_msg.right_offset = 0.0

        # Set scaling
        if scaling:
            if scaling_side == "left":
                perturbation_msg.left_scale = scaling_amplitude
            elif scaling_side == "right":
                perturbation_msg.right_scale = scaling_amplitude
            else:
                self._logger.warning("WARNING: unknown perturbation side.")

        # Set dynamic scaling
        if dynamic_scaling:
            perturbation_msg.dynamic_scale_amplitude = dynamic_scaling_amplitude
            perturbation_msg.dynamic_scale_interval = dynamic_scaling_interval

        # Set offset
        if offset:
            perturbation_msg.left_offset = 0.0 
            perturbation_msg.right_offset = 0.0 

        # Set wind and bernoulli
        perturbation_msg.wind = wind
        perturbation_msg.bernoulli = bernoulli

        # Publish Perturbation msg
        self._pub_perturbation.publish(perturbation_msg)

        # Send Perturbation to Influx
        try:
            writeEntry(
                self.influx_client,
                "/set_perturbation",
                [
                    ("left_scale", perturbation_msg.left_scale),
                    ("right_scale", perturbation_msg.right_scale),
                    ("dynamic_scale_amplitude", perturbation_msg.dynamic_scale_amplitude),
                    ("dynamic_scale_interval", perturbation_msg.dynamic_scale_interval),
                    ("left_offset", perturbation_msg.left_offset),
                    ("right_offset", perturbation_msg.right_offset),
                    ("bernoulli", perturbation_msg.bernoulli),
                    ("wind", perturbation_msg.wind),
                ],
            )
        except BaseException as e:
            self._logger.warning("Warning: Influx sending failed.")
            self._logger.warning(e)


    def get_speed(self, movement: Position):
        """Extract current linear and angular velocity estimates 
        from vicon reading send through corresponding topics."""

        # Get quaternions from current vicon reading
        quaternion = np.asarray(
            [movement.x_rot, movement.y_rot, movement.z_rot, movement.w]
        )

        # Get position
        try:
            r = R.from_quat(quaternion)
        except Exception:
            return np.asarray([np.nan, np.nan, np.nan]), np.asarray(
                [np.nan, np.nan, np.nan]
            )

        current_position = np.asarray(
            [movement.x_trans, movement.y_trans, movement.z_trans]
        )
        current_rot = r.as_euler("XYZ", degrees=True)  # extrinsic

        frame_duration = 1 / 100
        current_frame = movement.frame_number
        if self.previous_position is None:
            self.previous_position = current_position
            self.previous_rot = current_rot
            self.previous_quat = quaternion
            self.previous_frame = current_frame

        # Linear velocity
        linear_velocity = current_position - self.previous_position
        linear_velocity = r.apply(linear_velocity, inverse=True) / (
            (current_frame - self.previous_frame) * frame_duration * 1000
        )

        # Angular velocity
        angular_velocity = current_rot - self.previous_rot
        if angular_velocity[2] > 180:
            angular_velocity[2] -= 360
        elif angular_velocity[2] < -180:
            angular_velocity[2] += 360
        angular_velocity = r.apply(angular_velocity, inverse=True) / (
            (current_frame - self.previous_frame) * frame_duration
        )

        # Updating Position
        self.previous_position = current_position
        self.previous_rot = current_rot
        self.previous_frame = current_frame

        # Moving average
        self.lin_vels.append(linear_velocity)
        self.ang_vels.append(angular_velocity)
        self.previous_quat = quaternion

        if len(self.lin_vels) > 2:
            return np.average(self.lin_vels, axis=0), np.average(self.ang_vels, axis=0)
        else:
            return np.average(self.lin_vels, axis=0), np.average(self.ang_vels, axis=0)


    def follow_path(self, current_position: Position):
        """Create the path tracking command."""
        
        if self.path is None:
            return HumanControl()
        else:
            x, y = self.path[self.target]

        # Get current x, y position and orientation
        x_pos = current_position.x_trans
        y_pos = current_position.y_trans
        quaternion = np.asarray(
            [
                current_position.x_rot,
                current_position.y_rot,
                current_position.z_rot,
                current_position.w,
            ]
        )

        # Transform quaternion
        try:
            r = R.from_quat(quaternion)
        except Exception:
            self._logger.error(f"Quaternion conversion did not work properly.")
            return

        # Compute the error
        error_x = (x - x_pos) / 1000
        error_y = (y - y_pos) / 1000
        error_robot_frame = r.apply(np.asarray([error_x, error_y, 0]), inverse=True)
        angle_heading = np.arctan2(error_robot_frame[1], error_robot_frame[0])
        distance = np.linalg.norm(error_robot_frame)

        # SAFETY: Update smallest distance so far
        if distance < self._smaller_distance_so_far:
            self._smaller_distance_so_far = distance

        # If already at target, go to next target
        if distance < self.error_threshold:

            # If done with the path, print it
            if self.target == (len(self.path) - 1):
                new_msg = HumanControl()
                if not self.done:
                    self._send_system_control(adaptation_reset=True)
                    self.done = True
                    time.sleep(0.5)
                    self._logger.info("DONE with driver - case 1")
                self._logger.info("DONE with driver - case 2")
                return new_msg

            self.target += 1
            self._logger.info(
                f"Next target: {self.target}, at: {self.path[self.target]}"
            )

            # SAFETY: Reset smallest distance so far for this target
            self._smaller_distance_so_far = np.inf

            return self.follow_path(current_position)

        # SAFETY: if break safety condition, stop
        # if distance > self._smaller_distance_so_far + 1.0:
        #     self._logger.info("SAFETY CONSTRAINTS DONE with driver")
        #     if not self.done:
        #         self._send_system_control(adaptation_reset=True)
        #         self.done = True
        #         time.sleep(0.5)
        #         self._logger.info("SAFETY CONSTRAINTS DONE with driver - case 1")
        #     self.done_safety = True
        #     return HumanControl()

        # Compute the new vx command
        if -np.pi / 4 > angle_heading or angle_heading > np.pi / 4:
            # If need reorientation, no vx
            v_lin = 0.0
        else:
            # Else vx proportional to distance
            v_lin = np.clip(0.5 * distance, MIN_DRIVER_SPEED, MAX_DRIVER_SPEED)

        # Compute the new wz command
        wz = np.clip(1.7 * angle_heading, -0.7, 0.7)

        # Create HumanControl msg
        new_msg = HumanControl()
        new_msg.joystick.linear.x = v_lin
        new_msg.joystick.angular.z = -wz
        new_msg.flipper.angular.x = 0.0
        return new_msg


def main():
    import argparse
    import os
    import socket
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l", "--log", type=str, default="INFO", help="Log level to use"
    )

    # The code can't handle a name, only IP addresses, so do the mapping
    rds = os.getenv("ROS_DISCOVERY_SERVER")
    oldros, port = rds.split(":")
    try:
        int(oldros.split(".")[0])
    except Exception:
        rosname = socket.gethostbyname(oldros)
        os.environ["ROS_DISCOVERY_SERVER"] = f"{rosname}:{port}"

    print(f"Vicon: going to connect to {os.getenv('ROS_DISCOVERY_SERVER')}", flush=True)

    newargs = remove_ros_args(sys.argv)
    args = parser.parse_args(args=newargs[1:])

    rclpy.init(args=sys.argv)

    vicon = Vicon(args.log)

    rclpy.spin(vicon)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    vicon.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
