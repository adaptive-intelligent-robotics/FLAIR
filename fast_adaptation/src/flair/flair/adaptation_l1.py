#!/usr/bin/env python3
# from influxdb_client.client.write_api import SYNCHRONOUS
import logging
import multiprocessing
import os
import time
from copy import deepcopy
from typing import Tuple

import numpy as np
import rclpy
from FLAIR_msg.msg import (
    FLAIRTwist,
    FunctionalityControllerControl,
    HumanControl,
    Perturbation,
    SystemControl,
)
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from influxdb_client import InfluxDBClient, Point, WriteOptions
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Imu

# Defining a context to handle CUDA ressources anf Jax memory usage
# Choose forkserver as faster than spawn
context = multiprocessing.get_context("forkserver")

# Import the config
from src.flair.flair.adaptation_config_l1 import *

####################
# Common functions #


class MyStreamHandler(logging.StreamHandler):
    def flush(self) -> None:
        self.stream.flush()


def define_logger(logname: str, loglevel: str, header: str) -> logging.Logger:
    """Define a logger."""

    llevel = getattr(logging, loglevel.upper())
    if not isinstance(llevel, int):
        raise ValueError(f"Invalid log level: {loglevel}")

    formatter = logging.Formatter(
        fmt=header + "%(asctime)s.%(msecs)03d %(name)s %(message)s" + " \x1b[0m ",
        datefmt="%s",
    )
    sh = MyStreamHandler()
    sh.setLevel(llevel)
    sh.setFormatter(formatter)
    logger = logging.getLogger(logname)
    logger.setLevel(llevel)
    logger.addHandler(sh)

    return logger


def init_influx() -> InfluxDBClient:
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
    influx_client: InfluxDBClient,
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
        _ = influx_client.write(
            record=pt, org="FLAIR", bucket=os.environ["INFLUXDB_BUCKET"]
        )
    except BaseException as e:
        print(e, flush=True)
        pass


#############################
# Sensor Collection process #


class SensorCollection(Node):
    def __init__(
        self,
        sensor_collection_to_rl_collection_datapoint_queue: multiprocessing.Queue,
        sensor_collection_to_adaptation_datapoint_queue: multiprocessing.Queue,
        sensor_collection_to_rl_training_reset_queue: multiprocessing.Queue,
        sensor_collection_to_rl_collection_reset_queue: multiprocessing.Queue,
        rl_collection_to_sensor_collection_start_queue: multiprocessing.Queue,
        loglevel: str = "INFO",
    ) -> None:

        super().__init__("sensorcollection")

        # Import inside process to reduce memory usage
        import os

        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.01"

        # Set up logging
        self._logger = define_logger(
            "SensorCollection", loglevel, "\x1b[34;20m Sensor_COLLECTION"
        )
        self._logger.info("SensorCollection: Starting")

        # Store Queues as attribute
        self.sensor_collection_to_rl_collection_datapoint_queue = (
            sensor_collection_to_rl_collection_datapoint_queue
        )
        self.sensor_collection_to_adaptation_datapoint_queue = (
            sensor_collection_to_adaptation_datapoint_queue
        )
        self.sensor_collection_to_rl_training_reset_queue = (
            sensor_collection_to_rl_training_reset_queue
        )
        self.sensor_collection_to_rl_collection_reset_queue = (
            sensor_collection_to_rl_collection_reset_queue
        )
        self.rl_collection_to_sensor_collection_start_queue = (
            rl_collection_to_sensor_collection_start_queue
        )

        # Necessary attributes
        self.start_collection = False

        # Listen to reset message
        self._sub_system_control = self.create_subscription(
            SystemControl,
            "/system_control",
            self._system_control_callback,
            qos_profile_sensor_data,
        )
        self._sub_system_control_seen = False

        # Collect sensor data from IMU
        self._sub_imu = self.create_subscription(
            Imu, "/vectornav/imu", self._imu_callback, qos_profile_sensor_data
        )
        self._sub_imu_seen = False

        # Collect sensor data from Zed
        self._sub_zed_f = self.create_subscription(
            PoseWithCovarianceStamped,
            "/zed/forward/pose_with_covariance",
            self._zed_callback,
            qos_profile_sensor_data,
        )
        self._sub_zed_f_seen = False
        self.f_cam_dt = None

        # Initialise attributes for sensory Data
        self.euler_angles = np.zeros((3,))  # roll, pitch, yaw
        self.w = np.inf * np.zeros((3,))  # x, y, z
        self.v = np.inf * np.zeros((3,))  # x, y, z
        self.last_timestamp_zed = None
        self.last_timestamp_imu = None
        self.global_position = np.zeros((3,))

        # Attributes for Zed-to-Robot Transform
        forward_angles = [1.280, 28.417, 0]
        forward_position = [0.247, 0.0595, 0.273]
        self.forward_transform = np.eye(4)
        self.forward_init = np.eye(4)
        self.forward_transform[0:3] = np.hstack(
            (
                R.from_euler("xyz", forward_angles, degrees=True).as_matrix(),
                np.array(forward_position).reshape(3, 1),
            )
        )
        self.forward_init[0:3] = np.hstack(
            (
                R.from_euler(
                    "xyz", [0, 0, forward_angles[2]], degrees=True
                ).as_matrix(),
                np.array(forward_position).reshape(3, 1),
            )
        )
        self.forward_transform_inv = np.linalg.inv(self.forward_transform)

        self._logger.info(
            "SensorCollection: Completed __init__, waiting for RLCollection to start."
        )

    def _imu_callback(self, msg: Imu) -> None:
        """Get IMU data and send everything to other processes."""

        # Get latest sensor
        self.euler_angles = R.from_quat(
            [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        ).as_euler("xyz")
        self.w[0] = msg.angular_velocity.x
        self.w[1] = msg.angular_velocity.y
        self.w[2] = msg.angular_velocity.z  # self.lp_ang_filter(msg.angular_velocity.z)
        self.sensor_time = msg.header.stamp

        # Store latest IMU timestamp
        time_msg = self.get_clock().now()
        self.last_timestamp_imu = time_msg.nanoseconds / 10**9

        # Send via Queue to all other processes
        self._send_sensor()

    def _zed_callback(self, msg: PoseWithCovarianceStamped) -> None:
        """Get Zed data and store the outcome."""

        # Get latest reading
        current_pose = np.eye(4)
        mat = R.from_quat(
            [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ]
        ).as_matrix()
        current_pose[0:3] = np.hstack(
            (
                mat,
                np.array(
                    [
                        msg.pose.pose.position.x,
                        msg.pose.pose.position.y,
                        msg.pose.pose.position.z,
                    ]
                ).reshape(3, 1),
            )
        )

        # Transform
        old_position = self.global_position
        self.global_position = (
            self.forward_init @ current_pose @ self.forward_transform_inv
        )
        new_quat = R.from_matrix(self.global_position[:3, :3]).as_quat()
        self.global_position = self.global_position[:3, 3]
        camera_frame_transform = R.from_quat(new_quat).inv().as_matrix()

        # camera_frame_transform = (
        #     R.from_quat(
        #         [
        #             msg.pose.pose.orientation.x,
        #             msg.pose.pose.orientation.y,
        #             msg.pose.pose.orientation.z,
        #             msg.pose.pose.orientation.w,
        #         ]
        #     )
        #     .inv()
        #     .as_matrix()
        # )
        # old_position = self.global_position
        # self.global_position = np.array(
        #     [
        #         msg.pose.pose.position.x,
        #         msg.pose.pose.position.y,
        #         msg.pose.pose.position.z,
        #     ]
        # )

        current_timestamp = msg.header.stamp.sec + (msg.header.stamp.nanosec / 10**9)
        if self.last_timestamp_zed is None:
            self.v = np.zeros((3,))
        else:
            dt = current_timestamp - self.last_timestamp_zed
            if dt > 0 and dt < 0.5:
                word_frame_vel = (self.global_position - old_position) / dt
                self.v = camera_frame_transform @ word_frame_vel

            else:
                self._logger.info("WARNING: Shouldn't be here, if this is not a reset, something is wrong.")
                self.v = np.zeros((3,))
        self.last_timestamp_zed = current_timestamp

    def _system_control_callback(self, msg: SystemControl) -> None:
        """Reset the RL if asked by system topic."""
        if not self._sub_system_control_seen:
            self._logger.info("Present: SystemControl")
            self._sub_system_control_seen = True

        # This class is only interested in the reset info
        if msg.adaptation_reset:
            self._logger.debug("RESETING SENSOR DATASET " * 20)

            # Empty datapoint queue
            n_datapoint = 0
            try:
                for datapoint in iter(
                    self.sensor_collection_to_rl_collection_datapoint_queue.get_nowait,
                    "STOP",
                ):
                    n_datapoint += 1
            except BaseException:
                pass
            self._logger.debug(
                f"EMPTY {n_datapoint} from sensor_collection_to_rl_collection_datapoint_queue"
            )

            # Reseting RL Collection
            self.sensor_collection_to_rl_collection_reset_queue.put((True))

            # Empty reset queue
            try:
                self.sensor_collection_to_rl_training_reset_queue.get_nowait()
            except BaseException:
                pass
            self._logger.debug(
                "EMPTY sensor_collection_to_rl_training_reset_queue and reset RLTraining"
            )

            # Reseting RL
            self.sensor_collection_to_rl_training_reset_queue.put((True))

    def _send_sensor(self) -> None:
        """Send sensor data to other processes."""

        # Get current reading
        sensor_tx = self.global_position[0]
        sensor_ty = self.global_position[1]
        sensor_tz = self.global_position[2]
        sensor_vx = self.v[0]
        sensor_vy = self.v[1]
        sensor_vz = self.v[2]

        sensor_wx = self.w[0]
        sensor_wy = self.w[1]
        sensor_wz = self.w[2] * -1  # Warning: inverting z-axis here

        sensor_roll = self.euler_angles[0]
        sensor_pitch = self.euler_angles[1]
        sensor_yaw = self.euler_angles[2]

        # Compute state
        state = (
            sensor_vx,
            sensor_vy,
            sensor_wz,
            sensor_tx,
            sensor_ty,
            sensor_yaw,
            sensor_roll,
            sensor_pitch,
        )

        # Send datapoint to RLCollection
        self.sensor_collection_to_rl_collection_datapoint_queue.put(
            (
                self.sensor_time,
                state,
                sensor_tx,
                sensor_ty,
                sensor_tz,
                sensor_vx,
                sensor_vy,
                sensor_vz,
                sensor_yaw,
                sensor_roll,
                sensor_pitch,
                sensor_wx,
                sensor_wy,
                sensor_wz,
            )
        )

        # Empty datapoint to Adaptation queue
        try:
            self.sensor_collection_to_adaptation_datapoint_queue.get_nowait()
        except BaseException:
            pass

        # Send datapoint to Adaptation
        self.sensor_collection_to_adaptation_datapoint_queue.put((state, sensor_vx, sensor_wz))


# SensorCollection process
def start_sensor_collection_process(
    args,
    loglevel: str,
    sensor_collection_to_rl_collection_datapoint_queue: multiprocessing.Queue,
    sensor_collection_to_adaptation_datapoint_queue: multiprocessing.Queue,
    sensor_collection_to_rl_training_reset_queue: multiprocessing.Queue,
    sensor_collection_to_rl_collection_reset_queue: multiprocessing.Queue,
    rl_collection_to_sensor_collection_start_queue: multiprocessing.Queue,
) -> None:
    """Process to acquire sensor information to send them for point selection."""

    rclpy.init(args=args)
    sensor_collection = SensorCollection(
        sensor_collection_to_rl_collection_datapoint_queue,
        sensor_collection_to_adaptation_datapoint_queue,
        sensor_collection_to_rl_training_reset_queue,
        sensor_collection_to_rl_collection_reset_queue,
        rl_collection_to_sensor_collection_start_queue,
        loglevel,
    )
    rclpy.spin(sensor_collection)
    sensor_collection.destroy_node()


#########################
# RL Collection process #


def start_rl_collection_process(
    args,
    loglevel: str,
    adaptation_to_rl_collection_command_queue: multiprocessing.Queue,
    sensor_collection_to_rl_collection_datapoint_queue: multiprocessing.Queue,
    rl_collection_to_rl_training_datapoint_queue: multiprocessing.Queue,
    rl_training_to_rl_collection_start_queue: multiprocessing.Queue,
    rl_collection_to_sensor_collection_start_queue: multiprocessing.Queue,
    sensor_collection_to_rl_collection_reset_queue: multiprocessing.Queue,
) -> None:
    """Process for the RL Data Collection."""

    # Import inside process to reduce memory usage
    import os

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.01"
    # os.environ['JAX_PLATFORMS']='cpu'
    from functionality_controller.data_collection import DataCollection, DataQuality

    # Init Influx Client
    influx_client = init_influx()

    # Set up logging
    logger = define_logger("RLCollection", loglevel, "\x1b[31;20m RL_COLLECTION")
    logger.info("RLCollection: Starting")

    # Create the DataCollection object
    data_collection = DataCollection(
        logger=logger,
        filter_transition=FILTER_TRANSITION,
        filter_varying_angle=FILTER_VARYING_ANGLE,
        filter_turning_only=FILTER_TURNING_ONLY,
        buffer_size=BUFFER_SIZE,
        min_delay=MIN_DELAY,
        max_delay=MAX_DELAY,
        selection_size=SELECTION_SIZE,
        IQC_Q1=IQC_Q1,
        IQC_Qn=IQC_Qn,
        filter_transition_size=FILTER_TRANSITION_SIZE,
        filter_transition_tolerance=FILTER_TRANSITION_TOLERANCE,
        filter_varying_angle_size=FILTER_VARYING_ANGLE_SIZE,
        filter_varying_angle_tolerance=FILTER_VARYING_ANGLE_TOLERANCE,
        filter_turning_only_tolerance=FILTER_TURNING_ONLY_TOLERANCE,
    )

    # Attributes to compute datapoint quality metric
    datapoints_number = 0
    sensor_datapoints_number = 0
    data_collection_calls = 0
    data_collection_time = 0

    # Send DataCollection configuration to Influx
    try:
        writeEntry(
            influx_client,
            "/l1_collection_configuration",
            [
                ("Filter_transition", FILTER_TRANSITION),
                ("Filter_varying_angle", FILTER_VARYING_ANGLE),
                ("Filter_turning_only", FILTER_TURNING_ONLY),
                ("Buffer_size", BUFFER_SIZE),
                ("Min_delay", MIN_DELAY),
                ("Max_delay", MAX_DELAY),
                ("Selection_size", SELECTION_SIZE),
                ("IQC_Q1", IQC_Q1),
                ("IQC_Qn", IQC_Qn),
                ("Filter_transition_size", FILTER_TRANSITION_SIZE),
                ("Filter_transition_tolerance", FILTER_TRANSITION_TOLERANCE),
                ("Filter_varying_angle_size", FILTER_VARYING_ANGLE_SIZE),
                ("Filter_varying_angle_tolerance", FILTER_VARYING_ANGLE_TOLERANCE),
                ("Filter_turning_only_tolerance", FILTER_TURNING_ONLY_TOLERANCE),
            ],
        )
    except BaseException as e:
        logger.warning("Warning: Influx sending failed.")
        logger.warning(e)

    # Waiting for RLTraining to start to avoid continuous offset between collection and training
    #logger.info("RLCollection: Completed __init__, waiting for RLTraining to start.")
    #rl_training_to_rl_collection_start_queue.get()
    #logger.debug("RLTraining has started, waiting for commands to start collection.")

    # Waiting for the first command to avoid stacking sensor datapoints that do not match any command
    (
        command_x,
        command_y,
        rl_prediction_x,
        rl_prediction_y,
        intent_x,
        intent_y,
    ) = adaptation_to_rl_collection_command_queue.get()
    sensor_vx = None
    sensor_wz = None
    next_command_x = None

    # When got the first command, trigger sensor collection
    logger.debug("Got the first command, triggering Sensor Collection.")
    rl_collection_to_sensor_collection_start_queue.put((True))
    logger.debug("Starting RLCollection.")

    # Main loop
    while True:

        # Get a datapoint from Sensor Collection
        sensor_datapoint = sensor_collection_to_rl_collection_datapoint_queue.get()
        if sensor_vx is None:
            sensor_time = np.array([[sensor_datapoint[0]]])
            state = np.array([sensor_datapoint[1]])
            sensor_tx = np.array([[sensor_datapoint[2]]])
            sensor_ty = np.array([[sensor_datapoint[3]]])
            sensor_tz = np.array([[sensor_datapoint[4]]])
            sensor_vx = np.array([[sensor_datapoint[5]]])
            sensor_vy = np.array([[sensor_datapoint[6]]])
            sensor_vz = np.array([[sensor_datapoint[7]]])
            sensor_yaw = np.array([[sensor_datapoint[8]]])
            sensor_roll = np.array([[sensor_datapoint[9]]])
            sensor_pitch = np.array([[sensor_datapoint[10]]])
            sensor_wx = np.array([[sensor_datapoint[11]]])
            sensor_wy = np.array([[sensor_datapoint[12]]])
            sensor_wz = np.array([[sensor_datapoint[13]]])

            # Initialise a timer to ensure not acquiring datapoints forever
            overflow_timer = time.time()
        else:
            sensor_time = np.append(sensor_time, [[sensor_datapoint[0]]], axis=0)
            state = np.append(state, [sensor_datapoint[1]], axis=0)
            sensor_tx = np.append(sensor_tx, [[sensor_datapoint[2]]], axis=0)
            sensor_ty = np.append(sensor_ty, [[sensor_datapoint[3]]], axis=0)
            sensor_tz = np.append(sensor_tz, [[sensor_datapoint[4]]], axis=0)
            sensor_vx = np.append(sensor_vx, [[sensor_datapoint[5]]], axis=0)
            sensor_vy = np.append(sensor_vy, [[sensor_datapoint[6]]], axis=0)
            sensor_vz = np.append(sensor_vz, [[sensor_datapoint[7]]], axis=0)
            sensor_yaw = np.append(sensor_yaw, [[sensor_datapoint[8]]], axis=0)
            sensor_roll = np.append(sensor_roll, [[sensor_datapoint[9]]], axis=0)
            sensor_pitch = np.append(sensor_pitch, [[sensor_datapoint[10]]], axis=0)
            sensor_wx = np.append(sensor_wx, [[sensor_datapoint[11]]], axis=0)
            sensor_wy = np.append(sensor_wy, [[sensor_datapoint[12]]], axis=0)
            sensor_wz = np.append(sensor_wz, [[sensor_datapoint[13]]], axis=0)

        # Try to get all sensor_datapoints piled in the queue
        n_sensor_datapoints = 1
        start_t = time.time()
        try:
            for sensor_datapoint in iter(
                sensor_collection_to_rl_collection_datapoint_queue.get_nowait, "STOP"
            ):

                # start_add = time.time()
                sensor_time = np.append(sensor_time, [[sensor_datapoint[0]]], axis=0)
                state = np.append(state, [sensor_datapoint[1]], axis=0)
                sensor_tx = np.append(sensor_tx, [[sensor_datapoint[2]]], axis=0)
                sensor_ty = np.append(sensor_ty, [[sensor_datapoint[3]]], axis=0)
                sensor_tz = np.append(sensor_tz, [[sensor_datapoint[4]]], axis=0)
                sensor_vx = np.append(sensor_vx, [[sensor_datapoint[5]]], axis=0)
                sensor_vy = np.append(sensor_vy, [[sensor_datapoint[6]]], axis=0)
                sensor_vz = np.append(sensor_vz, [[sensor_datapoint[7]]], axis=0)
                sensor_yaw = np.append(sensor_yaw, [[sensor_datapoint[8]]], axis=0)
                sensor_roll = np.append(sensor_roll, [[sensor_datapoint[9]]], axis=0)
                sensor_pitch = np.append(sensor_pitch, [[sensor_datapoint[10]]], axis=0)
                sensor_wx = np.append(sensor_wx, [[sensor_datapoint[11]]], axis=0)
                sensor_wy = np.append(sensor_wy, [[sensor_datapoint[12]]], axis=0)
                sensor_wz = np.append(sensor_wz, [[sensor_datapoint[13]]], axis=0)

                n_sensor_datapoints += 1
                # logger.debug(f"Time to add one sensor_datapoint: {time.time() - start_add}")

        except BaseException:
            # logger.debug(e)
            # logger.debug(f"Got {n_sensor_datapoints} sensor_datapoints")
            pass

        # Increase metric counter
        # logger.debug(f"Time to get {n_sensor_datapoints} sensor_datapoints: {time.time() - start_t}")
        sensor_datapoints_number += n_sensor_datapoints

        # Send sensor information to Influx
        try:
            for n_sensor_datapoints in range(0, state.shape[0]):
                timestamp = sensor_time[n_sensor_datapoints, 0]
                timestamp = int(timestamp.sec) * int(1e9) + int(timestamp.nanosec)
                writeEntry(
                    influx_client,
                    "/l1_sensor",
                    [
                        ("time", timestamp),
                        ("tx", sensor_tx[n_sensor_datapoints, 0]),
                        ("ty", sensor_ty[n_sensor_datapoints, 0]),
                        ("tz", sensor_tz[n_sensor_datapoints, 0]),
                        ("vx", sensor_vx[n_sensor_datapoints, 0]),
                        ("vy", sensor_vy[n_sensor_datapoints, 0]),
                        ("vz", sensor_vz[n_sensor_datapoints, 0]),
                        ("yaw", sensor_yaw[n_sensor_datapoints, 0]),
                        ("roll", sensor_roll[n_sensor_datapoints, 0]),
                        ("pitch", sensor_pitch[n_sensor_datapoints, 0]),
                        ("wx", sensor_wx[n_sensor_datapoints, 0]),
                        ("wy", sensor_wy[n_sensor_datapoints, 0]),
                        ("wz", sensor_wz[n_sensor_datapoints, 0]),
                    ],
                    timestamp=timestamp,
                )
        except BaseException as e:
            logger.warning("Warning: Influx sending failed.")
            logger.warning(e)

        # Attempt to get command from Adaptation
        try:
            (
                next_command_x,
                next_command_y,
                next_rl_prediction_x,
                next_rl_prediction_y,
                next_intent_x,
                next_intent_y,
            ) = adaptation_to_rl_collection_command_queue.get_nowait()

        except BaseException as e:
            pass

        # If have been acquiring only sensor datapoints for more than 3secs, wait for command to restart
        if next_command_x is None and (time.time() - overflow_timer) > 3:

            logger.info("Entering waiting mode: no more command going through.")
            next_command_x = None
            sensor_vx = None
            n_sensor_datapoints = 0

            # Waiting for command
            (
                command_x,
                command_y,
                rl_prediction_x,
                rl_prediction_y,
                intent_x,
                intent_y,
            ) = adaptation_to_rl_collection_command_queue.get()

            # Empty datapoint queue
            logger.debug("Got a new command, emptying Sensor Collection queue.")
            n_datapoint = 0
            try:
                for datapoint in iter(
                    self.sensor_collection_to_rl_collection_datapoint_queue.get_nowait,
                    "STOP",
                ):
                    n_datapoint += 1
            except BaseException:
                pass
            logger.info("Exiting waiting mode.")

        # Attempt to get reset signal from Sensor Collection process
        try:
            _ = sensor_collection_to_rl_collection_reset_queue.get_nowait()
            logger.debug("RESETING RL Collection" * 20)

            # Reset Data Collection
            data_collection.reset()
            datapoints_number = 0
            sensor_datapoints_number = 0
            data_collection_calls = 0
            data_collection_time = 0

            command_x = next_command_x
            command_y = next_command_y
            rl_prediction_x = next_rl_prediction_x
            rl_prediction_y = next_rl_prediction_y
            intent_x = next_intent_x
            intent_y = next_intent_y
            next_command_x = None
            sensor_vx = None

            # Empty datapoint queue
            n_datapoint = 0
            try:
                for datapoint in iter(
                    rl_collection_to_rl_training_datapoint_queue.get_nowait, "STOP"
                ):
                    n_datapoint += 1
            except BaseException:
                pass
            logger.debug(
                f"EMPTY {n_datapoint} from rl_collection_to_rl_training_datapoint_queue"
            )

        except BaseException:
            pass

        # If got the next command, call the DataCollection
        if next_command_x is not None:

            # Call the DataCollection object
            try:
                start_t = time.time()
                (
                    final_datapoint,
                    final_datapoints_msg,
                    metrics,
                    error_code,
                ) = data_collection.data_collection(
                    state=state,
                    command_x=command_x,
                    command_y=command_y,
                    gp_prediction_x=rl_prediction_x,
                    gp_prediction_y=rl_prediction_y,
                    intent_x=intent_x,
                    intent_y=intent_y,
                    sensor_time=sensor_time,
                    sensor_x=sensor_vx,
                    sensor_y=sensor_wz,
                    sensor_wx=sensor_wx,
                    sensor_wy=sensor_wy,
                )
                collection_time = time.time() - start_t
                # logger.debug(f"Data collection took {collection_time}")

            except BaseException as e:
                final_datapoint = None
                final_datapoints_msg = None
                metrics = None
                collection_time = 0
                logger.warning(f"Data Collection call failed.")
                logger.warning(e)

            # Send to Influx
            try:
                writeEntry(
                    influx_client, "/l1_datapoint_filter", [("filter_code", error_code)]
                )
            except BaseException as e:
                logger.warning("Warning: Influx sending failed.")
                logger.warning(e)

            # Update metrics
            data_collection_calls += 1
            data_collection_time += collection_time

            # Store next commands
            command_x = next_command_x
            command_y = next_command_y
            rl_prediction_x = next_rl_prediction_x
            rl_prediction_y = next_rl_prediction_y
            intent_x = next_intent_x
            intent_y = next_intent_y

            # Update attributes used by conditions
            next_command_x = None
            sensor_vx = None

            # If Data Collection returned a datapoint, send it to the RL
            if final_datapoint is not None:

                # Send to RLTraining
                # logger.debug(f"Sending 1 datapoint to RL Training.")
                rl_collection_to_rl_training_datapoint_queue.put(final_datapoint)
                datapoints_number += 1

                # Send to Influx
                try:
                    influx_msgs = final_datapoints_msg
                    writeEntry(
                        influx_client,
                        "/l1_datapoint",
                        influx_msgs,
                        timestamp=int(final_datapoint[1]) * int(1e9)
                        + int(final_datapoint[2]),
                    )
                except BaseException as e:
                    logger.warning("Warning: Influx sending failed.")
                    logger.warning(e)

            # If Data Collection returned metrics for Influx, send them to Influx
            if DEBUG_RL_COLLECTION:
                if metrics is not None:
                    try:
                        writeEntry(
                            influx_client,
                            "/l1_datapoint_analysis",
                            [("DataCollection", metrics)],
                        )
                    except BaseException as e:
                        logger.debug("Warning: Influx sending failed.")
                        logger.debug(e)


#######################
# RL training process #

# RLtraining process
def start_rl_training_process(
    loglevel: str,
    rl_training_to_adaptation_model_queue: multiprocessing.Queue,
    rl_collection_to_rl_training_datapoint_queue: multiprocessing.Queue,
    rl_training_to_rl_collection_start_queue: multiprocessing.Queue,
    rl_training_to_adaptation_start_queue: multiprocessing.Queue,
    sensor_collection_to_rl_training_reset_queue: multiprocessing.Queue,
) -> None:
    """Thread for the RL Training."""

    # Create a logger
    logger = define_logger("RLTraining", loglevel, "\x1b[32;20m TRAINER ")
    # logging.basicConfig(level=logging.DEBUG)

    # Import inside process to reduce memory usage
    import os

    # os.environ['JAX_PLATFORMS']='cuda'
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.50"
    from functionality_controller.datapoint import DataPoints
    from functionality_controller.functionality_controller_utils import (
        compute_borders,
        compute_clipping,
        compute_metric_map,
    )

    # Init Influx Client
    influx_client = init_influx()

    # Create empty DataPoints for addition
    empty_datapoints = DataPoints.init(max_capacity=DATAPOINT_BATCH_SIZE, state_dims=8)
    jiting_datapoints = DataPoints.add(
        datapoint=empty_datapoints,
        point_id=0,
        sensor_time_sec=0.0,
        sensor_time_nanosec=0.0,
        command_time_sec=0.0,
        command_time_nanosec=0.0,
        state=np.zeros(8),
        gp_prediction_x=0.0,
        gp_prediction_y=0.0,
        command_x=0.0,
        command_y=0.0,
        intent_x=0.0,
        intent_y=0.0,
        sensor_x=0.0,
        sensor_y=0.0,
    )

    # Create RL
    logger.info("RLTraining: Starting")

    from functionality_controller.residual_rl_td3 import ResidualTD3

    residual_td3 = ResidualTD3(
        logger=logger,
        jiting_datapoints=jiting_datapoints,
        min_command=MIN_COMMAND, 
        max_command=MAX_COMMAND,
        learning_rate=LEARNING_RATE,
        policy_noise=POLICY_NOISE,
        noise_clip=NOISE_CLIP,
        gamma=GAMMA,
        tau=TAU,
        batch_size=BATCH_SIZE,
        policy_frequency=POLICY_FREQUENCY,
        min_diff_datapoint=MIN_DIFF_DATAPOINT,
        dataset_size=DATASET_SIZE,
        datapoint_batch_size=DATAPOINT_BATCH_SIZE,
        minibatch_size=MINIBATCH_SIZE,
        auto_reset_error_buffer_size=ERROR_BUFFER_SIZE,
        auto_reset_angular_rot_weight=WEIGHT_ANGULAR_ROT,
        auto_reset_threshold=NEW_SCENARIO_THRESHOLD,
    )

    # Define a reset function used multiple time within the loop
    def RLTraining_reset(residual_td3: ResidualTD3, auto_reset: bool) -> None:
        """Function called for any reset of the RL."""

        # Empty model queue
        try:
            rl_training_to_adaptation_model_queue.get_nowait()
        except BaseException:
            pass
        logger.debug("EMPTY model from rl_training_to_adaptation_model_queue")

        # Send new reseted model to Adaptation
        rl_training_to_adaptation_model_queue.put(
            (
                residual_td3.actor_state.params,
            )
        )

        # Empty reset queue
        try:
            for reset in iter(
                sensor_collection_to_rl_training_reset_queue.get_nowait, "STOP"
            ):
                num_train_counter = 0
        except BaseException:
            pass
        logger.debug("EMPTY reset from sensor_collection_to_rl_training_reset_queue")

        # Empty datapoint queue
        n_datapoint = 0
        try:
            for datapoint in iter(
                rl_collection_to_rl_training_datapoint_queue.get_nowait, "STOP"
            ):
                n_datapoint += 1
        except BaseException:
            pass
        logger.debug(
            f"EMPTY {n_datapoint} from rl_collection_to_rl_training_datapoint_queue"
        )

        # Send reset to influx
        try:
            writeEntry(
                influx_client,
                "/l1_reset",
                [("reset", not (auto_reset)), ("auto_reset", auto_reset)],
            )
        except BaseException as e:
            logger.warning("Warning: Influx sending failed.")
            logger.warning(e)

    # Reset everything
    RLTraining_reset(residual_td3, auto_reset=False)
    rl_map_sending_counter = 0

    # Send RLTraining configuration to Influx
    try:
        writeEntry(
            influx_client,
            "/l1_training_configuration",
            [
                ("Use_reset", USE_RESET),
                ("Learning_rate", LEARNING_RATE),
                ("Policy_noise", POLICY_NOISE),
                ("Noise_clip", NOISE_CLIP),
                ("Gamma", GAMMA),
                ("Tau", TAU),
                ("Batch_size", BATCH_SIZE),
                ("Poliy_frequency", POLICY_FREQUENCY),
                ("Observation_state", OBSERVATION_STATE),
                ("Dataset_min_diff", MIN_DIFF_DATAPOINT),
                ("Dataset_batch_size", DATAPOINT_BATCH_SIZE),
                ("Dataset_size", DATASET_SIZE),
                ("Reset_minibatch_size", MINIBATCH_SIZE),
                ("Reset_error_buffer_size", ERROR_BUFFER_SIZE),
                ("Reset_weight_angular_rot", WEIGHT_ANGULAR_ROT),
                ("Reset_new_scenario_threshold", NEW_SCENARIO_THRESHOLD),
            ],
        )
    except BaseException as e:
        logger.warning("Warning: Influx sending failed.")
        logger.warning(e)

    # Trigger the Data Collectiona and Adaptation only when fully started
    # to avoid accumulating datapoints and the robot moving because of damage before adaptation
    logger.debug("Model booted, triggering Data Collection and Adaptation")
    rl_training_to_rl_collection_start_queue.put((True))
    rl_training_to_adaptation_start_queue.put((True))

    logger.info("RLTraining: Completed __init__")

    while True:

        # Wait until we get a datapoint from RLCollection
        start_t = time.time()
        (
            point_id,
            sensor_time_sec,
            sensor_time_nanosec,
            command_time_sec,
            command_time_nanosec,
            rl_prediction_x,
            rl_prediction_y,
            command_x,
            command_y,
            intent_x,
            intent_y,
            sensor_x,
            sensor_y,
            _,
            state,
        ) = rl_collection_to_rl_training_datapoint_queue.get()
        datapoints = DataPoints.add(
            datapoint=empty_datapoints,
            point_id=point_id,
            sensor_time_sec=sensor_time_sec,
            sensor_time_nanosec=sensor_time_nanosec,
            command_time_sec=command_time_sec,
            command_time_nanosec=command_time_nanosec,
            state=state,
            gp_prediction_x=rl_prediction_x,
            gp_prediction_y=rl_prediction_y,
            command_x=command_x,
            command_y=command_y,
            intent_x=intent_x,
            intent_y=intent_y,
            sensor_x=sensor_x,
            sensor_y=sensor_y,
        )
        # logger.debug(f"Time to get datapoint: {time.time() - start_t}")

        # Reset metrics and booleans
        start_loop_t = time.time()
        reset_time = 0.0
        send_map_time = 0.0
        updated = False

        # Attempt to get reset signal from RL collection process
        try:
            _ = sensor_collection_to_rl_training_reset_queue.get_nowait()

            logger.debug("RESETING RL Training " * 20)
            start_t = time.time()
            residual_td3.reset()
            RLTraining_reset(residual_td3, auto_reset=False)
            reset_time = time.time() - start_t
            # logger.debug(f"Time to reset: {time.time() - start_t}")

        except BaseException as e:
            pass

        # Try to get all datapoints piled in the queue
        start_t = time.time()
        n_datapoint = 1
        try:
            for datapoint in iter(
                rl_collection_to_rl_training_datapoint_queue.get_nowait, "STOP"
            ):
                (
                    point_id,
                    sensor_time_sec,
                    sensor_time_nanosec,
                    command_time_sec,
                    command_time_nanosec,
                    rl_prediction_x,
                    rl_prediction_y,
                    command_x,
                    command_y,
                    intent_x,
                    intent_y,
                    sensor_x,
                    sensor_y,
                    _,
                    state,
                ) = datapoint
                datapoints = DataPoints.add(
                    datapoint=datapoints,
                    point_id=point_id,
                    sensor_time_sec=sensor_time_sec,
                    sensor_time_nanosec=sensor_time_nanosec,
                    command_time_sec=command_time_sec,
                    command_time_nanosec=command_time_nanosec,
                    state=state,
                    gp_prediction_x=rl_prediction_x,
                    gp_prediction_y=rl_prediction_y,
                    command_x=command_x,
                    command_y=command_y,
                    intent_x=intent_x,
                    intent_y=intent_y,
                    sensor_x=sensor_x,
                    sensor_y=sensor_y,
                )
                n_datapoint += 1

        except BaseException:
            pass
        num_datapoints_queue = n_datapoint
        empty_queue_time = time.time() - start_t
        # logger.debug(f"Time to get {n_datapoint} datapoints: {empty_queue_time}")
        logger.debug(f"Got {n_datapoint} datapoints")

        # Add all datapoints to the dataset
        start_t = time.time()
        residual_td3.update(datapoints)
        dataset_addition_time = time.time() - start_t
        # logger.debug(f"Time to update dataset: {dataset_addition_time}")

        # If using auto-reset, apply it
        start_t = time.time()
        if USE_RESET:

            # Run the RL ME insertion check
            auto_reset, error_increase, error = residual_td3.auto_reset()

            # Send the force reset to influx
            try:
                writeEntry(
                    influx_client,
                    "/l1_auto_reset",
                    [
                        ("auto_reset", int(auto_reset)),
                        ("current_error", error),
                        ("current_error_increase", error_increase),
                    ],
                )
            except BaseException as e:
                logger.warning("Warning: Influx sending failed.")
                logger.warning(e)

            # Force the reset of the other processes
            if auto_reset:
                RLTraining_reset(residual_td3, auto_reset=True)

        self_reset_time = time.time() - start_t
        # logger.debug(f"Time to self reset: {self_reset_time}")

        # Train the RL
        start_t = time.time()
        # updated = residual_td3.rl_update()
        train_time = time.time() - start_t
        # logger.debug(f"Time to train gp: {train_time}")

        # Get reset signal from RL collection process again, to avoid
        # sending a new model to Adaptation if reseting
        try:
            _ = sensor_collection_to_rl_training_reset_queue.get_nowait()

            start_t = time.time()
            logger.debug("RESETING RL Training " * 20)
            residual_td3.reset()
            RLTraining_reset(residual_td3, auto_reset=False)
            updated = False
            reset_time += time.time() - start_t
            # logger.debug(f"Time to reset: {time.time() - start_t}")

        except BaseException:
            pass

        # If the model has been updated, send to Adaptation and Influx
        if updated:

            # Empty model queue
            start_t = time.time()
            try:
                rl_training_to_adaptation_model_queue.get_nowait()
            except BaseException:
                pass

            # Sending model to RL
            rl_training_to_adaptation_model_queue.put(
                (
                    residual_td3.actor_state.params,
                )
            )
            send_map_time = time.time() - start_t

            # Only send to Influx every RL_MAP_FREQUENCY
            if rl_map_sending_counter > RL_MAP_FREQUENCY:

                # Sending dataset to Influx
                start_t = time.time()
                try:
                    writeEntry(
                        influx_client,
                        "/l1_datasets",
                        [("dataset", residual_td3.dataset.to_string())],
                    )
                except BaseException as e:
                    logger.warning("Warning: Influx sending failed.")
                    logger.warning(e)
                # logger.debug(f"Time to send Dataset to Influx: {time.time() - start_t}")

                # Reseting counter
                rl_map_sending_counter = 0

        # Update loop metrics
        loop_time = time.time() - start_loop_t
        rl_map_sending_counter += 1

        # Send metrics to Influx
        if DEBUG_RL_TRAINING:
            start_t = time.time()
            fields = [
                ("rl_num_datapoints", residual_td3.N),
                ("rl_num_minibatch_datapoints", residual_td3.minibatch_N),
                ("num_datapoints_queue", num_datapoints_queue),
                ("total_loop_time", loop_time),
                ("reset_time", reset_time),
                ("empty_queue_time", empty_queue_time),
                ("dataset_addition_time", dataset_addition_time),
                ("self_reset_time", self_reset_time),
                ("rl_train_time", train_time),
                ("send_map_time", send_map_time),
            ]
            try:
                writeEntry(influx_client, "/l1_debug_timing", fields)
            except BaseException as e:
                logger.warning("Warning: Influx sending failed.")
                logger.warning(e)
            # logger.debug(f"Time to send metrics to Influx: {time.time() - start_t}")


######################
# Main Adaptation class #


class Adaptation(Node):
    def __init__(
        self,
        sensor_collection_to_adaptation_datapoint_queue: multiprocessing.Queue,
        adaptation_to_rl_collection_command_queue: multiprocessing.Queue,
        loglevel: str = "INFO",
    ) -> None:
        super().__init__("Adaptation")

        # Set up logging
        self._logger = define_logger("Adaptation", loglevel, "\x1b[33;20m Adaptation ")
        self._logger.info("Adaptation: Starting")

        # Init Influx Client
        self.influx_client = init_influx()
        self.NANO_CONVERSION = 1.0e9

        # Import inside process to reduce memory usage
        import os

        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".1"
        # os.environ['JAX_PLATFORMS']='cpu'
        from threading import Thread

        from functionality_controller.functionality_controller_l1 import (
            FunctionalityControllerL1,
        )
        from src.flair.flair.perturbation import PerturbationTransform

        # Queues to communicate with other processes
        self.adaptation_to_rl_collection_command_queue = (
            adaptation_to_rl_collection_command_queue
        )
        self.sensor_collection_to_adaptation_datapoint_queue = (
            sensor_collection_to_adaptation_datapoint_queue
        )

        self.start_adaptation = False

        # Publish our requested movement
        self._pub_robot = self.create_publisher(
            FLAIRTwist, "/cmd_vel", qos_profile_sensor_data
        )
        self._pub_flipper = self.create_publisher(
            FLAIRTwist, "/gvrbot_flipper_effort", qos_profile_sensor_data
        )
        self._pub_bridge_sent = False

        # Commands from the human
        self._sub_human = self.create_subscription(
            HumanControl,
            "/cmd_vel_human",
            self._cmd_vel_human_callback,
            qos_profile_sensor_data,
        )
        self._sub_cmd_vel_human_seen = False

        # System control
        self._sub_system_control = self.create_subscription(
            SystemControl,
            "/system_control",
            self._system_control_callback,
            qos_profile_sensor_data,
        )
        self._sub_system_control_seen = False

        # Perturbation
        self._sub_set_perturbation = self.create_subscription(
            Perturbation,
            "/set_perturbation",
            self._set_perturbation_callback,
            qos_profile_sensor_data,
        )
        self._sub_set_perturbation_seen = False

        # Boolean used to off the Adaptation
        self.adaptation_off = True  # False

        # Init Adaptation
        self._adaptation = FunctionalityControllerL1(
            min_command=MIN_COMMAND, 
            max_command=MAX_COMMAND,
        )
        self.state = None
        self.sensor_x = None
        self.sensor_y = None

        # Init Perturbation transform
        perturbation_logger = define_logger(
            "Perturbation", loglevel, "\x1b[36;20m PERTURBATION"
        )
        self.perturbation_transform = PerturbationTransform(
            wheel_base=WHEEL_BASE,
            wheel_radius=WHEEL_RADIUS,
            wheel_max_velocity=WHEEL_MAX_VELOCITY,
            logger=perturbation_logger,
        )

        # Pre-jit main functions of Adaptation
        _, _, _  = self._adaptation.get_command(
            0.0,
            0.0,
            0.0,
            np.zeros((8,)),
            0.0,
            0.0,
        )

        self._logger.info("Adaptation: Completed __init__")

    def _system_control_callback(self, msg: SystemControl) -> None:
        """Apply SystemControl changes."""
        if not self._sub_system_control_seen:
            self._logger.info("Present: SystemControl")
            self._sub_system_control_seen = True

        # Adaptation off/on
        if msg.adaptation_on:
            if self.adaptation_off:
                self._logger.info(f"got a control, adaptation is ENABLED.")
            self.adaptation_off = False
        else:
            if not self.adaptation_off:
                self._logger.info(f"got a control, adaptation is DISABLED.")
            self.adaptation_off = True

        # Adaptation reset
        if msg.adaptation_reset:
            self._logger.debug("RESETING Adaptation " * 20)
            self.state = None
            self._adaptation.reset()

    def _set_perturbation_callback(self, msg: Perturbation) -> None:
        """Apply Perturbation changes."""
        if not self._sub_set_perturbation_seen:
            self._logger.info("Present: Perturbation")
            self._sub_set_perturbation_seen = True

        # Get the latest state from Queue
        try:
            (self.state, self.sensor_x, self.sensor_y) = (
                self.sensor_collection_to_adaptation_datapoint_queue.get_nowait()
            )
        except BaseException:
            pass

        # Set the new Perturbation
        self.perturbation_transform.new_perturbation(msg, self.state)

    def _cmd_vel_human_callback(self, msg: HumanControl) -> None:
        """Adapt HumanControl."""

        if not self._sub_cmd_vel_human_seen:
            self._logger.info("Present: HumanControl")
            self._sub_cmd_vel_human_seen = True

        # Apply adaptation
        adaptation_msg = self.adaptation(deepcopy(msg))

        # Apply perturbation
        final_msg = self.perturbation_transform.apply_perturbation(
            deepcopy(adaptation_msg), self.state
        )

        # Send new command msg
        self._send_movement(final_msg)

        # Send to Influx
        try:
            end_timestamp = self.get_clock().now().to_msg()
            end_timestamp = int(end_timestamp.sec) * int(1e9) + int(
                end_timestamp.nanosec
            )
            start_timestamp = msg.joystick.original_stamp
            start_timestamp = int(start_timestamp.sec) * int(1e9) + int(
                start_timestamp.nanosec
            )
            writeEntry(
                self.influx_client,
                "/l1_adaptation",
                [
                    ("human_cmd_lin_x", msg.joystick.linear.x),
                    ("human_cmd_ang_z", msg.joystick.angular.z),
                    ("human_cmd_flipper", msg.flipper.angular.x),
                    ("adaptation_cmd_lin_x", adaptation_msg.joystick.linear.x),
                    ("adaptation_cmd_ang_z", adaptation_msg.joystick.angular.z),
                    ("adaptation_cmd_flipper", adaptation_msg.flipper.angular.x),
                    ("perturbation_cmd_lin_x", final_msg.joystick.linear.x),
                    ("perturbation_cmd_ang_z", final_msg.joystick.angular.z),
                    ("perturbation_cmd_flipper", final_msg.flipper.angular.x),
                    ("original_id", msg.joystick.original_id),
                    ("original_start", start_timestamp),
                    ("adaptation_end", end_timestamp),
                ],
                timestamp=end_timestamp,
            )
        except BaseException as e:
            self._logger.warning("Warning: Influx sending failed.")
            self._logger.warning(e)

    def _send_movement(self, movement: FunctionalityControllerControl) -> None:
        """Publish the Adaptation command to the robot."""
        if not self._pub_bridge_sent:
            self._logger.info("Sent: FunctionalityControllerControl")
            self._pub_bridge_sent = True

        msg = movement.joystick
        self._pub_robot.publish(msg)
        msg = movement.flipper
        self._pub_flipper.publish(msg)

    def adaptation(self, msg: HumanControl) -> FunctionalityControllerControl:
        """Main function applied when getting a command."""

        # Create new command msg
        new_msg = FunctionalityControllerControl()
        new_msg.joystick.original_id = msg.joystick.original_id
        new_msg.joystick.original_stamp = msg.joystick.original_stamp
        new_msg.flipper.original_id = msg.flipper.original_id
        new_msg.flipper.original_stamp = msg.flipper.original_stamp

        # Get the latest state from Queue
        try:
            (self.state, self.sensor_x, self.sensor_y) = (
                self.sensor_collection_to_adaptation_datapoint_queue.get_nowait()
            )
        except BaseException:
            pass

        # Extract commands
        cmd_lin_x = msg.joystick.linear.x
        cmd_ang_z = msg.joystick.angular.z
        cmd_flipper = msg.flipper.angular.x

        # Get command from adaptation
        (
            lqr_prediction_x,
            lqr_prediction_y,
            lqr_prediction_flipper,
        ) = self._adaptation.get_command(
            cmd_lin_x,
            cmd_ang_z,
            cmd_flipper,
            self.state, 
            self.sensor_x,
            self.sensor_y,
        )
        #self._logger.debug(
        #        f"Running with Commands {cmd_lin_x}{cmd_ang_z} velocities{self.sensor_x} and {self.sensor_y} and Predicted commands {lqr_prediction_x} {lqr_prediction_y}"
        #    )

        # If the Adaptation is off, just pass commands through
        if self.adaptation_off:
            command_x = msg.joystick.linear.x
            command_y = msg.joystick.angular.z
            command_flipper = msg.flipper.angular.x

            new_msg.joystick.linear.x = command_x
            new_msg.joystick.angular.z = command_y
            new_msg.flipper.angular.x = command_flipper

        # If the Adaptation is on, apply the Adaptation correction
        else:
            command_x = lqr_prediction_x
            command_y = lqr_prediction_y
            command_flipper = lqr_prediction_flipper

            new_msg.joystick.linear.x = command_x
            new_msg.joystick.angular.z = command_y
            new_msg.flipper.angular.x = command_flipper

        #self._logger.debug(f"Errors Prediction {lqr_prediction_x - cmd_lin_x}, {lqr_prediction_y - cmd_ang_z}")

        # Empty DataCollection queue
        try:
            self.adaptation_to_rl_collection_command_queue.get_nowait()
        except BaseException:
            pass

        # Send to DataCollection
        self.adaptation_to_rl_collection_command_queue.put(
            (
                deepcopy(command_x),
                deepcopy(command_y),
                deepcopy(lqr_prediction_x),
                deepcopy(lqr_prediction_y),
                deepcopy(msg.joystick.linear.x),
                deepcopy(msg.joystick.angular.z),
            )
        )

        # Return final msg
        return new_msg



# Adaptation process
def start_adaptation_process(
    args,
    loglevel: str,
    sensor_collection_to_adaptation_datapoint_queue: multiprocessing.Queue,
    adaptation_to_rl_collection_command_queue: multiprocessing.Queue,
) -> None:
    """Thread for the Adaptation."""

    # Start Adaptation process
    rclpy.init(args=args)
    antoine = Adaptation(
        sensor_collection_to_adaptation_datapoint_queue=sensor_collection_to_adaptation_datapoint_queue,
        adaptation_to_rl_collection_command_queue=adaptation_to_rl_collection_command_queue,
        loglevel=loglevel,
    )
    rclpy.spin(antoine)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    antoine.destroy_node()
    rclpy.shutdown()


########
# Main #


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
    except BaseException:
        rosname = socket.gethostbyname(oldros)
        os.environ["ROS_DISCOVERY_SERVER"] = f"{rosname}:{port}"

    print(
        f"Adaptation: going to connect to {os.getenv('ROS_DISCOVERY_SERVER')}",
        flush=True,
    )

    newargs = rclpy.utilities.remove_ros_args(sys.argv)
    args = parser.parse_args(args=newargs[1:])

    # Create all the Queues
    sensor_collection_to_rl_collection_datapoint_queue = context.Queue()
    sensor_collection_to_adaptation_datapoint_queue = context.Queue()

    sensor_collection_to_rl_training_reset_queue = context.Queue()
    sensor_collection_to_rl_collection_reset_queue = context.Queue()

    adaptation_to_rl_collection_command_queue = context.Queue()

    rl_collection_to_rl_training_datapoint_queue = context.Queue()

    rl_collection_to_sensor_collection_start_queue = context.Queue()
    rl_training_to_rl_collection_start_queue = context.Queue()

    # Start Sensor Collection process
    sensor_collection_process = context.Process(
        target=start_sensor_collection_process,
        args=(
            sys.argv,
            args.log,
            sensor_collection_to_rl_collection_datapoint_queue,
            sensor_collection_to_adaptation_datapoint_queue,
            sensor_collection_to_rl_training_reset_queue,
            sensor_collection_to_rl_collection_reset_queue,
            rl_collection_to_sensor_collection_start_queue,
        ),
    )
    sensor_collection_process.daemon = True
    sensor_collection_process.start()

    # Start RL Collection process
    #rl_collection_process = context.Process(
    #    target=start_rl_collection_process,
    #    args=(
    #        sys.argv,
    #        args.log,
    #        adaptation_to_rl_collection_command_queue,
    #        sensor_collection_to_rl_collection_datapoint_queue,
    #        rl_collection_to_rl_training_datapoint_queue,
    #        rl_training_to_rl_collection_start_queue,
    #        rl_collection_to_sensor_collection_start_queue,
    #        sensor_collection_to_rl_collection_reset_queue,
    #    ),
    #)
    #rl_collection_process.daemon = True
    #rl_collection_process.start()

    # Start RL Training process
    # rl_training_process = context.Process(
    #     target=start_rl_training_process,
    #     args=(
    #         args.log,
    #         rl_training_to_adaptation_model_queue,
    #         rl_collection_to_rl_training_datapoint_queue,
    #         rl_training_to_rl_collection_start_queue,
    #         rl_training_to_adaptation_start_queue,
    #         sensor_collection_to_rl_training_reset_queue,
    #     ),
    # )
    # rl_training_process.daemon = True
    # rl_training_process.start()

    # Start Adaptation
    start_adaptation_process(
        sys.argv,
        args.log,
        sensor_collection_to_adaptation_datapoint_queue,
        adaptation_to_rl_collection_command_queue,
    )


if __name__ == "__main__":
    main()
