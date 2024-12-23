import logging
import random
from typing import Dict

import rclpy
import roslibpy
from FLAIR_msg.msg import FLAIRTwist, Latency
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args

random.seed(0)


class MyStreamHandler(logging.StreamHandler):
    def flush(self) -> None:
        self.stream.flush()


class Bridge(object):
    def __init__(self, ros1master: str = "10.1.1.10", loglevel: str = "INFO") -> None:

        # Set up logging
        llevel = getattr(logging, loglevel.upper())
        if not isinstance(llevel, int):
            raise ValueError(f"Invalid log level: {loglevel}")

        sh = MyStreamHandler()
        sh.setLevel(llevel)
        self._logger = logging.getLogger("Bridge")
        self._logger.setLevel(llevel)
        self._logger.addHandler(sh)
        self._logger.info("Bridge: Starting")

        # ROS 2 setup
        rclpy.init()
        self.ros2_node = rclpy.create_node("Bridge")

        # Latency publisher
        self.ros2_p_latency = self.ros2_node.create_publisher(
            Latency, "/latency", qos_profile_sensor_data
        )
        self.ros2_p_latency_sent = False

        # Cmd_vel publisher propagating back the cmd applied to the robot
        self.ros2_p_cmdvel = self.ros2_node.create_publisher(
            FLAIRTwist, "/real/cmd_vel", qos_profile_sensor_data
        )
        self.ros2_p_cmdvel_sent = False
        self.ros2_p_flipper = self.ros2_node.create_publisher(
            FLAIRTwist, "/real/gvrbot_flipper_effort", qos_profile_sensor_data
        )
        self.ros2_p_flipper_sent = False

        # Cmd_vel subscription, sending cmd to robot
        self.ros2_s_cmd_vel = self.ros2_node.create_subscription(
            FLAIRTwist, "/cmd_vel", self._ros2_cmdvel_callback, qos_profile_sensor_data
        )
        self.ros2_s_cmd_vel_seen = False

        # Flipper subscription, sending flipper cmd to robot
        self.ros2_s_gvr_flip = self.ros2_node.create_subscription(
            FLAIRTwist,
            "/gvrbot_flipper_effort",
            self._ros2_gvrflip_callback,
            qos_profile_sensor_data,
        )
        self.ros2_s_gvr_flip_seen = False

        # ROS 1 setup
        self._logger.debug(f"trying to connect to master {ros1master}")
        self.ros1_node = roslibpy.Ros(host=ros1master, port=9090)
        self.ros1_node.on_ready(self._is_ready)
        self.ros1_p_cmd_vel = None

        self._logger.info("Bridge: Completed __init__")

    def _is_ready(self):
        # ROS 1 push/pull
        self._logger.info("I'm ready")
        self.ros1_p_cmd_vel = roslibpy.Topic(
            self.ros1_node, "/cmd_vel", "geometry_msg/Twist"
        )
        self.ros1_p_cmd_vel.advertise()
        self.ros1_p_gvr_flip = roslibpy.Topic(
            self.ros1_node, "/gvrbot_flipper_effort", "geometry_msg/Twist"
        )
        self.ros1_p_gvr_flip.advertise()

        self.ros1_s_cmd_vel = roslibpy.Topic(
            self.ros1_node, "/real/cmd_vel", "geometry_msg/Twist"
        )
        self.ros1_s_cmd_vel.subscribe(self._ros1_cmd_vel_callback)
        self.ros1_s_gvr_flip = roslibpy.Topic(
            self.ros1_node, "/real/gvrbot_flipper_effort", "geometry_msg/Twist"
        )
        self.ros1_s_gvr_flip.subscribe(self._ros1_flipper_callback)

    def _ros2_cmdvel_callback(self, msg: FLAIRTwist):
        # Got a ROS 2 cmd_vel, send it over to ROS 1
        if not self.ros2_s_cmd_vel_seen:
            self._logger.info("Present: Cmd_Vel")
            self.ros2_s_cmd_vel_seen = True
        self._logger.debug("ROS2 - ROS1 - /cmd_vel")

        # Check msg original info have been passed safely
        if msg.original_id > 1000 or (
            msg.original_stamp.sec == 0 and msg.original_stamp.nanosec == 0
        ):
            self._logger.info(
                "ERROR: one of the component if not propagating original id and timestamp."
            )

        # Send msg to latency
        else:
            if not self.ros2_p_latency_sent:
                self._logger.info("Sent: Latency")
                self.ros2_p_latency_sent = True
            latency_msg = Latency()
            latency_msg.original_id = msg.original_id
            latency_msg.original_start = msg.original_stamp
            latency_msg.module_end = self.ros2_node.get_clock().now().to_msg()
            latency_msg.module_name = "Rosbridge"
            self.ros2_p_latency.publish(latency_msg)

        # Transfert msg to robot
        if self.ros1_p_cmd_vel:
            new_msg: Dict = dict()
            new_msg["linear"] = dict()
            new_msg["linear"]["x"] = msg.linear.x
            new_msg["linear"]["y"] = msg.linear.y
            new_msg["linear"]["z"] = msg.linear.z
            new_msg["angular"] = dict()
            new_msg["angular"]["x"] = msg.angular.x
            new_msg["angular"]["y"] = msg.angular.y
            new_msg["angular"]["z"] = msg.angular.z
            self.ros1_p_cmd_vel.publish(roslibpy.Message(new_msg))
            self._logger.debug(f"sending ROS1: {new_msg}")
            self.ros1_node.run()

    def _ros2_gvrflip_callback(self, msg: FLAIRTwist):
        # Got a ROS 2 cmd_vel, send it over to ROS 1
        if not self.ros2_s_gvr_flip_seen:
            self._logger.info("Present: Gvrbot Flip")
            self.ros2_s_gvr_flip_seen = True
        self._logger.debug("ROS2 - ROS1 - /gvr_flipper_effort")

        # Check msg original info have been passed safely
        if msg.original_id > 1000 or (
            msg.original_stamp.sec == 0 and msg.original_stamp.nanosec == 0
        ):
            self._logger.info(
                "ERROR: one of the component if not propagating original id and timestamp."
            )

        # Transfert msg to robot
        if self.ros1_p_gvr_flip:
            new_msg: Dict = dict()
            new_msg["linear"] = dict()
            new_msg["linear"]["x"] = msg.linear.x
            new_msg["linear"]["y"] = msg.linear.y
            new_msg["linear"]["z"] = msg.linear.z
            new_msg["angular"] = dict()
            new_msg["angular"]["x"] = msg.angular.x
            new_msg["angular"]["y"] = msg.angular.y
            new_msg["angular"]["z"] = msg.angular.z
            self.ros1_p_gvr_flip.publish(roslibpy.Message(new_msg))
            self._logger.debug(f"sending ROS1: {new_msg}")
            self.ros1_node.run()

    def _ros1_cmd_vel_callback(self, msg: dict):
        # Got a ROS 1 /real/cmd_vel, send it over to ROS 2
        if not self.ros2_p_cmdvel_sent:
            self._logger.info("Sent: Real_Cmd_Vel")
            self.ros2_p_cmdvel_sent = True
        self._logger.debug("ROS1 - ROS2 - /real/cmd_vel")
        new_msg = FLAIRTwist()
        new_msg.linear.x = msg["linear"]["x"]
        new_msg.linear.y = msg["linear"]["y"]
        new_msg.linear.z = msg["linear"]["z"]
        new_msg.angular.x = msg["angular"]["x"]
        new_msg.angular.y = msg["angular"]["y"]
        new_msg.angular.z = msg["angular"]["z"]
        self.ros2_p_cmdvel.publish(roslibpy.Message(new_msg))

    def _ros1_flipper_callback(self, msg: dict):
        # Got a ROS 1 /real/gvrbot_flipper_effort, send it over to ROS 2
        if not self.ros2_p_flipper_sent:
            self._logger.info("Sent: Real_Flipper")
            self.ros2_p_flipper_sent = True
        self._logger.debug("ROS1 - ROS2 - /real/gvrbot_flipper_effort")
        new_msg = FLAIRTwist()
        new_msg.linear.x = msg["linear"]["x"]
        new_msg.linear.y = msg["linear"]["y"]
        new_msg.linear.z = msg["linear"]["z"]
        new_msg.angular.x = msg["angular"]["x"]
        new_msg.angular.y = msg["angular"]["y"]
        new_msg.angular.z = msg["angular"]["z"]
        self.ros2_p_flipper.publish(roslibpy.Message(new_msg))


def main():
    import argparse
    import os
    import socket
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l", "--log", type=str, default="INFO", help="Log level to use"
    )
    parser.add_argument(
        "-r", "--ros1master", type=str, help="IP address of ROS 1 Master"
    )

    # The code can't handle a name, only IP addresses, so do the mapping
    rds = os.getenv("ROS_DISCOVERY_SERVER")
    oldros, port = rds.split(":")
    try:
        int(oldros.split(".")[0])
    except Exception:
        rosname = socket.gethostbyname(oldros)
        os.environ["ROS_DISCOVERY_SERVER"] = f"{rosname}:{port}"

    print(
        f"Bridge: going to connect to {os.getenv('ROS_DISCOVERY_SERVER')}", flush=True
    )

    newargs = remove_ros_args(sys.argv)
    args = parser.parse_args(args=newargs[1:])

    bridge = Bridge(args.ros1master, args.log)

    bridge.ros1_node.run()
    rclpy.spin(bridge.ros2_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    bridge.ros2_node.destroy_node()
    bridge.ros2_node.shutdown()
    bridge.ros1_node.terminate()


if __name__ == "__main__":
    main()
