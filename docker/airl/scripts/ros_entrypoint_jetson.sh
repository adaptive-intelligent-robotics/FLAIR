#!/bin/bash
set -e

# setup ros2 environment
source "/opt/ros/$ROS_DISTRO/install/setup.bash" --
source "/root/ros2_ws/install/local_setup.bash" --

export ROS_DISCOVERY_SERVER=10.1.1.220:11811
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/me/config/super_client_profile.xml

# Welcome information
echo "ZED ROS2 Docker Image"
echo "---------------------"
echo 'ROS distro: ' $ROS_DISTRO
echo 'DDS middleware: ' $RMW_IMPLEMENTATION 
echo "---"  
echo 'Available ZED packages:'
ros2 pkg list | grep zed
echo "---------------------"    
exec "$@"
