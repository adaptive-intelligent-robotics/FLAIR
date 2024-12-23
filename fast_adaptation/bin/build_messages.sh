#!/bin/sh
sudo find . -name '*.pyc' -exec rm {} \;
sudo rm -fr build install log
u=`id -u`;docker run -d --rm -v `pwd`:/me -e HOME=/tmp --name build-ros ros:pl bash -c "echo '$USER:x:$u:100::/tmp:/bin/bash' >>/etc/passwd;sleep infinity"
docker exec -it -u `id -u` build-ros bash -c "source /opt/ros/humble/setup.bash; colcon build --symlink-install; exit"
docker kill build-ros
