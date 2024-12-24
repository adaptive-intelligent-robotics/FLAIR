# FLAIR
This is the official repository for the FLAIR algorithm and the paper "Getting Robots Back On Track: Reconstituting Control in Unexpected Situations with Online Learning".

Our system is based on ROS2 and docker-compose. 

## Building Docker Images

Run the following commands inside the `docker` container:

```
docker build -t airl:flair -f airl/Dockerfile .
docker build -t airl:ros2_light -f airl/Dockerfile.ros2_light .
docker build -t airl:zeds -f airl/Dockerfile.stereolabs .
```

## Starting Script

Before running any part of the scripts, it is necessary to compile the ROS scripts via the [build_messages.sh](fast_adaptation/bin/build_messages.sh) script.
```
cd fast_adaptation && ./build_messages.sh
```

Once this step is done, it is possible to start the full pipeline.
The script to start our docker pipeline can be run via the [start_flair.sh](fast_adaptation/bin/start_flair.sh) script.
```
./start_flair.sh
```

We run the docker-compose pipeline on our NVIDIA Jetson Orin 32Gb that is directly connected to a VectorNav V100 IMU and a Zed2 Camera in addition to the GVRBot. The docker-compose file that is executed can be found [here](fast_adaptation/docker_compose_flair.yaml).


**Note** Our setup requires the user to execute the following command to be executed to see the robot:
```
sudo route delete default
```
Once the command is executed, it is possible to ssh into the robot to start the rosbridge server (the GVRBot runs ROS whereas our codebase is written in ROS2).


### Virtual Driver
Finally, to launch the virtual driver, it is necessary to restart the container called `vicon` which will then listen to the VICON system sending out messages to the rostopic named after the tracked object. In our case this is the `"/vicon/GVRBot_IC/GVRBot_IC"` topic. (See [File](fast_adaptation/src/flair/flair/vicon.py)). In case you are running your the code on your own system, you need to adjust the topic names.
Run the command:

```
./restart_container.sh vicon
```
### Structure

The main script implementing FLAIR is called [adaptation.py](fast_adaptation/src/flair/flair/adaptation.py). This script starts the multiple processes taking care of the data collection, processing and training. The code for each of these functionalities can be found in the [functionality_controller](fast_adaptation/src/flair/functionality_controller) folder which is imported into the `adaptation.py` script. The controller for the code can be found in the [functionality_controller.py](fast_adaptation/src/flair/functionality_controller/functionality_controller.py) file, which is the code directly producing the commands.

## Configurations

All the parameters that can be tweaked can be found in a configuration file called [adaptation_config.py](fast_adaptation/src/flair/flair/adaptation_config.py). All the parameters listed here have been used for the experimental runs reported in our paper.

The script [vicon.py](fast_adaptation/src/flair/flair/vicon.py) contains the exact waypoints used for our experiments (which correspond to our lab settings), but they can easily be replaced by your path. At the beginning of the file, you can chose the perturbation you would like to apply to the robot. This file can be replaced by a manual controller for example if required.

### Baselines

In addition to the code for FLAIR, we publish our baselines under the files [adaptation_rl.py](fast_adaptation/src/flair/flair/adaptation_rl.py) and [adaptation_lqr.py](fast_adaptation/src/flair/flair/adaptation_lqr.py). Similarly, the files producing the commands at each step are called [functionality_controller_rl.py](fast_adaptation/src/flair/functionality_controller/functionality_controller_rl.py) and [functionality_controller_lqr.py](fast_adaptation/src/flair/functionality_controller/functionality_controller_lqr.py).

The script to start our docker pipeline for the baselines can be run via the [rl_start.sh](fast_adaptation/bin/rl_start.sh) script.
```
./rl_start.sh
```

or 
```
./lqr_start.sh
```
All the parameters that can be tweaked can be found in a configuration file called [adaptation_config_rl.py](fast_adaptation/src/flair/flair/adaptation_config_rl.py) or [adaptation_config_lqr.py](fast_adaptation/src/flair/flair/adaptation_config_lqr.py). All the parameters listed here have been used for the experimental runs reported in our paper.

