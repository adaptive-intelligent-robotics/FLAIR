import os

import numpy as np

USE_RESET = False

##################
# TD3 parameters #

MIN_COMMAND = -2.0
MAX_COMMAND = 2.0
LEARNING_RATE = 3e-4
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
GAMMA = 0.99
TAU = 0.005
BATCH_SIZE = 64
POLICY_FREQUENCY = 2


####################
# Robot parameters #

ROBOT_WIDTH = 0.33  # meters
WHEEL_BASE = 0.35  # meters
WHEEL_RADIUS = 0.0862  # wheel radius
WHEEL_MAX_VELOCITY = 27  # radians/second


######################
# Dataset parameters #

MIN_DIFF_DATAPOINT = BATCH_SIZE

DATAPOINT_BATCH_SIZE = 20  # 50 before fixed-grid
USE_GRID_DATASET = True
DATASET_SIZE = 1000000


####################
# Reset parameters #

MINIBATCH_SIZE = 30
ERROR_BUFFER_SIZE = 30
WEIGHT_ANGULAR_ROT = 0.8
NEW_SCENARIO_THRESHOLD = 0.34  # 0.26 #0.33


##############################
# Data Collection parameters #

FILTER_TRANSITION = (
    True  # Do not return datapoint if the command is not constant over the interval
)
FILTER_VARYING_ANGLE = (
    False  # Do not return datapoint if the angles are not constant over the interval
)
FILTER_TURNING_ONLY = False  # Do not return datapoint if turning only in command

# Signal matching parameters
BUFFER_SIZE = 20  # In nb command points
MIN_DELAY = 100000000  # In nanoseconds
MAX_DELAY = 500000000  # In nanoseconds

# Point selection parameters
SELECTION_SIZE = 5  # In nb command points
IQC_Q1 = 0.05
IQC_Qn = 0.95

# Command non-constant filtering
FILTER_TRANSITION_SIZE = 3  # In nb command points
FILTER_TRANSITION_TOLERANCE = 0.2  # In m/s

# Angle non-constant filtering
FILTER_VARYING_ANGLE_SIZE = 30  # In nb sensor points
FILTER_VARYING_ANGLE_TOLERANCE = 0.3  # In rad/s

# Turning only filtering
FILTER_TURNING_ONLY_TOLERANCE = 0.05


###################
# Debug parameter #

DEBUG_RL_COLLECTION = True
DEBUG_RL_TRAINING = True
RL_MAP_FREQUENCY = 10  # Frequency to send to Influx the dataset for debugging
