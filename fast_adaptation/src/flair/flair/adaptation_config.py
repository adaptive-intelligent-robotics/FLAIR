import numpy as np
import os

# Default configuration to set parameters automatically
ADAPTATION_TREADMILL = "off"
ADAPTATION_WIND = "off"

USE_RESET = True

#############################
# Adaptation version choice #

NO_STATE_FAST = 0
STATE_FAST = 1
STATE_FAST_GP = 2

ADAPTATION_VERSION = STATE_FAST
if ADAPTATION_TREADMILL == "on":
    ADAPTATION_VERSION = STATE_FAST
elif ADAPTATION_WIND == "on":
    ADAPTATION_VERSION = STATE_FAST


#############################
# Edge avoidance parameters #

ADAPTATION_EDGE_AVOIDANCE = "off"
TREADMILL_HALF_WIDTH = 0.39
SAFETY_THRESHOLD = 0.25


######################
# Robot parameters #

ROBOT_WIDTH = 0.33 # meters
WHEEL_BASE = 0.35 # meters
WHEEL_RADIUS = 0.0862 # wheel radius
WHEEL_MAX_VELOCITY = 27 # radians/second

DEFAULT_OBS_NOISE = np.asarray([0.01750011])
DEFAULT_VARIANCE = np.array([0.04772636])
if ADAPTATION_VERSION == STATE_FAST or ADAPTATION_VERSION == STATE_FAST_GP:
    DEFAULT_LENGTHSCALE = np.array([0.69437875, 0.56324503, 10.0])
else:
    DEFAULT_LENGTHSCALE = np.array([0.69437875, 0.56324503])


###################
# Grid parameters #

GRID_RESOLUTION = 101 
MIN_COMMAND = -2.0
MAX_COMMAND = 2.0


######################
# Dataset parameters #

MIN_DIFF_DATAPOINT = 2

DATAPOINT_BATCH_SIZE = 20 # 50 before fixed-grid
USE_GRID_DATASET = True
DATASET_SIZE = 1000
DATASET_GRID_CELL_SIZE = 15 # 20 before fixed-grid
DATASET_GRID_NEIGH = 0.6 # 0.5 before fixed-grid
DATASET_GRID_NOVELTY_THRESHOLD = 0.01

####################
# Prior parameters #

MAX_P_VALUE = 0.5
P_SOFT_UPDATE_SIZE = 0.3
MIN_SPREAD = 0.2


###########################################
# State prior parameters (STATE_FAST only) #

if ADAPTATION_TREADMILL == "on":

    MULTI_FUNCTON = True
    REMOVE_OFFSET = False

    # x-position state
    STATE_DIM = 4  # 4 for x-position, 5 for yaw
    STATE_MIN_DATASET = -0.7  # -0.7 for x-position, -0.4 for yaw
    STATE_MAX_DATASET = 0.7  # 0.7 for x-position, 0.4 for yaw
    STATE_MIN_OPT_CLIP = -0.6  # -0.6 for x-position, -0.4 for yaw 
    STATE_MAX_OPT_CLIP = 0.6  # 0.6 for x-position, 0.4 for yaw

    # Clipping for the p-values
    MAX_P_VALUE = 0.8
    P1_MIN = - MAX_P_VALUE/STATE_MAX_OPT_CLIP
    P1_MAX = MAX_P_VALUE/STATE_MAX_OPT_CLIP
    P2_MIN = - MAX_P_VALUE/(STATE_MAX_OPT_CLIP**2)
    P2_MAX = MAX_P_VALUE/(STATE_MAX_OPT_CLIP**2)
    P3_MIN = - MAX_P_VALUE/(STATE_MAX_OPT_CLIP**3)
    P3_MAX = MAX_P_VALUE/(STATE_MAX_OPT_CLIP**3)

elif ADAPTATION_WIND == "on":

    MULTI_FUNCTON = True
    REMOVE_OFFSET = True

    # yaw state
    STATE_DIM = 5  # 4 for x-position, 5 for yaw
    STATE_MIN_DATASET = -3.14  # -0.7 for x-position, -0.4 for yaw
    STATE_MAX_DATASET = 3.14  # 0.7 for x-position, 0.4 for yaw
    STATE_MIN_OPT_CLIP = -3.14  # -0.6 for x-position, -0.4 for yaw 
    STATE_MAX_OPT_CLIP = 3.14  # 0.6 for x-position, 0.4 for yaw

    # Clipping for the p-values
    MAX_P_VALUE = 0.3
    P1_MIN = - MAX_P_VALUE/STATE_MAX_OPT_CLIP
    P1_MAX = MAX_P_VALUE/STATE_MAX_OPT_CLIP
    P2_MIN = - MAX_P_VALUE/(STATE_MAX_OPT_CLIP**2)
    P2_MAX = MAX_P_VALUE/(STATE_MAX_OPT_CLIP**2)
    P3_MIN = - MAX_P_VALUE/(STATE_MAX_OPT_CLIP**3)
    P3_MAX = MAX_P_VALUE/(STATE_MAX_OPT_CLIP**3)

else:

    MULTI_FUNCTON = True
    REMOVE_OFFSET = True 

    # yaw state
    STATE_DIM = 5  # 4 for x-position, 5 for yaw
    STATE_MIN_DATASET = -3.14  # -0.7 for x-position, -0.4 for yaw
    STATE_MAX_DATASET = 3.14  # 0.7 for x-position, 0.4 for yaw
    STATE_MIN_OPT_CLIP = -3.14  # -0.6 for x-position, -0.4 for yaw 
    STATE_MAX_OPT_CLIP = 3.14  # 0.6 for x-position, 0.4 for yaw

    # Clipping for the p-values
    MAX_P_VALUE = 0.5
    P1_MIN = - MAX_P_VALUE/STATE_MAX_OPT_CLIP
    P1_MAX = MAX_P_VALUE/STATE_MAX_OPT_CLIP
    P2_MIN = - MAX_P_VALUE/(STATE_MAX_OPT_CLIP**2)
    P2_MAX = MAX_P_VALUE/(STATE_MAX_OPT_CLIP**2)
    P3_MIN = - MAX_P_VALUE/(STATE_MAX_OPT_CLIP**3)
    P3_MAX = MAX_P_VALUE/(STATE_MAX_OPT_CLIP**3)


####################
# Reset parameters #

MINIBATCH_SIZE = 30
ERROR_BUFFER_SIZE = 30
WEIGHT_ANGULAR_ROT = 0.8 
NEW_SCENARIO_THRESHOLD = 0.34 #0.26 #0.33


##############################
# Data Collection parameters #

FILTER_TRANSITION = True # Do not return datapoint if the command is not constant over the interval
FILTER_VARYING_ANGLE = False # Do not return datapoint if the angles are not constant over the interval
FILTER_TURNING_ONLY = False # Do not return datapoint if turning only in command

# Signal matching parameters
BUFFER_SIZE = 20 # In nb command points
MIN_DELAY = 100000000 # In nanoseconds
MAX_DELAY = 500000000 # In nanoseconds

# Point selection parameters
SELECTION_SIZE = 5 # In nb command points
IQC_Q1 = 0.05
IQC_Qn = 0.95

# Command non-constant filtering
FILTER_TRANSITION_SIZE = 3 # In nb command points
FILTER_TRANSITION_TOLERANCE = 0.2 # In m/s

# Angle non-constant filtering
FILTER_VARYING_ANGLE_SIZE = 30 # In nb sensor points
FILTER_VARYING_ANGLE_TOLERANCE = 0.3 # In rad/s

# Turning only filtering
FILTER_TURNING_ONLY_TOLERANCE = 0.05


###################
# Debug parameter #

DEBUG_GP_COLLECTION = True
DEBUG_GP_TRAINING = True
GP_MAP_FREQUENCY = 10 # Frequency to send to Influx the GPTraining map for debugging

