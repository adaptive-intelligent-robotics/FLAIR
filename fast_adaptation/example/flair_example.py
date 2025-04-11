import sys
import os
import time
import csv
import traceback
import argparse
from copy import deepcopy
from functools import partial

import numpy as np
from typing import Tuple, Dict, Any

import jax
import jax.numpy as jnp
from brax.v1.io import html
from brax.math import quat_to_euler
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)

# Import the building blocks from the main FLAIR pipeline
sys.path.append(os.path.abspath("../src/flair"))
from flair.adaptation_config import *
from functionality_controller.functionality_controller import FunctionalityController
from functionality_controller.datapoint import DataPoints
from functionality_controller.functionality_controller_utils import (
    compute_borders,
    compute_clipping,
)
from functionality_controller.data_collection import DataCollection
    

# Import the path config
from path_config import chicane_path, chicane_number_laps, wind_path, wind_number_laps

# Import the environment
from utils.set_up_hexapod import set_up_hexapod

# Uncomment this to debug if time is spent rejiting functions.
#import logging
#logging.basicConfig(level=logging.DEBUG)

# Define new version of robot-specific parameters
# ROBOT_WIDTH = 
# WHEEL_BASE = 
# WHEEL_RADIUS = 
# WHEEL_MAX_VELOCITY = 

# Create a dummy Logger that just prints
class PrintLogger:

    def debug(msg: str) -> None:
        print(f"        DEBUG: {msg}")

    def info(msg: str) -> None:
        print(f"        INFO: {msg}")

    def warning(msg: str) -> None:
        print(f"    WARNING: {msg}")

    def error(msg: str) -> None:
        print(f"    ERROR: {msg}")

    def critical(msg: str) -> None:
        print(f"    CRITICAL: {msg}")

# Create fake ros timestamps
class ROSTimestamp:
    def __init__(self, timestep: int, rate_hz: float):
        total_nanoseconds = timestep * (1e9 / rate_hz)
        self.sec = int(total_nanoseconds // 1e9)
        self.nanosec = int(total_nanoseconds % 1e9)

# Create fake UTC timestamps
from datetime import datetime, timedelta, timezone
def timestep_to_utc_timestamp(timestep: int, sim_start: datetime, rate_hz: float):
    elapsed = timedelta(seconds=timestep / rate_hz)
    dt = sim_start + elapsed
    return dt.isoformat(sep=' ', timespec='microseconds')

##########
# Driver #

class Driver:
    """
    Automatic driver for the robot. Similar to the code
    of Vicon in the real-world robotics pipeline. 
    Most of the code is taken from there but removing the ROS 
    components. 
    """

    def __init__(self, path: np.array, number_laps: int) -> None:

        self.path = path
        self.number_laps = number_laps
        self.target = 0
        self.target_tx, self.target_ty = self.path[self.target]
        self.lap = 0

        # Parameters
        self.error_threshold = 0.5
        self.min_driver_speed = 0.0
        self.max_driver_speed = 0.08
        self.max_driver_rotation = 0.1

    def reset(self) -> None:
        self.target = 0
        self.target_tx, self.target_ty = self.path[self.target]
        self.lap = 0

    def get_driver_configuration(self) -> Tuple[float, float]:
        return self.min_driver_speed, self.max_driver_speed

    def follow_path(
            self, x_pos: float, y_pos: float, quaternion: np.ndarray
        ) -> Tuple[bool, float, float]:
        """
        Create the path tracking command from the sensor readings.
        Code from the Vicon driver of the robotic experiments.
        """
        
        # Current target position
        x, y = self.path[self.target]
        self.target_tx, self.target_ty = x, y

        # Transform quaternion
        r = R.from_quat(quaternion)

        # Compute the error
        error_x = (x - x_pos) 
        error_y = (y - y_pos) 
        error_robot_frame = r.apply(np.asarray([error_x, error_y, 0]), inverse=True)

        print("error_robot_frame", error_robot_frame)
        angle_heading = np.arctan2(error_robot_frame[1], error_robot_frame[0])
        distance = np.linalg.norm(error_robot_frame)


        # If already at target, go to next target
        if distance < self.error_threshold:

            # If done with the path, print it
            if self.target == (len(self.path)-1):
                print(f"  Driver is done.")
                return True, 0.0, 0.0

            # Else recursively call this function
            print(f"\n  Driver - reached target {self.target}, at: {self.path[self.target]}, sensors: ({x_pos}, {y_pos}).")
            self.target += 1
            self.lap = self.target // (len(self.path) // self.number_laps)
            print(f"  Driver - next target: {self.target}, at: {self.path[self.target]}, lap: {self.lap}.")
            return self.follow_path(
                x_pos=x_pos,
                y_pos=y_pos,
                quaternion=quaternion,
            )
        
        # vx proportional to distance
        v_lin = np.clip(0.05 * distance, self.min_driver_speed, self.max_driver_speed)
        
        # Compute the new wz command
        wz = np.clip(1.0 * angle_heading, -self.max_driver_rotation, self.max_driver_rotation)

        # if abs(angle_heading) > np.pi / 2:
        #     # Choose to drive backward
        #     angle_heading = angle_heading - np.pi if angle_heading > 0 else angle_heading + np.pi
        #     angle_heading = (angle_heading + np.pi) % (2 * np.pi) - np.pi
 
        #     v_lin = -np.clip(0.1 * distance, self.min_driver_speed, self.max_driver_speed)
        #     wz = np.clip(2.0 * angle_heading, -self.max_driver_rotation, self.max_driver_rotation)
        # else:
        #     # Drive forward
        #     v_lin = np.clip(0.1 * distance, self.min_driver_speed, self.max_driver_speed)
        #     wz = np.clip(2.0 * angle_heading, -self.max_driver_rotation, self.max_driver_rotation)

        # print(f"Commanded Velocities: {v_lin:.2f} m/s, {wz:.2f} rad/s")


        return False, v_lin, -wz


#########
# FLAIR #

class FLAIR:
    """
    Merge of the GPDataset, GPTraining and Adaptation thread from 
    the real-world pipeline. Based on the same code from 
    src/flair/functionality_controller/, but in an asynchroneous version. 
    """

    def __init__(
        self, adaptation_off: bool,map_elites_map: str, grid_resolution: int, min_command: float, max_command: float
    ) -> None:

        self.adaptation_off = bool(adaptation_off)
        self.min_command = float(min_command)
        self.max_command = float(max_command)
        self.grid_resolution = int(grid_resolution)

        # Create empty DataPoints for addition
        self.datapoints = DataPoints.init(max_capacity=DATAPOINT_BATCH_SIZE, state_dims=8)
        jiting_datapoints = DataPoints.add(
            datapoint=self.datapoints,
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

        # Create the DataCollection object
        self.data_collection = DataCollection(
            logger=PrintLogger,
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

        if ADAPTATION_VERSION == NO_STATE_FAST:

            from functionality_controller.adaptive_gp_no_state_fast import AdaptiveGP

            self.adaptive_gp = AdaptiveGP(
                logger=PrintLogger,
                jiting_datapoints=jiting_datapoints,
                grid_resolution=self.grid_resolution,
                min_command=self.min_command,
                max_command=self.max_command,
                robot_width=ROBOT_WIDTH,
                default_obs_noise=DEFAULT_OBS_NOISE,
                default_lengthscale=DEFAULT_LENGTHSCALE,
                default_variance=DEFAULT_VARIANCE,
                min_diff_datapoint=MIN_DIFF_DATAPOINT,
                use_grid_dataset=USE_GRID_DATASET,
                dataset_size=DATASET_SIZE,
                dataset_grid_cell_size=DATASET_GRID_CELL_SIZE,
                dataset_grid_neighbours=DATASET_GRID_NEIGH,
                dataset_grid_novelty_threshold=DATASET_GRID_NOVELTY_THRESHOLD,
                datapoint_batch_size=DATAPOINT_BATCH_SIZE,
                max_p_value=MAX_P_VALUE,
                p_soft_update_size=P_SOFT_UPDATE_SIZE,
                min_spread=MIN_SPREAD,
                minibatch_size=MINIBATCH_SIZE,
                auto_reset_error_buffer_size=ERROR_BUFFER_SIZE,
                auto_reset_angular_rot_weight=WEIGHT_ANGULAR_ROT,
                auto_reset_threshold=NEW_SCENARIO_THRESHOLD,
            )

        elif ADAPTATION_VERSION == STATE_FAST or ADAPTATION_VERSION == STATE_FAST_GP:

            from functionality_controller.adaptive_gp_state_fast import AdaptiveGP

            self.adaptive_gp = AdaptiveGP(
                logger=PrintLogger,
                jiting_datapoints=jiting_datapoints,
                grid_resolution=self.grid_resolution,
                min_command=self.min_command,
                max_command=self.max_command,
                robot_width=ROBOT_WIDTH,
                default_obs_noise=DEFAULT_OBS_NOISE,
                default_lengthscale=DEFAULT_LENGTHSCALE,
                default_variance=DEFAULT_VARIANCE,
                min_diff_datapoint=MIN_DIFF_DATAPOINT,
                use_grid_dataset=USE_GRID_DATASET,
                dataset_size=DATASET_SIZE,
                dataset_grid_cell_size=DATASET_GRID_CELL_SIZE,
                dataset_grid_neighbours=DATASET_GRID_NEIGH,
                dataset_grid_novelty_threshold=DATASET_GRID_NOVELTY_THRESHOLD,
                datapoint_batch_size=DATAPOINT_BATCH_SIZE,
                max_p_value=MAX_P_VALUE,
                multi_function=MULTI_FUNCTON,
                remove_offset=REMOVE_OFFSET,
                state_dim=STATE_DIM,
                state_min_dataset=STATE_MIN_DATASET,
                state_max_dataset=STATE_MAX_DATASET,
                state_min_opt_clip=STATE_MIN_OPT_CLIP,
                state_max_opt_clip=STATE_MAX_OPT_CLIP,
                p1_min=P1_MIN,
                p1_max=P1_MAX,
                p2_min=P2_MIN,
                p2_max=P2_MAX,
                p3_min=P3_MIN,
                p3_max=P3_MAX,
                minibatch_size=MINIBATCH_SIZE,
                auto_reset_error_buffer_size=ERROR_BUFFER_SIZE,
                auto_reset_angular_rot_weight=WEIGHT_ANGULAR_ROT,
                auto_reset_threshold=NEW_SCENARIO_THRESHOLD,
            )

        else:
            assert 0, "ERROR: unknown Adaptation version"

        # Attributes used to compute clipping
        self.border_idx = compute_borders(self.adaptive_gp.all_descriptors)

        # Init Adaptation
        self.adaptation = FunctionalityController(
            grid_resolution=self.grid_resolution,
            min_command=self.min_command,
            max_command=self.max_command,
            state_dim=STATE_DIM,
            state_min_opt_clip=STATE_MIN_OPT_CLIP,
            state_max_opt_clip=STATE_MAX_OPT_CLIP,
            robot_width=ROBOT_WIDTH,
            max_p_value=MAX_P_VALUE,
        )
        self.scales = [(1.0, 1.0), (0.6, 1.0), (1.0, 0.8)]
        self.scale_id = 0
        self.state = None

        # Pre-jit main functions of Adaptation
        _, _, _, _, _, _, _, = self.adaptation.get_command(
            0.0,
            0.0,
            0.0,
            use_state=(ADAPTATION_VERSION == STATE_FAST),
            use_state_gp=(ADAPTATION_VERSION == STATE_FAST_GP),
        )

        # Generic reset
        self.reset()

    def reset(self) -> None:

        # Reset main objects
        self.data_collection.reset()
        self.adaptive_gp.reset()
        self.adaptation.reset()

        # Attributes
        self.p1 = 0
        self.p2 = 0
        self.a = 0 
        self.b = 0
        self.c = 0
        self.d = 0
        self.offset = 0

    def add_datapoint(
        self,
        state: np.ndarray,
        sensor_time: int,
        #sensor_tx: float,
        #sensor_ty: float,
        #sensor_tz: float,
        sensor_vx: float,
        #sensor_vy: float,
        #sensor_vz: float,
        #sensor_yaw: float,
        #sensor_roll: float,
        #sensor_pitch: float,
        sensor_wx: float,
        sensor_wy: float,
        sensor_wz: float,
        adaptation_cmd_lin_x: float, 
        adaptation_cmd_ang_z: float, 
        gp_prediction_x: float, 
        gp_prediction_y: float, 
        human_cmd_lin_x: float, 
        human_cmd_ang_z: float,
    ) -> None:

        # Pass through the data collection
        (
            final_datapoint,
            final_datapoints_msg,
            metrics,
            error_code,
        ) = self.data_collection.data_collection(
            state=state,
            command_x=adaptation_cmd_lin_x,
            command_y=adaptation_cmd_ang_z,
            gp_prediction_x=gp_prediction_x,
            gp_prediction_y=gp_prediction_y,
            intent_x=human_cmd_lin_x,
            intent_y=human_cmd_ang_z,
            sensor_time=sensor_time,
            sensor_x=sensor_vx,
            sensor_y=sensor_wz,
            sensor_wx=sensor_wx,
            sensor_wy=sensor_wy,
        )

        # Accumulate the datapoints to later update the model
        if final_datapoint is not None:
            self.datapoints = DataPoints.add(
                datapoint=self.datapoints,
                point_id=final_datapoint[0],
                sensor_time_sec=final_datapoint[1],
                sensor_time_nanosec=final_datapoint[2],
                command_time_sec=final_datapoint[3],
                command_time_nanosec=final_datapoint[4],
                state=final_datapoint[14],
                gp_prediction_x=final_datapoint[5],
                gp_prediction_y=final_datapoint[6],
                command_x=final_datapoint[7],
                command_y=final_datapoint[8],
                intent_x=final_datapoint[9],
                intent_y=final_datapoint[10],
                sensor_x=final_datapoint[11],
                sensor_y=final_datapoint[12],
            )

    def train_model(self) -> None:

        # Add all datapoints to the dataset
        self.adaptive_gp.update(self.datapoints)

        # If using auto-reset, apply it
        if USE_RESET:

            # Run the GP ME insertion check
            auto_reset, error_increase, error = self.adaptive_gp.auto_reset()

            # Reset also the other objects
            if auto_reset:
                self.reset()

        # Train the GP
        updated, _, _ = self.adaptive_gp.gp_update()

        # If the model has been updated, update it in Adaptation
        if updated:

            # Computing the new clipping
            max_x, max_y = compute_clipping(
                self.border_idx, self.adaptive_gp.all_corrected_descriptors
            )

            # Update the introspections
            self.p1 = float(self.adaptive_gp.xy_learned_params['mean_function']['rotation'][0])
            self.p2 = float(self.adaptive_gp.xy_learned_params['mean_function']['rotation'][1])
            self.a = self.adaptive_gp.xy_learned_params['mean_function']['rotation'][2]
            self.b = self.adaptive_gp.xy_learned_params['mean_function']['rotation'][3]
            self.c = self.adaptive_gp.xy_learned_params['mean_function']['rotation'][4]
            self.d = self.adaptive_gp.xy_learned_params['mean_function']['rotation'][5]
            self.offset = self.adaptive_gp.xy_learned_params['mean_function']['rotation'][6]

            # Set the new model in adaptation
            if ADAPTATION_VERSION == STATE_FAST_GP:
                send_dataset_state = self.adaptive_gp.dataset.state
            else:
                send_dataset_state = None
            self.adaptation.all_corrected_descriptors = self.adaptive_gp.all_corrected_descriptors
            self.adaptation.uncertainties = self.adaptive_gp.uncertainties
            self.adaptation.learned_params = self.adaptive_gp.xy_learned_params["mean_function"]["rotation"]
            self.adaptation.cov_alpha = self.adaptive_gp.xy_learned_params["mean_function"]["offset"]
            self.adaptation.max_x = max_x
            self.adaptation.max_y = max_y

            self.adaptation.dataset_state = send_dataset_state
            self.adaptation.kernel_x = self.adaptive_gp.kernel_x


    def get_command(
        self, 
        human_cmd_lin_x: float, 
        human_cmd_ang_z: float, 
        state: np.ndarray,
    ) -> Tuple[float, float, float, float, float, float]:

        # Get command from adaptation
        self.adaptation.state = jnp.array(state)
        (
            gp_prediction_x,
            gp_prediction_y,
            _,
            descriptor_x,
            descriptor_z,
            uncertainty_x,
            uncertainty_y,
        ) = self.adaptation.get_command(
            jnp.array(human_cmd_lin_x),
            jnp.array(human_cmd_ang_z),
            0.0,
            use_state=(ADAPTATION_VERSION == STATE_FAST),
            use_state_gp=(ADAPTATION_VERSION == STATE_FAST_GP),
        )

        # If the Adaptation is off, just pass commands through
        if self.adaptation_off:
            command_x = human_cmd_lin_x
            command_y = human_cmd_ang_z

        # If the Adaptation is on, apply the Adaptation correction
        else:
            command_x = gp_prediction_x
            command_y = gp_prediction_y

        # Return final msg
        return command_x, command_y, gp_prediction_x, gp_prediction_y, human_cmd_lin_x, human_cmd_ang_z


#######################
# Environment Manager #

class EnvironmentManager:
    """
    Wrap the Brax environment to give the necessary
    methods and attributes. 
    """

    def __init__(self, map_elites_map: str, sensor_freq: float) -> None:

        # Create a random key
        random_seed = int(time.time() * 1e6) % (2**32)
        self.random_key = jax.random.PRNGKey(random_seed)

        # Load the config of the considered replication
        with open(f"{map_elites_map}/config.csv", mode='r', newline='') as file:
            reader = csv.DictReader(file)
            self.env_config = [row for row in reader]
            if len(self.env_config) > 1:
                print(f"{len(self.env_config)} runs of MAP-Elites, keeping thr first one.")
            self.env_config = self.env_config[0]
        self.env_name = "hexapod_no_reward_velocity" # Avoid the reward computation at each timestep
        print(f"  Initialising the environment: {self.env_name}.")

        # Create the environment
        # Set episode_length None to not end the environment 
        self.random_key, subkey = jax.random.split(self.random_key)
        (
            self.env,
            _,
            init_policies_fn,
            policy_structure,
            min_bd,
            max_bd,
            _,
            subkey,
        ) = set_up_hexapod(
            env_name=self.env_name,
            episode_length=None,
            batch_size=None,
            random_key=subkey,
        )

        # Infer number of repetitions
        self.dt = self.env.sys.config.dt
        #self.repetitions = max(round((1 / sensor_freq) / self.dt), 1)
        #print(f"\n    Returning a sensor every {1 / sensor_freq} and using simulation dt of {self.dt}.")
        self.repetitions = 5 # Chosen from RTE to give enough time to the sinusoidal controllers
        print(f"    Repeating the environment {self.repetitions} times between sensor readings.")

        # Get the grid resolution
        grid_shape = [int(x) for x in self.env_config["euclidean_grid_shape"].split("_")]
        assert len(grid_shape) == 2, "!!!ERROR!!! grid bd should be vx and wz."
        assert grid_shape[0] == grid_shape[1], "!!!ERROR!!! require same discretisation on all dimensions."
        self.grid_resolution = grid_shape[0]

        # Get the grid limits
        assert min_bd[0] == min_bd[1], "!!!ERROR!!! grid should be squared."
        assert max_bd[0] == max_bd[1], "!!!ERROR!!! grid should be squared."
        self.min_command = min_bd[0]
        self.max_command = max_bd[0]
        print(f"\n    Using grid with resolution: {self.grid_resolution} and min/max: {self.min_command}, {self.max_command}.")

        # Getting the path to the map from MAP-Elites
        results_repertoire = self.env_config["results_repertoire"]
        if results_repertoire.rfind("/") == len(results_repertoire) - 1:
            results_repertoire = results_repertoire[:-1]
        results_repertoire = results_repertoire[results_repertoire.rfind("/") + 1 :]
        results_repertoire = os.path.join(map_elites_map, results_repertoire)
        results_repertoire = results_repertoire + "/"
        print(f"    Loading the repertoire in {results_repertoire}.")

        # Loading the map from MAP-Elites
        init_policies, subkey = init_policies_fn(1, subkey)
        init_policy = jax.tree_map(lambda x: x[0], init_policies)
        _, reconstruction_fn = jax.flatten_util.ravel_pytree(init_policy)
        self.repertoire = MapElitesRepertoire.load(
            reconstruction_fn=reconstruction_fn,
            path=results_repertoire,
        )

        # Prepare the controller structure
        self.cmd_lin_x = None
        self.cmd_ang_z = None
        self.params = None
        self.timestep = 0

        # Prepare the evaluation functions
        self.inference_fn = jax.jit(policy_structure.apply)
        self.reset_fn = jax.jit(self.env.reset)
        self.step_fn = jax.jit(self.env.step)
        self.random_key, subkey = jax.random.split(self.random_key)
        self.env_state = self.reset_fn(subkey)

        #@partial(jax.jit, static_argnames=("play_step_fn", "length"))
        #def generate_unroll_time(
        #    env_state: Any,
        #    timestep: int,
        #    policy_params: jnp.ndarray,
        #    random_key: jax.random.key,
        #    length: int,
        #    play_step_fn: Any,
        #) -> Tuple[Any, jnp.ndarray, int]:

        #    def _scan_play_step_fn(
        #        carry: Tuple[Any, jnp.ndarray, jax.random.key, int], unused_arg: Any
        #    ) -> Tuple[Tuple[Any, jnp.ndarray, jax.random.key, int], jnp.ndarray]:
        #        env_state, policy_params, random_key, transitions, timestep = play_step_fn(
        #            *carry
        #        )
        #        return (env_state, policy_params, random_key, timestep), transitions

        #    (state, _, _, timestep), transitions = jax.lax.scan(
        #        _scan_play_step_fn,
        #        (env_state, policy_params, random_key, timestep),  
        #        (),
        #        length=length,
        #    )
        #    return state, transitions, timestep

        #self.generate_unroll_time = jax.jit(partial(
        #    generate_unroll_time, 
        #    length=self.repetitions, 
        #    play_step_fn=play_step_fn,
        #))


    def get_grid_details(self) -> Tuple[np.ndarray, float, float]:
        return self.grid_resolution, self.min_command, self.max_command

    def reset(self) -> Any:
        self.random_key, subkey = jax.random.split(self.random_key)
        self.env_state = self.reset_fn(subkey)
        return self.env_state

    def step(self, cmd_lin_x: float, cmd_ang_z: float) -> Any:

        # Get the corresponding controller from the map
        if self.cmd_lin_x == None or self.cmd_ang_z == None or cmd_lin_x != self.cmd_lin_x or cmd_ang_z != self.cmd_ang_z:
            batch_of_descriptors = jnp.expand_dims(jnp.asarray([cmd_lin_x, cmd_ang_z]), axis=0)
            indices = get_cells_indices(
                batch_of_descriptors=batch_of_descriptors, 
                centroids=self.repertoire.centroids,
            )
            self.params = jax.tree_util.tree_map(
                lambda x: x[indices].squeeze(), self.repertoire.genotypes
            )
            self.cmd_lin_x = cmd_lin_x
            self.cmd_ang_z = cmd_ang_z
            self.timestep = 0

        for _ in range (self.repetitions):

            # Get the action
            action = self.inference_fn(self.params, self.env_state, self.timestep)
            self.timestep += 1

            # Apply it in the environment
            self.env_state = self.step_fn(self.env_state, action)

        return self.env_state

    def get_sensor(self) -> Tuple:

        qp = self.env_state.qp

        # Position of the torso
        sensor_tx, sensor_ty, sensor_tz = qp.pos[0]

        # Velocity of the torso
        sensor_vx, sensor_vy, sensor_vz = qp.vel[0]

        # Angle of the torso
        quaternion = qp.rot[0]
        sensor_yaw, sensor_roll, sensor_pitch = quat_to_euler(quaternion)

        # Angular velocity of the torso
        sensor_wx, sensor_wy, sensor_wz = qp.ang[0]

        # Build the state similarly to main robot pipeline
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
        

        return (
            state,
            quaternion,
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

########################
# Perturbation Manager #

class PerturbationManager:

    def __init__(
        self,
        wheel_base: float,
        wheel_radius: float,
        wheel_max_velocity: float,
    ) -> None:

        self.wheel_base = wheel_base
        self.wheel_radius = wheel_radius
        self.wheel_max_velocity = wheel_max_velocity

        # Initialise perturbation values
        self.track_scaling = False
        self.left_track_scaling = 1.0
        self.right_track_scaling = 1.0
        self.wind = False

    def new_perturbation(
        self, 
        left_scaling: float,
        right_scaling: float,
        wind: bool,
        state: np.ndarray
    ) -> None:
        """Update the perturbation to apply."""

        # Clip the scaling (same as on real robot)
        new_left_track_scaling = np.clip(left_scaling, 0.5, 1)
        new_right_track_scaling = np.clip(right_scaling, 0.5, 1)

        # Detect scaling change
        if (
            self.left_track_scaling != new_left_track_scaling
            or self.right_track_scaling != new_right_track_scaling
        ):

            self.left_track_scaling = new_left_track_scaling
            self.right_track_scaling = new_right_track_scaling
            if self.left_track_scaling != 1 or self.right_track_scaling != 1:
                print(f"Starting scaling perturbation.")
                self.track_scaling = True
            else:
                print(f"Stopping scaling perturbation.")
                self.track_scaling = False

        # Detect wind change
        if self.wind != wind:
            if wind:
                print(f"Starting wind perturbation.")
            else:
                print(f"Stopping wind perturbation.")
            self.wind = wind

    def apply_perturbation(
        self, 
        cmd_lin_x: float, 
        cmd_ang_z: float, 
        state: Tuple
    ) -> Tuple[float, float]:
        """Apply the corruption due to any perturbation."""

        if self.wind:
            return self._corrupt_wind(cmd_lin_x=cmd_lin_x, cmd_ang_z=cmd_ang_z, angle=current_state[5])
        elif self.track_scaling:
            return self._corrupt_state(cmd_lin_x=cmd_lin_x, cmd_ang_z=cmd_ang_z)
        return cmd_lin_x, cmd_ang_z 

    def _corrupt_state(
        self, cmd_lin_x: float, cmd_ang_z: float,
    ) -> Tuple[float, float]:

        lin = cmd_lin_x
        ang = -cmd_ang_z

        left_vel = (lin - ang * self.wheel_base / 2) / self.wheel_radius
        right_vel = (lin + ang * self.wheel_base / 2) / self.wheel_radius

        left_vel *= self.left_track_scaling
        right_vel *= self.right_track_scaling

        if abs(left_vel) >= abs(right_vel):
            if abs(left_vel) > self.wheel_max_velocity:
                ratio = right_vel / left_vel
                left_vel = np.clip(
                    left_vel, -self.wheel_max_velocity, self.wheel_max_velocity
                )
                right_vel = left_vel * ratio
        else:
            if abs(right_vel) > self.wheel_max_velocity:
                ratio = left_vel / right_vel
                right_vel = np.clip(
                    right_vel, -self.wheel_max_velocity, self.wheel_max_velocity
                )
                left_vel = right_vel * ratio

        linear_vel_corrupted = (left_vel + right_vel) * 0.5 * self.wheel_radius
        angular_vel_corrupted = (
            (right_vel - left_vel) * self.wheel_radius * (1 / self.wheel_base)
        )

        return linear_vel_corrupted, -angular_vel_corrupted

    def _corrupt_wind(
        self, cmd_lin_x: float, cmd_ang_z: float, angle: float,
    ) -> Tuple[float, float]:

        point = 0.0
        error = point - angle
        error *= 0.6/np.pi
        perturbation = np.clip(abs(error), 0, 0.6)

        if error < 0:
            old_perturbation = self.left_track_scaling
            self.left_track_scaling = 1.0 - perturbation
        else:
            old_perturbation = self.right_track_scaling
            self.right_track_scaling = 1.0 - perturbation

        cmd_lin_x, cmd_ang_z = self._corrupt_state(cmd_lin_x=cmd_lin_x, cmd_ang_z=cmd_ang_z)
        if error < 0:
            self.left_track_scaling = old_perturbation
        else:
            self.right_track_scaling = old_perturbation

        return cmd_lin_x, cmd_ang_z



###################
# Metrics Manager #

class MetricManager:
    """
    Save all the configs and metrics, similar to the role of
    Influx in the real robotics pipeline. 
    """

    def __init__(
        self, results: str, adaptation_off: bool, circuit: str, perturbation_off: bool, save_html: bool
    ) -> None:
        self.folder = results
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)
        self.adaptation_off = adaptation_off
        self.circuit = circuit
        self.perturbation_off = perturbation_off
        self.save_html = save_html

        # Naming based on configuration
        self.common_file_name = "flair"
        if self.adaptation_off:
            self.common_file_name = "no_flair"
        if self.perturbation_off:
            self.common_file_name += "_no_perturb"
        self.common_file_name + f"_{circuit}"

        # Naming for dataframe, following convention from get_datas
        self.rep_name = ""
        if self.perturbation_off:
            self.rep_name += "NO_perturbation_"
        elif circuit == "chicane_static":
            self.rep_name += "Static_0.7_"
        elif circuit == "chicane_dynamic":
            self.rep_name += "Dynamic_Value_0.4_0.7_"
        if adaptation_off:
            self.rep_name += "Adaptation_OFF"
        else:
            self.rep_name += "Adaptation_ON"

        # Create a csv to save common dataframe
        self.empty_metrics()
        main_metrics = self.create_main_metrics(0, 0, 0, "", 0, "", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.main_file_name = f"{self.folder}/hexapod_main_{self.common_file_name}.csv"
        self._create_metrics(
            file_name=self.main_file_name,
            metrics=main_metrics,
            name="main",
        )

    def empty_metrics(self) -> None:
        self.main_metrics = {}
        self.gp_damage_introspection_metrics = {}
        self.learnt_state_functions_metrics = {}
        self.adaptation_metrics = {}

    def create_main_metrics(
        self, 
        timing: int,
        timestep: int,
        rep: int,
        damage_type: str,
        scaling_value: float,
        section: str,
        #section_start_time: int,
        #section_end_time: int,
        lap: int,
        #lap_start_time: int,
        #lap_end_time: int,
        target_id: int,
        target_tx: float,
        target_ty: float,
        tx: float,
        ty: float,
        human_cmd_lin_x: float,
        human_cmd_ang_z: float,
        vx: float,
        wz: float,
    ) -> Dict:

        # Same content as the final dataframe used for analysis
        main_metrics = {
            "Time": [timing],
            "Timesteps": [timestep],
            "Reps": [self.rep_name],
            "Damage_Type": [damage_type],
            "Scaling_Value": [scaling_value],
            "Sections": [section],
            "Sections_index": [rep],
            #"Sections_start_time": [section_start_time],
            #"Sections_end_time": [section_end_time],
            "Laps": [lap],
            #"Laps_start_time": [lap_start_time],
            #"Laps_end_time": [lap_end_time],
            "target_id": [target_id],
            "target_tx": [target_tx],
            "target_ty": [target_ty],
            "tx": [tx],
            "ty": [ty],
            "human_cmd_lin_x": [human_cmd_lin_x],
            "human_cmd_ang_z": [human_cmd_ang_z],
            "vx": [vx],
            "wz": [wz],
        }
        return main_metrics

    def add_main_metrics(
        self, 
        main_metrics,
        timing: int,
        timestep: int,
        rep: int,
        damage_type: str,
        scaling_value: float,
        section: str,
        lap: int,
        target_id: int,
        target_tx: float,
        target_ty: float,
        tx: float,
        ty: float,
        human_cmd_lin_x: float,
        human_cmd_ang_z: float,
        vx: float,
        wz: float,
    ) -> Dict:

        new_main_metrics = self.create_main_metrics(
            timing=timing,
            timestep=timestep,
            rep=rep,
            damage_type=damage_type,
            scaling_value=scaling_value,
            section=section,
            lap=lap,
            target_id=target_id,
            target_tx=target_tx,
            target_ty=target_ty,
            tx=tx,
            ty=ty,
            human_cmd_lin_x=human_cmd_lin_x,
            human_cmd_ang_z=human_cmd_ang_z,
            vx=vx,
            wz=wz,
        )
        main_metrics = {k: main_metrics[k] + new_main_metrics[k] for k in main_metrics}
        return main_metrics


    def create_adaptation_metrics(
        self, 
        p1: float,
        p2: float,
        a: float,
        b: float,
        c: float,
        d: float,
        offset: float,
        human_cmd_lin_x: float,
        human_cmd_ang_z: float,
        adaptation_cmd_lin_x: float,
        adaptation_cmd_ang_z: float,
        perturbation_cmd_lin_x: float,
        perturbation_cmd_ang_z: float,
    ) -> Tuple[Dict, Dict, Dict]:

        # Same content as /gp_damage_introspection in influx
        gp_damage_introspection_metrics = {
            "p1": [p1],
            "p2": [p2],
        }

        # Same content as /learnt_state_functions in influx
        learnt_state_functions_metrics = {
            "a": [a],
            "b": [b],
            "c": [c],
            "d": [d],
            "offset": [offset],
        }

        # Same content as /adaptation in influx 
        # (missing some time-related field as this is asynchroneous)
        adaptation_metrics = {
            "human_cmd_lin_x": [human_cmd_lin_x],
            "human_cmd_ang_z": [human_cmd_ang_z],
            "adaptation_cmd_lin_x": [adaptation_cmd_lin_x],
            "adaptation_cmd_ang_z": [adaptation_cmd_ang_z],
            "perturbation_cmd_lin_x": [perturbation_cmd_lin_x],
            "perturbation_cmd_ang_z": [perturbation_cmd_ang_z],
        }

        return gp_damage_introspection_metrics, learnt_state_functions_metrics, adaptation_metrics

    def add_adaptation_metrics(
        self, 
        gp_damage_introspection_metrics: Dict,
        learnt_state_functions_metrics: Dict,
        adaptation_metrics: Dict,
        p1: float,
        p2: float,
        a: float,
        b: float,
        c: float,
        d: float,
        offset: float,
        human_cmd_lin_x: float,
        human_cmd_ang_z: float,
        adaptation_cmd_lin_x: float,
        adaptation_cmd_ang_z: float,
        perturbation_cmd_lin_x: float,
        perturbation_cmd_ang_z: float,
    ) -> Tuple[Dict, Dict, Dict]:

        (
            new_gp_damage_introspection_metrics, 
            new_learnt_state_functions_metrics, 
            new_adaptation_metrics,
        ) = self.create_adaptation_metrics(
            p1=p1,
            p2=p2,
            a=a,
            b=b,
            c=c,
            d=d,
            offset=offset,
            human_cmd_lin_x=human_cmd_lin_x,
            human_cmd_ang_z=human_cmd_ang_z,
            adaptation_cmd_lin_x=adaptation_cmd_lin_x,
            adaptation_cmd_ang_z=adaptation_cmd_ang_z,
            perturbation_cmd_lin_x=perturbation_cmd_lin_x,
            perturbation_cmd_ang_z=perturbation_cmd_ang_z,
        )
        gp_damage_introspection_metrics = {k: gp_damage_introspection_metrics[k] + new_gp_damage_introspection_metrics[k] for k in gp_damage_introspection_metrics}
        learnt_state_functions_metrics = {k: learnt_state_functions_metrics[k] + new_learnt_state_functions_metrics[k] for k in learnt_state_functions_metrics}
        adaptation_metrics = {k: adaptation_metrics[k] + new_adaptation_metrics[k] for k in adaptation_metrics}
        return gp_damage_introspection_metrics, learnt_state_functions_metrics, adaptation_metrics


    def create_metrics_rep(self, rep: int) -> None:

        self.empty_metrics()

        # Main metrics
        main_metrics = self.create_main_metrics(0, 0, 0, "", 0, "", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.rep_main_file_name = f"{self.folder}/hexapod_{self.circuit}_replication_main_{rep}.csv"
        self._create_metrics(
            file_name=self.rep_main_file_name,
            metrics=main_metrics,
            name="main",
        )

        # Adaptation metrics
        (
            gp_damage_introspection_metrics, 
            learnt_state_functions_metrics, 
            adaptation_metrics ,
        ) = self.create_adaptation_metrics(
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        )

        self.rep_gp_damage_introspection_file_name = (
            f"{self.folder}/hexapod_{self.circuit}_replication_gp_damage_introspection_{rep}.csv"
        )
        self._create_metrics(
            file_name=self.rep_gp_damage_introspection_file_name,
            metrics=gp_damage_introspection_metrics,
            name="gp_damage_introspection",
        )

        self.rep_learnt_state_functions_file_name = (
            f"{self.folder}/hexapod_{self.circuit}_replication_learnt_state_functions_{rep}.csv"
        )
        self._create_metrics(
            file_name=self.rep_learnt_state_functions_file_name,
            metrics=learnt_state_functions_metrics,
            name="learnt_state_functions",
        )

        self.rep_adaptation_file_name = (
            f"{self.folder}/hexapod_{self.circuit}_replication_adaptation_{rep}.csv"
        )
        self._create_metrics(
            file_name=self.rep_adaptation_file_name,
            metrics=adaptation_metrics,
            name="adaptation",
        )

        # If saving html, create a folder to save data
        if self.save_html:
            self.rep_html_file_name = f"{self.folder}/hexapod_{self.circuit}_replication_html_{rep}.html"
            self.rollout = []
            print(f"  Saving html in {self.rep_html_file_name}.")
    
    def save_metrics_rep(self, env_sys: Any) -> None:

        # Save all metrics
        self._save_metrics(
            file_name=self.main_file_name,
            metrics=self.main_metrics,
        )
        self._save_metrics(
            file_name=self.rep_main_file_name,
            metrics=self.main_metrics,
        )
        self._save_metrics(
            file_name=self.rep_gp_damage_introspection_file_name,
            metrics=self.gp_damage_introspection_metrics,
        )
        self._save_metrics(
            file_name=self.rep_learnt_state_functions_file_name,
            metrics=self.learnt_state_functions_metrics,
        )
        self._save_metrics(
            file_name=self.rep_adaptation_file_name,
            metrics=self.adaptation_metrics,
        )

        plt.figure()
        plt.plot(self.main_metrics["tx"], self.main_metrics["ty"], marker='o', label="Sensors")
        plt.plot(self.main_metrics["target_tx"], self.main_metrics["target_ty"], marker='x', label="Targets")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{self.folder}/hexapod_{self.circuit}_replication_trajectory_{rep}.png")

        plt.figure()
        plt.plot(self.main_metrics["vx"], marker='o', label="Sensors Vx")
        plt.plot(self.main_metrics["human_cmd_lin_x"], marker='x', label="Command Vx")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{self.folder}/hexapod_{self.circuit}_replication_vx_{rep}.png")

        plt.figure()
        plt.plot(self.main_metrics["wz"], marker='o', label="Sensors Wz")
        plt.plot(self.main_metrics["human_cmd_ang_z"], marker='x', label="Command Wz")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{self.folder}/hexapod_{self.circuit}_replication_wz_{rep}.png")


        # Save html
        if self.save_html:
            html_file = html.render(env_sys, [s.qp for s in self.rollout])
            f = open(self.rep_html_file_name, "w")
            f.write(html_file)
            f.close()

        # Empty everything
        self.empty_metrics()


    def add_main_metrics_rep(
        self, 
        env_state: Any, 
        timing: int,
        timestep: int,
        rep: int,
        damage_type: str,
        scaling_value: float,
        section: str,
        lap: int,
        target_id: int,
        target_tx: float,
        target_ty: float,
        tx: float,
        ty: float,
        human_cmd_lin_x: float,
        human_cmd_ang_z: float,
        vx: float,
        wz: float,
    ) -> None:

        if self.main_metrics == {}:
            self.main_metrics = self.create_main_metrics(
                timing=timing,
                timestep=timestep,
                rep=rep,
                damage_type=damage_type,
                scaling_value=scaling_value,
                section=section,
                lap=lap,
                target_id=target_id,
                target_tx=target_tx,
                target_ty=target_ty,
                tx=tx,
                ty=ty,
                human_cmd_lin_x=human_cmd_lin_x,
                human_cmd_ang_z=human_cmd_ang_z,
                vx=vx,
                wz=wz,
            )
        else:
            self.main_metrics = self.add_main_metrics(
                main_metrics=self.main_metrics,
                timing=timing,
                timestep=timestep,
                rep=rep,
                damage_type=damage_type,
                scaling_value=scaling_value,
                section=section,
                lap=lap,
                target_id=target_id,
                target_tx=target_tx,
                target_ty=target_ty,
                tx=tx,
                ty=ty,
                human_cmd_lin_x=human_cmd_lin_x,
                human_cmd_ang_z=human_cmd_ang_z,
                vx=vx,
                wz=wz,
            )
        
        # If saving html
        if self.save_html:
            self.rollout.append(env_state)

    def add_adaptation_metrics_rep(
        self, 
        p1: float,
        p2: float,
        a: float,
        b: float,
        c: float,
        d: float,
        offset: float,
        human_cmd_lin_x: float,
        human_cmd_ang_z: float,
        adaptation_cmd_lin_x: float,
        adaptation_cmd_ang_z: float,
        perturbation_cmd_lin_x: float,
        perturbation_cmd_ang_z: float,
    ) -> None:

        if self.gp_damage_introspection_metrics == {}:
            (
                self.gp_damage_introspection_metrics, 
                self.learnt_state_functions_metrics, 
                self.adaptation_metrics,
            ) = self.create_adaptation_metrics(
                p1=p1,
                p2=p2,
                a=a,
                b=b,
                c=c,
                d=d,
                offset=offset,
                human_cmd_lin_x=human_cmd_lin_x,
                human_cmd_ang_z=human_cmd_ang_z,
                adaptation_cmd_lin_x=adaptation_cmd_lin_x,
                adaptation_cmd_ang_z=adaptation_cmd_ang_z,
                perturbation_cmd_lin_x=perturbation_cmd_lin_x,
                perturbation_cmd_ang_z=perturbation_cmd_ang_z,
            )
        else:
            (
                self.gp_damage_introspection_metrics, 
                self.learnt_state_functions_metrics, 
                self.adaptation_metrics,
            ) = self.add_adaptation_metrics(
                gp_damage_introspection_metrics=self.gp_damage_introspection_metrics,
                learnt_state_functions_metrics=self.learnt_state_functions_metrics,
                adaptation_metrics=self.adaptation_metrics,
                p1=p1,
                p2=p2,
                a=a,
                b=b,
                c=c,
                d=d,
                offset=offset,
                human_cmd_lin_x=human_cmd_lin_x,
                human_cmd_ang_z=human_cmd_ang_z,
                adaptation_cmd_lin_x=adaptation_cmd_lin_x,
                adaptation_cmd_ang_z=adaptation_cmd_ang_z,
                perturbation_cmd_lin_x=perturbation_cmd_lin_x,
                perturbation_cmd_ang_z=perturbation_cmd_ang_z,
            )

    def save_gp_collection_configuration(self) -> None:
        """Same content as /gp_collection_configuration in influx."""

        config = {
            "Filter_transition": FILTER_TRANSITION,
            "Filter_varying_angle": FILTER_VARYING_ANGLE,
            "Filter_turning_only": FILTER_TURNING_ONLY,
            "Buffer_size": BUFFER_SIZE,
            "Min_delay": MIN_DELAY,
            "Max_delay": MAX_DELAY,
            "Selection_size": SELECTION_SIZE,
            "IQC_Q1": IQC_Q1,
            "IQC_Qn": IQC_Qn,
            "Filter_transition_size": FILTER_TRANSITION_SIZE,
            "Filter_transition_tolerance": FILTER_TRANSITION_TOLERANCE,
            "Filter_varying_angle_size": FILTER_VARYING_ANGLE_SIZE,
            "Filter_varying_angle_tolerance": FILTER_VARYING_ANGLE_TOLERANCE,
            "Filter_turning_only_tolerance": FILTER_TURNING_ONLY_TOLERANCE,
        }
        self._save_configuration(
            file_name=f"{self.folder}/gp_collection_configuration.csv",
            config=config,
        )

    def save_gp_training_configuration(
        self, grid_resolution: int, min_command: float, max_command: float
    ) -> None:
        """Same content as /gp_training_configuration in influx."""

        config = {
            "Version": ADAPTATION_VERSION,
            "Use_reset": USE_RESET,
            "GP_obs_noise": str(DEFAULT_OBS_NOISE),
            "GP_variance": str(DEFAULT_VARIANCE),
            "GP_lengthscale": str(DEFAULT_LENGTHSCALE),
            "Grid_resolution": grid_resolution,
            "Min_command": min_command,
            "Max_command": max_command,
            "Dataset_min_diff": MIN_DIFF_DATAPOINT,
            "Dataset_batch_size": DATAPOINT_BATCH_SIZE,
            "Dataset_use_grid": USE_GRID_DATASET,
            "Dataset_size": DATASET_SIZE,
            "Dataset_grid_cell_size": DATASET_GRID_CELL_SIZE,
            "Dataset_grid_neigh": DATASET_GRID_NEIGH,
            "Dataset_grid_novelty_threshold": DATASET_GRID_NOVELTY_THRESHOLD,
            "Max_p_value": MAX_P_VALUE,
            "p_soft_update": P_SOFT_UPDATE_SIZE,
            "Min_spread": MIN_SPREAD,
            "State_multi_function": MULTI_FUNCTON,
            "State_remove_offset": REMOVE_OFFSET,
            "State_dim": STATE_DIM,
            "State_min_dataset": STATE_MIN_DATASET,
            "State_max_dataset": STATE_MAX_DATASET,
            "State_min_opt_clip": STATE_MIN_OPT_CLIP,
            "State_max_opt_clip": STATE_MAX_OPT_CLIP,
            "State_p1_min": P1_MIN,
            "State_p1_max": P1_MAX,
            "State_p2_min": P2_MIN,
            "State_p2_max": P2_MAX,
            "State_p3_min": P3_MIN,
            "State_p3_max": P3_MAX,
            "Reset_minibatch_size": MINIBATCH_SIZE,
            "Reset_error_buffer_size": ERROR_BUFFER_SIZE,
            "Reset_weight_angular_rot": WEIGHT_ANGULAR_ROT,
            "Reset_new_scenario_threshold": NEW_SCENARIO_THRESHOLD,
        }
        self._save_configuration(
            file_name=f"{self.folder}/gp_training_configuration.csv",
            config=config,
        )

    def save_driver_configuration(
        self, 
        min_driver_speed: float, 
        max_driver_speed: float, 
        path: np.ndarray, 
        scaling: np.ndarray,
        scaling_side: np.ndarray,
        scaling_amplitude: np.ndarray, 
        wind: np.ndarray,
        broken_leg: np.ndarray,
        broken_leg_index: np.ndarray,
    ) -> None:
        """Same content as /vicon_configuration in influx."""

        config = {
            "path": str(path),
            "small_loop": str(wind_path * wind_number_laps),
            "chicane": str(chicane_path * chicane_number_laps),
            "scaling": str(scaling),
            "scaling_sides": str(scaling_side),
            "scaling_amplitudes": str(scaling_amplitude),
            "broken_leg": str(broken_leg),
            "broken_leg_index": str(broken_leg_index),
            "wind": str(wind),
            "adaptation_on": not(self.adaptation_off),
            "min_speed": min_driver_speed,
            "max_speed": max_driver_speed,
        }
        self._save_configuration(
            file_name=f"{self.folder}/vicon_configuration.csv",
            config=config,
        )

    def _save_configuration(self, file_name: str, config: Dict) -> None:
        print(f"  Saving configuration in {file_name}.")
        with open(file_name, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(config.keys())
            writer.writerow(config.values())

    def _create_metrics(self, file_name: str, metrics: Dict, name: str) -> None:
        print(f"  Saving {name} metrics in {file_name}.")
        with open(file_name, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(metrics.keys())

    def _save_metrics(self, file_name: str, metrics: Dict) -> None:
        with open(file_name, mode="a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=metrics.keys())
            rows = zip(*metrics.values())
            for row in rows:
                writer.writerow(dict(zip(metrics.keys(), row)))


########
# Main #

if __name__ == "__main__":
    """
    Unlike for the real robot, this code is not made for real time.
    Thus, it does not use ROS nor multiprocessing but perform all steps sequentially.
    """

    #############
    # Arguments #

    parser = argparse.ArgumentParser()

    # Result folder to load the map from
    parser.add_argument("--map-elites-map", default="map_elites_map", type=str)
    parser.add_argument("--results", default="example_results", type=str)

    # Adaptation On or Off
    parser.add_argument("--adaptation-off", action="store_true")

    # Frequency set to the same ratio as the real robot
    parser.add_argument("--sensor-freq", default=50, type=float)
    parser.add_argument("--command-freq", default=10, type=float)

    # As not asynchroneous, set reasonable model training frequency
    parser.add_argument("--model-training-freq", default=2, type=float)

    # Circuit: chicane static, chicane dynamic or wind
    parser.add_argument("--circuit", default="chicane_static", type=str)
    parser.add_argument("--perturbation-off", action="store_true")

    # Number of reps
    parser.add_argument("--num-reps", default=1, type=int)

    # Save video
    parser.add_argument("--save-html", action="store_true")

    args = parser.parse_args()
    args.command_sensor_ratio = int(args.sensor_freq / args.command_freq)
    args.model_command_ratio = int(args.command_freq / args.model_training_freq)

    ##################
    # Initialisation #

    print("\nInitialising the pipeline.")
    start_t = time.time()

    # Metric Manager initialisation
    metric_manager = MetricManager(
        results=args.results, 
        adaptation_off=args.adaptation_off,
        circuit=args.circuit, 
        perturbation_off=args.perturbation_off,
        save_html=args.save_html,
    )

    # Circuit initialisation 
    if args.circuit == "chicane_static":
        path = chicane_path * chicane_number_laps
        number_laps = chicane_number_laps
        section = "Chicane"
    elif args.circuit == "chicane_dynamic":
        path = chicane_path * chicane_number_laps
        number_laps = chicane_number_laps
        section = "Chicane"
    elif args.circuit == "chicane_broken_leg":
        path = chicane_path * chicane_number_laps
        number_laps = chicane_number_laps
        section = "Chicane"
    elif args.circuit == "wind":
        path = wind_path * wind_number_laps
        number_laps = wind_number_laps
        section = "Wind"
    else:
        assert 0, "!!!ERROR!!! Undefined circuit."

    # Perturbation initialisation
    scaling = [False for way_point in range (len(path))]
    scaling_side = ["right" for way_point in range (len(path))]
    broken_leg = [False for way_point in range (len(path))]
    broken_leg_index = [4 for way_point in range (len(path))]
    scaling_amplitude = [0.7 for way_point in range (len(path))]
    wind = [False for way_point in range (len(path))]
    if not(args.perturbation_off):
        if args.circuit == "chicane_static":
            scaling = [True for way_point in range (len(path))]
            damage_type = "static_scaling"
            scaling_value = 0.7
        elif args.circuit == "chicane_dynamic":
            scaling = [True for way_point in range (len(path))]
            scaling_amplitude = [0.7] * 5 + [0.4] * 12
            damage_type = "dynamic_value_scaling"
            scaling_value = 0.7
        elif args.circuit == "chicane_broken_leg":
            broken_leg = [True for way_point in range (len(path))]
            broken_leg_index = [4 for way_point in range (len(path))]
            damage_type = "broken_leg"
            scaling_value = 4
        elif args.circuit == "wind":
            wind = [True for way_point in range (len(path))]
            damage_type = "wind"
            scaling_value = 1.0
        else:
            assert 0, "!!!ERROR!!! Undefined circuit."
    else:
        damage_type = "none"
        scaling_value = 1.0

    # Environment initialisation 
    env_manager = EnvironmentManager(map_elites_map=args.map_elites_map, sensor_freq=args.sensor_freq)
    grid_resolution, min_command, max_command = env_manager.get_grid_details()
    
    """
    # Some timing debug
    start_t = time.time()
    env_state = env_manager.env.reset(env_manager.random_key)
    print(f"\n\nReset took {time.time() - start_t}.")

    cmd_lin_x = 0.0
    cmd_ang_z = 0.1

    start_t = time.time()
    batch_of_descriptors = jnp.expand_dims(jnp.asarray([cmd_lin_x, cmd_ang_z]), axis=0)
    indices = get_cells_indices(
        batch_of_descriptors=batch_of_descriptors, 
        centroids=env_manager.repertoire.centroids,
    )
    params = jax.tree_util.tree_map(
        lambda x: x[indices].squeeze(), env_manager.repertoire.genotypes
    )
    timestep = 0
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready(), params
    )  # ensure timing accuracy
    print(f"\n\nLoading parameters took {time.time() - start_t}.")

    start_t = time.time()
    action = env_manager.inference_fn(params, env_state, timestep)
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready(), action
    )  # ensure timing accuracy
    print(f"\n\nInference took {time.time() - start_t}.")

    start_t = time.time()
    env_step = jax.jit(env_manager.env.step)
    print(f"\n\nStep jitting took {time.time() - start_t}.")

    # Apply it in the environment
    start_t = time.time()
    env_state = env_step(env_state, action)
    print(f"\n\nEnvironment-only step took {time.time() - start_t}.")

    start_t = time.time()
    for _ in range (5):
        env_state = env_step(env_state, action)
    print(f"\n\n5 steps took {time.time() - start_t}.")

    start_t = time.time()
    env_state = env_manager.step(
        cmd_lin_x=0.0,
        cmd_ang_z=0.1,
    )
    print(f"\n\nEnvironment-manager step with {env_manager.repetitions} repetitions took {time.time() - start_t}.")


    """

    # Perturbation initialisation
    perturbation_manager = PerturbationManager(
        wheel_base=WHEEL_BASE,
        wheel_radius=WHEEL_RADIUS,
        wheel_max_velocity=WHEEL_MAX_VELOCITY,
    )

    # FLAIR Initialisation 
    flair = FLAIR(
        adaptation_off=args.adaptation_off,
        map_elites_map=args.map_elites_map,
        grid_resolution=grid_resolution,
        min_command=min_command,
        max_command=max_command,
    )

    # Driver initialisation
    driver = Driver(path=path, number_laps=number_laps)
    min_driver_speed, max_driver_speed = driver.get_driver_configuration()

    # Save all the configurations
    metric_manager.save_gp_collection_configuration()
    metric_manager.save_gp_training_configuration(
        grid_resolution=grid_resolution,
        min_command=min_command,
        max_command=max_command,
    )
    metric_manager.save_driver_configuration(
        min_driver_speed=min_driver_speed,
        max_driver_speed=max_driver_speed,
        path=path,
        scaling=scaling,
        scaling_side=scaling_side,
        scaling_amplitude=scaling_amplitude,
        wind=wind,
        broken_leg=broken_leg,
        broken_leg_index=broken_leg_index,
    )

    print(f"Done initialising the pipeline, took: {time.time() - start_t}.")
    start_t = time.time()

    #############
    # Main Loop #

    # For each replication
    for rep in range (args.num_reps):

        print(f"\nPerforming rep {rep + 1} / {args.num_reps}.")
        print(f"Initialising and reseting elements.")
        start_rep_t = time.time()

        # Reset the environment
        env_state = env_manager.reset()

        # Reset the driver
        driver.reset()
        driver_done = False

        # Reset FLAIR
        buffer_initialised = False
        flair.reset()

        # Get the first sensor reading from the environment
        (
            state,
            quaternion,
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
        ) = env_manager.get_sensor()

        # Prepare the metrics for this rep
        metric_manager.create_metrics_rep(rep=rep)
        sim_start = datetime.now(tz=timezone.utc)
        timestep = 1
        timing = timestep_to_utc_timestamp(timestep, sim_start=sim_start, rate_hz=args.sensor_freq)

        # Save the first metrics for this rep
        metric_manager.add_main_metrics_rep(
            env_state=env_state,
            timing=timing, 
            timestep=timestep,
            rep=rep,
            damage_type=damage_type,
            scaling_value=scaling_value,
            section=section,
            lap=driver.lap,
            target_id=driver.target,
            target_tx=driver.target_tx,
            target_ty=driver.target_ty,
            tx=sensor_tx,
            ty=sensor_ty,
            human_cmd_lin_x=0.0,
            human_cmd_ang_z=0.0,
            vx=sensor_vx,
            wz=sensor_wz,
        )

        print(f"Done initialising and reseting, took: {time.time() - start_rep_t}.")
        print(f"\n  Driver starting.")

        try:

            # While not done with the path
            call_since_last_training = 0
            total_timesteps = 0
            while not driver_done:

                # debug_start_t = time.time()

                # First, train the model if necessary
                if call_since_last_training >= args.model_command_ratio and not args.adaptation_off:
                    print(f"    Debug - Training model.")
                    flair.train_model()
                    call_since_last_training = 0
                    # print(f"    Debug - Done training model, took: {time.time() - debug_start_t}.")
                else:
                    call_since_last_training += 1

                # Second, get the velocity command from the driver
                driver_done, human_cmd_lin_x, human_cmd_ang_z = driver.follow_path(
                    x_pos=sensor_tx,
                    y_pos=sensor_ty,
                    quaternion=quaternion,
                )
                #human_cmd_lin_x = 0.0
                #human_cmd_ang_z = -0.1

                # Third, get the corresponding adaptation
                (
                    adaptation_cmd_lin_x, 
                    adaptation_cmd_ang_z, 
                    gp_prediction_x, 
                    gp_prediction_y, 
                    human_cmd_lin_x, 
                    human_cmd_ang_z,
                ) = flair.get_command(
                    human_cmd_lin_x=human_cmd_lin_x, 
                    human_cmd_ang_z=human_cmd_ang_z,
                    state=state,
                )

                # Fourth, apply the perturbation
                (
                    perturbation_cmd_lin_x, 
                    perturbation_cmd_ang_z, 
                ) = perturbation_manager.apply_perturbation(
                    adaptation_cmd_lin_x, 
                    adaptation_cmd_ang_z, 
                    state,
                )

                # Fifth, save the adaptation metrics
                metric_manager.add_adaptation_metrics_rep(
                    p1=flair.p1,
                    p2=flair.p2,
                    a=flair.a,
                    b=flair.b,
                    c=flair.c,
                    d=flair.d,
                    offset=flair.offset,
                    human_cmd_lin_x=human_cmd_lin_x,
                    human_cmd_ang_z=human_cmd_ang_z,
                    adaptation_cmd_lin_x=adaptation_cmd_lin_x,
                    adaptation_cmd_ang_z=adaptation_cmd_ang_z,
                    perturbation_cmd_lin_x=perturbation_cmd_lin_x,
                    perturbation_cmd_ang_z=perturbation_cmd_ang_z,
                )

                # print(f"    Debug - Pre-env time: {time.time() - debug_start_t}.")
                # debug_start_t = time.time()

                # Sixth, apply it for command_sensor_ratio timesteps
                for sensor_timestep in range(args.command_sensor_ratio):

                    timestep += 1
                    timing = timestep_to_utc_timestamp(timestep, sim_start=sim_start, rate_hz=args.sensor_freq)

                    # Step the environment
                    env_state = env_manager.step(
                        cmd_lin_x=perturbation_cmd_lin_x,
                        cmd_ang_z=adaptation_cmd_ang_z,
                    )

                    # Get new sensor reading
                    (
                        state,
                        quaternion,
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
                    ) = env_manager.get_sensor()
                    if sensor_vx > 130 or sensor_vx < -130 or sensor_wz > 130 or sensor_wz < -130:
                        print("!!!ERROR!!! Robot exploded, exiting.")
                        driver_done = True
                        break
                    # Add to the buffers for FLAIR
                    sensor_time = ROSTimestamp(timestep, args.sensor_freq)
                    if not buffer_initialised:
                        buffer_state = np.array([state])
                        buffer_sensor_time = np.array([[sensor_time]])
                        buffer_sensor_vx = np.array([[sensor_vx]])
                        buffer_sensor_wx = np.array([[sensor_wx]])
                        buffer_sensor_wy = np.array([[sensor_wy]])
                        buffer_sensor_wz = np.array([[sensor_wz]])
                        buffer_initialised = True
                    else:
                        buffer_state =  np.append(buffer_state, [state], axis=0)
                        buffer_sensor_time = np.append(buffer_sensor_time, [[sensor_time]], axis=0)
                        buffer_sensor_vx = np.append(buffer_sensor_vx, [[sensor_vx]], axis=0)
                        buffer_sensor_wx = np.append(buffer_sensor_wx, [[sensor_wx]], axis=0)
                        buffer_sensor_wy = np.append(buffer_sensor_wy, [[sensor_wy]], axis=0)
                        buffer_sensor_wz = np.append(buffer_sensor_wz, [[sensor_wz]], axis=0)
                        buffer_initialised = True

                    # Save the metrics
                    metric_manager.add_main_metrics_rep(
                        env_state=env_state,
                        timing=timing,
                        timestep=timestep,
                        rep=rep,
                        damage_type=damage_type,
                        scaling_value=scaling_value,
                        section=section,
                        lap=driver.lap,
                        target_id=driver.target,
                        target_tx=driver.target_tx,
                        target_ty=driver.target_ty,
                        tx=sensor_tx,
                        ty=sensor_ty,
                        human_cmd_lin_x=human_cmd_lin_x,
                        human_cmd_ang_z=human_cmd_ang_z,
                        vx=sensor_vx,
                        wz=sensor_wz,
                    )

                # print(f"    Debug - Env steps time: {time.time() - debug_start_t}.")
                # debug_start_t = time.time()

                # Add to the adaptation dataset
                flair.add_datapoint(
                    state=buffer_state,
                    sensor_time=buffer_sensor_time,
                    sensor_vx=buffer_sensor_vx,
                    sensor_wx=buffer_sensor_wx,
                    sensor_wy=buffer_sensor_wy,
                    sensor_wz=buffer_sensor_wz,
                    adaptation_cmd_lin_x=adaptation_cmd_lin_x, 
                    adaptation_cmd_ang_z=adaptation_cmd_ang_z, 
                    gp_prediction_x=gp_prediction_x, 
                    gp_prediction_y=gp_prediction_y, 
                    human_cmd_lin_x=human_cmd_lin_x, 
                    human_cmd_ang_z=human_cmd_ang_z,
                )
                buffer_initialised = False

                # print(f"    Debug - Data Collection time: {time.time() - debug_start_t}.")

                if total_timesteps % 100 == 0:
                    print(f"\n    Debug - total_timesteps: {total_timesteps}.")
                    print(f"    Debug - Human: {human_cmd_lin_x}, {human_cmd_ang_z}.")
                    print(f"    Debug - Adapt: {adaptation_cmd_lin_x}, {adaptation_cmd_ang_z}.")
                    print(f"    Debug - Targets: {driver.target_tx}, {driver.target_ty}.")
                    print(f"    Debug - Sensors: {sensor_tx}, {sensor_ty}.")
                total_timesteps += 1


            print(f"Done with rep {rep + 1} / {args.num_reps}, took: {time.time() - start_rep_t}.")

        except Exception:
            print(f"Failed rep {rep + 1} / {args.num_reps}, took: {time.time() - start_rep_t}.")
            print("Still saving the metrics and html.")
            traceback.print_exc()

        # Save the metrics
        metric_manager.save_metrics_rep(env_manager.env.sys)

    print(f"\nDone with all replications and runs, took {time.time() - start_t}")

