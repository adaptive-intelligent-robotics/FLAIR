import sys
import os
import time
import csv
import argparse
from copy import deepcopy

import numpy as np
from typing import Tuple, Dict, Any

import jax
import jax.numpy as jnp
from brax.math import quat_to_euler
from scipy.spatial.transform import Rotation as R

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
from path_config import chicane_path, wind_path

# Import the environment
from utils.set_up_hexapod import set_up_hexapod

# Define new version of robot-specific parameters
# ROBOT_WIDTH = 
# WHEEL_BASE = 
# WHEEL_RADIUS = 
# WHEEL_MAX_VELOCITY = 

# Create a dummy Logger that just prints
class PrintLogger:

    def debug(msg):
        print(f"DEBUG: {msg}")

    def info(msg):
        print(f"INFO: {msg}")

    def warning(msg):
        print(f"WARNING: {msg}")

    def error(msg):
        print(f"ERROR: {msg}")

    def critical(msg):
        print(f"CRITICAL: {msg}")

##########
# Driver #

class Driver:
    """
    Automatic driver for the robot. Similar to the code
    of Vicon in the real-world robotics pipeline. 
    Most of the code is taken from there but removing the ROS 
    components. 
    """

    def __init__(self, path: np.array) -> None:

        self.path = path
        self.target = 0

        # Parameters
        self.error_threshold = 0.1
        self.min_driver_speed = 0.01
        self.max_driver_speed = 0.1

    def reset(self) -> None:
        self.target = 0

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

        # Transform quaternion
        r = R.from_quat(quaternion)

        # Compute the error
        error_x = (x - x_pos) / 1000
        error_y = (y - y_pos) / 1000
        error_robot_frame = r.apply(np.asarray([error_x, error_y, 0]), inverse=True)
        angle_heading = np.arctan2(error_robot_frame[1], error_robot_frame[0])
        distance = np.linalg.norm(error_robot_frame)

        # If already at target, go to next target
        if distance < self.error_threshold:

            # If done with the path, print it
            if self.target == (len(self.path)-1):
                return True, 0.0, 0.0

            # Else recursively call this function
            self.target += 1
            print(f"Next target: {self.target}, at: {self.path[self.target]}")
            return self.follow_path(
                x_pos=x_pos,
                y_pos=y_pos,
                quaternion=quaternion,
            )
        
        # Compute the new vx command
        if -np.pi / 4 > angle_heading or angle_heading > np.pi / 4:
            # If need reorientation, no vx
            v_lin = 0.0
        else:
            # Else vx proportional to distance
            v_lin = np.clip(0.5 * distance, self.min_driver_speed, self.max_driver_speed)

        # Compute the new wz command
        wz = np.clip(1.7 * angle_heading, -0.7, 0.7)

        return False, v_lin, -wz


#########
# FLAIR #

class FLAIR:
    """
    Merge of the GPDataset, GPTraining and Adaptation thread from 
    the real-world pipeline. Based on the same code from 
    src/flair/functionality_controller/, but in an asynchroneous version. 
    """

    def __init__(self, map_elites_map: str, grid_resolution: np.ndarray, min_command: float, max_command: float):

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
                grid_resolution=grid_resolution,
                min_command=min_command,
                max_command=max_command,
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
                grid_resolution=grid_resolution,
                min_command=min_command,
                max_command=max_command,
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
            grid_resolution=grid_resolution,
            min_command=min_command,
            max_command=max_command,
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

    def reset(self):

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
        state: np.ndarray,
        sensor_tx: float,
        sensor_ty: float,
        sensor_tz: float,
        sensor_vx: float,
        sensor_vy: float,
        sensor_vz: float,
        sensor_yaw: float,
        sensor_roll: float,
        sensor_pitch: float,
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
            self.datapoints = DataPoints.add(
                datapoint=self.datapoints,
                point_id=final_datapoint[0],
                sensor_time_sec=final_datapoint[1],
                sensor_time_nanosec=final_datapoint[2],
                command_time_sec=final_datapoint[3],
                command_time_nanosec=final_datapoint[4],
                state=final_datapoint[14:],
                gp_prediction_x=final_datapoint[5],
                gp_prediction_y=final_datapoint[6],
                command_x=final_datapoint[7],
                command_y=final_datapoint[8],
                intent_x=final_datapoint[9],
                intent_y=final_datapoint[10],
                sensor_x=final_datapoint[11],
                sensor_y=final_datapoint[12],
            )


    def get_command(
        self, 
        human_cmd_lin_x: float, 
        human_cmd_ang_z: float, 
        state: np.ndarray,
    ) -> Tuple[float, float, float, float, float, float]:

        auto_reset = False

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
            self.p1 = float(learned_params[0])
            self.p2 = float(learned_params[1])
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

    def __init__(self, map_elites_map: str):

        # Create a random key
        random_seed = int(time.time() * 1e6) % (2**32)
        self.random_key = jax.random.PRNGKey(random_seed)

        # Load the config of the considered replication
        with open(f"{args.map_elites_map}/config.csv", mode='r', newline='') as file:
            reader = csv.DictReader(file)
            self.env_config = [row for row in reader]
            if len(self.env_config) > 1:
                print(f"{len(self.env_config)} runs of MAP-Elites, keeping thr first one.")
            self.env_config = self.env_config[0]
        self.env_name = self.env_config["env_name"]
        print(f"Initialising the environment: {self.env_name}.")

        # Create the environment
        # Set episode_length None to not end the environment 
        self.random_key, subkey = jax.random.split(self.random_key)
        (
            self.env,
            _,
            init_policies_fn,
            self.policy_structure,
            min_bd,
            max_bd,
            _,
            subkey,
        ) = set_up_hexapod(
            env_name=self.env_name,
            episode_length=None,
            batch_size=1,
            random_key=subkey,
        )

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
        print(f"    Using grid with resolution: {self.grid_resolution} and min/max: {self.min_command}, {self.max_command}.")

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
        self.inference_fn = jax.jit(self.policy_structure.apply)

    def get_grid_details(self) -> Tuple[np.ndarray, float, float]:
        return self.grid_resolution, self.min_command, self.max_command

    def reset(self):
        self.random_key, subkey = jax.random.split(self.random_key)
        self.env_state = self.env.reset(subkey)

    def step(self, cmd_lin_x: float, cmd_ang_z: float) -> None:

        # Get the corresponding controller from the map
        if self.cmd_lin_x == None or self.cmd_ang_z == None or cmd_lin_x != self.cmd_lin_x or cmd_ang_z != self.cmd_ang_z:
            index = get_cells_indices(jnp.asarray([cmd_lin_x, cmd_ang_z]), self.repertoire.centroids)
            self.params = jax.tree_util.tree_map(
                lambda x: x[indices].squeeze(), self.repertoire.genotypes
            )
            self.cmd_lin_x = cmd_lin_x
            self.cmd_ang_z = cmd_ang_z
            self.timestep = 0

        # Get the action
        action = self.inference_fn(self.params, self.env_state.obs, self.timestep)
        self.timestep += 1

        # Apply it in the environment
        self.env_state = self.env.step(self.env_state, action)

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
    ):

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

    def __init__(self, results: str, circuit: str, save_html: bool):
        self.folder = results
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)
        self.circuit = circuit
        self.save_html = save_html
        self.empty_metrics()

    def empty_metrics(self):

        # Main comparison dataframe
        self.main_metrics = {
            "Timesteps": [],
            "Reps": [],
            "Sections_start_time": [],
            "Sections_end_time": [],
            "Sections": [],
            "Sections_index": [],
            "Laps": [],
            "Laps_start_time": [],
            "Laps_end_time": [],
            "tx": [],
            "ty": [],
            "index": [],
            "target_tx": [],
            "target_ty": [],
            "Time": [],
            "Damage_Type": [],
            "Scaling_Value": [],
            "human_cmd_lin_x": [],
            "human_cmd_ang_z": [],
            "vx": [],
            "wz": [],
        }

        # Same content as /gp_damage_introspection in influx
        self.gp_damage_introspection_metrics = {
            "p1": [],
            "p2": [],
        }

        # Same content as /learnt_state_functions in influx
        self.learnt_state_functions_metrics = {
            "a": [],
            "b": [],
            "c": [],
            "d": [],
            "offset": [],
        }

        # Same content as /adaptation in influx 
        # (missing some time-related field as this is asynchroneous)
        self.adaptation_metrics = {
            "human_cmd_lin_x": [],
            "human_cmd_ang_z": [],
            "adaptation_cmd_lin_x": [],
            "adaptation_cmd_ang_z": [],
            "adaptation_cmd_flipper": [],
            "perturbation_cmd_lin_x": [],
            "perturbation_cmd_ang_z": [],
            "perturbation_cmd_flipper": [],
        }

    def _create_metrics(self, file_name: str, metrics: Dict, name: str) -> None:
        with open(file_name, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(metrics.keys())
        print(f"  Saving {name} metrics in {file_name}.")

    def create_metrics_rep(self, rep: int):

        self.empty_metrics()

        # Create a csv to save data for this rep
        self.rep_main_file_name = f"{self.folder}/{self.circuit}_replication_main_{rep}.csv"
        self._create_metrics(
            file_name=self.rep_main_file_name,
            metrics=self.main_metrics,
            name="main",
        )

        self.rep_gp_damage_introspection_file_name = f"{self.folder}/{self.circuit}_replication_gp_damage_introspection_{rep}.csv"
        self._create_metrics(
            file_name=self.rep_gp_damage_introspection_file_name,
            metrics=self.gp_damage_introspection_metrics,
            name="gp_damage_introspection",
        )

        self.rep_learnt_state_functions_file_name = f"{self.folder}/{self.circuit}_replication_learnt_state_functions_{rep}.csv"
        self._create_metrics(
            file_name=self.rep_learnt_state_functions_file_name,
            metrics=self.learnt_state_functions_metrics,
            name="learnt_state_functions",
        )

        self.rep_adaptation_file_name = f"{self.folder}/{self.circuit}_replication_adaptation_{rep}.csv"
        self._create_metrics(
            file_name=self.rep_adaptation_file_name,
            metrics=self.adaptation_metrics,
            name="adaptation",
        )

        # If saving html, create a folder to save data
        if self.save_html:
            self.rep_html_file_name = f"{self.folder}/{self.circuit}_replication_{rep}.html"
            self.rollout = []
            print(f"  Saving html in {self.rep_html_file_name}.")
    
    def _save_metrics(self, file_name: str, metrics: Dict) -> None:
        with open(file_name, mode="a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(metrics.values())

    def save_metrics_rep(self, env_state: Any, ):

        # If saving html
        if self.save_html:
            self.rollout.append(env_state)

    def save_html(self, env_sys: Any):
        if self.save_html:
            html_file = html.render(env_sys, [s.pipeline_state for s in self.rollout])
            f = open(self.rep_html_file_name, "w")
            f.write(html_file)
            f.close()


    def _save_configuration(self, file_name: str, config: Dict) -> None:
        print(f"Saving configuration in {file_name}.")
        with open(file_name, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(config.keys())
            writer.writerow(config.values())

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
        adaptation_on: bool, 
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
            "small_loop": str(wind_path),
            "chicane": str(chicane_path),
            "scaling": str(scaling),
            "scaling_sides": str(scaling_side),
            "scaling_amplitudes": str(scaling_amplitude),
            "broken_leg": str(broken_leg),
            "broken_leg_index": str(broken_leg_index),
            "wind": str(wind),
            "adaptation_on": adaptation_on,
            "min_speed": min_driver_speed,
            "max_speed": max_driver_speed,
        }
        self._save_configuration(
            file_name=f"{self.folder}/vicon_configuration.csv",
            config=config,
        )

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

    # Command / sensor ratio - set to the same value as real robot
    parser.add_argument("--command-sensor-ratio", default=7, type=float)

    # Circuit: chicane static, chicane dynamic or wind
    parser.add_argument("--circuit", default="chicane_static", type=str)

    # Number of reps
    parser.add_argument("--num-reps", default=1, type=int)

    # Save video
    parser.add_argument("--save-html", action="store_true")

    args = parser.parse_args()

    ##################
    # Initialisation #

    # Metric Manager initialisation
    metric_manager = MetricManager(results=args.results, circuit=args.circuit, save_html=args.save_html)

    # Circuit initialisation 
    if args.circuit == "chicane_static":
        path = chicane_path
    elif args.circuit == "chicane_dynamic":
        path = chicane_path
    elif args.circuit == "chicane_broken_leg":
        path = chicane_path
    elif args.circuit == "wind":
        path = wind_path
    else:
        assert 0, "!!!ERROR!!! Undefined circuit."

    # Perturbation initialisation
    scaling = [False for way_point in range (len(path))]
    scaling_side = ["right" for way_point in range (len(path))]
    broken_leg = [False for way_point in range (len(path))]
    broken_leg_index = [4 for way_point in range (len(path))]
    scaling_amplitude = [0.7 for way_point in range (len(path))]
    wind = [False for way_point in range (len(path))]
    if args.circuit == "chicane_static":
        scaling = [True for way_point in range (len(path))]
    elif args.circuit == "chicane_dynamic":
        scaling = [True for way_point in range (len(path))]
        scaling_amplitude = [0.7] * 5 + [0.4] * 12
    elif args.circuit == "chicane_broken_leg":
        broken_leg = [True for way_point in range (len(path))]
        broken_leg_index = [4 for way_point in range (len(path))]
    elif args.circuit == "wind":
        wind = [True for way_point in range (len(path))]
    else:
        assert 0, "!!!ERROR!!! Undefined circuit."

    # Environment initialisation 
    env_manager = EnvironmentManager(map_elites_map=args.map_elites_map)
    grid_resolution, min_command, max_command = env_manager.get_grid_details()

    # Perturbation initialisation
    perturb_manager = PerturbationManager(
        wheel_base=WHEEL_BASE,
        wheel_radius=WHEEL_RADIUS,
        wheel_max_velocity=WHEEL_MAX_VELOCITY,
    )

    # FLAIR Initialisation 
    flair = FLAIR(
        map_elites_map=args.map_elites_map,
        grid_resolution=grid_resolution,
        min_command=min_command,
        max_command=max_command,
    )

    # Driver initialisation
    driver = Driver(path=path)
    min_driver_speed, max_driver_speed = driver.get_driver_configuration()

    # Save all the configurations
    metric_manager.save_gp_collection_configuration()
    metric_manager.save_gp_training_configuration(
        grid_resolution=grid_resolution,
        min_command=min_command,
        max_command=max_command,
    )
    metric_manager.save_driver_configuration(
        adaptation_on=not(args.adaptation_off),
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

    #############
    # Main Loop #

    # For each replication
    for rep in range (args.num_reps):
        print(f"Performing rep {rep} / {args.num_reps}.")

        # Prepare the metrics for this rep
        metric_manager.create_metrics_rep(rep=rep)

        # Reset the environment
        env_manager.reset()

        # Reset the driver
        driver.reset()
        driver_done = False

        # While not done with the path
        while not driver_done:

            # First, get the sensor reading from the environment
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

            # Second, get the velocity command from the driver
            driver_done, human_cmd_lin_x, human_cmd_ang_z = driver.follow_path(
                x_pos=sensor_tx,
                y_pos=sensor_ty,
                quaternion=quaternion,
            )

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

            # Fifth, apply it for command_sensor_ratio timesteps
            for timestep in range(args.command_sensor_ratio):

                # Step the environment
                env_manager.step(
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

                # Add to the adaptation dataset
                flair.add_datapoint(
                    state=state,
                    sensor_tx=sensor_tx,
                    sensor_ty=sensor_ty,
                    sensor_tz=sensor_tz,
                    sensor_vx=sensor_vx,
                    sensor_vy=sensor_vy,
                    sensor_vz=sensor_vz,
                    sensor_yaw=sensor_yaw,
                    sensor_roll=sensor_roll,
                    sensor_pitch=sensor_pitch,
                    sensor_wx=sensor_wx,
                    sensor_wy=sensor_wy,
                    sensor_wz=sensor_wz,
                    adaptation_cmd_lin_x=adaptation_cmd_lin_x, 
                    adaptation_cmd_ang_z=adaptation_cmd_ang_z, 
                    gp_prediction_x=gp_prediction_x, 
                    gp_prediction_y=gp_prediction_y, 
                    human_cmd_lin_x=human_cmd_lin_x, 
                    human_cmd_ang_z=human_cmd_ang_z,
                )

                # Save the metrics
                metric_manager.save_metrics_rep(self.env_state)

        # Save the html
        metric_manager.save_html(env_manager.env.sys)

