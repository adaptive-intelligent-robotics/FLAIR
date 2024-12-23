# Written or Adapted by the Imperial College London Team for the FLAIR project, 2023
# Authors for this file:
# Maxime Allard
# Manon Flageat
# Bryan Lim
# Antoine Cully

from typing import Any, Dict, Tuple

from jax.config import config

config.update("jax_enable_x64", True)
import copy
import time
from scipy.optimize import lsq_linear

import gpjax as gpx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from gpjax.base import meta_leaves, meta_map
from gpjax.dataset import Dataset
from gpjax.fit import fit
from gpjax.kernels import RBF

from functionality_controller.datapoint import DataPoints
from functionality_controller.dataset_fifo import StateFIFODataset, FIFODataset
from functionality_controller.dataset_grid import MultiDimsStateInputGridFIFODataset, OUT_OF_BOUND
from functionality_controller.gp.gpjax_map_v2 import MAPMean, MAPPrior

# from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression


class AdaptiveGP:
    """GP class used by STATE_FAST version of the GP."""

    def __init__(
        self,
        logger: Any,
        jiting_datapoints: DataPoints,
        grid_resolution: int,
        min_command: int,
        max_command: int,
        robot_width: float,
        default_obs_noise: np.ndarray,
        default_lengthscale: np.ndarray,
        default_variance: np.ndarray,
        min_diff_datapoint: int,
        use_grid_dataset: bool,
        dataset_size: int,
        dataset_grid_cell_size: int,
        dataset_grid_neighbours: float,
        dataset_grid_novelty_threshold: float,
        datapoint_batch_size: int,
        max_p_value: float,
        multi_function: bool,
        remove_offset: bool,
        state_dim: int,
        state_min_dataset: float,
        state_max_dataset: float,
        state_min_opt_clip: float,
        state_max_opt_clip: float,
        p1_min: float,
        p1_max: float,
        p2_min: float,
        p2_max: float,
        p3_min: float,
        p3_max: float,
        minibatch_size: int,
        auto_reset_error_buffer_size: int,
        auto_reset_angular_rot_weight: float,
        auto_reset_threshold: float,
    ) -> None:
        """
        Args:
            logger
            jiting_datapoints: empty DataPoints uses to trigger jitting of the Dataset.
            grid_resolution: resolution of the command grid, used by all processes across FC.
            min_command: min value for the command grid, used by all processes across FC.
            max_command: max value for the command grid, used by all processes across FC.
            robot_width: property of the robot.
            default_obs_noise: property of the robot.
            default_lengthscale: property of the robot.
            default_variance: property of the robot.
            min_diff_datapoint: minimum number of datapoint diff to train.
            use_grid_dataset: if False, use Grid dataset, if True, use FIFO dataset.
            dataset_size: temporal FIFO size.
            dataset_grid_cell_size: Grid dataset parameter.
            dataset_grid_neighbours: Grid dataset parameter.
            dataset_grid_novelty_threshold: Grid dataset parameter.
            datapoint_batch_size: size of datapoints, used bu dataset.
            max_p_value: maximum prior value.
            multi_function: if False, use a single function prior.
            remove_offset: if True, set the offset to 0 all the time.
            state_dim: which dimension to use for the state.
            state_min_dataset: min of state-dimension in the dataset.
            state_max_dataset: max of state-dimension in the dataset.
            state_min_opt_clip: min to clip the state before prior computation.
            state_max_opt_clip: max to clip the state before prior computation.
            p_min_max: min and max for each coefficient of the prior.
            minibatch_size: size of the minibatch used for auto-reset.
            auto_reset_error_buffer_size: size of tge error buffer used for auto-reset.
            auto_reset_angular_rot_weight: weight of the angular-command in the auto-reset formula.
            auto_reset_threshold: threshold to trigger auto-reset.
        """

        # Store attributes
        self._logger = logger
        self.robot_width = robot_width
        self.default_obs_noise = jnp.asarray(default_obs_noise)
        self.default_lengthscale = jnp.asarray(default_lengthscale)
        self.default_variance = jnp.asarray(default_variance)
        self.min_diff_datapoint = min_diff_datapoint
        self.max_p_value = max_p_value
        self.multi_function = multi_function
        self.remove_offset = remove_offset
        self.state_dim = state_dim
        self.state_min_opt_clip = state_min_opt_clip
        self.state_max_opt_clip = state_max_opt_clip
        self.p1_min = p1_min
        self.p1_max = p1_max
        self.p2_min = p2_min
        self.p2_max = p2_max
        self.p3_min = p3_min
        self.p3_max = p3_max
        self.minibatch_size = minibatch_size
        self.auto_reset_error_buffer_size = auto_reset_error_buffer_size
        self.auto_reset_angular_rot_weight = auto_reset_angular_rot_weight
        self.auto_reset_threshold = auto_reset_threshold
        # Init P parameter
        self.p_raw = np.zeros(shape=(5,))

        # Create the grid
        x_axis = jnp.linspace(min_command, max_command, num=grid_resolution)
        self.all_descriptors = jnp.asarray(jnp.meshgrid(x_axis, x_axis)).T.reshape(
            -1, 2
        )

        # Set random key
        self.random_key = jax.random.PRNGKey(0)

        # Create the jited dataset handling addition
        if use_grid_dataset:
            # Grid dataset
            self.dataset = MultiDimsStateInputGridFIFODataset.create(
                size=dataset_size,
                min_command=min_command,
                max_command=max_command,
                cell_depth=dataset_grid_cell_size,
                input_bins=grid_resolution,
                prop_neighbours=dataset_grid_neighbours,
                novelty_threshold=dataset_grid_novelty_threshold,
                state_dimensions=jnp.asarray([self.state_dim]),
                state_dimensions_min=jnp.asarray([state_min_dataset]),
                state_dimensions_max=jnp.asarray([state_max_dataset]),
                datapoint_batch_size=datapoint_batch_size,
            )
            self.dataset_add_fn = MultiDimsStateInputGridFIFODataset.add
            self.dataset_reset_fn = MultiDimsStateInputGridFIFODataset.reset

        else:
            # FIFO dataset
            self.dataset = StateFIFODataset.create(dataset_size)
            self.dataset_add_fn = StateFIFODataset.add
            self.dataset_reset_fn = StateFIFODataset.reset

        # Create minibatch for auto-reset detection
        self.minibatch_dataset = FIFODataset.create(self.minibatch_size)
        self.minibatch_dataset_add_fn = FIFODataset.add
        self.minibatch_dataset_reset_fn = FIFODataset.reset

        # Init dataset parameters
        self.N = dataset_size
        self.total_position = 0
        self.minibatch_N = self.minibatch_size
        self.minibatch_total_position = 0
        self.latest_train_total_position = 0

        # Init auto-reset parameters
        self.mean_intent_error_buffer = []
        self.last_reset_t = time.time()

        # Default empty parameters
        self.default_rotation = jnp.asarray([1., 1., 0., 0., 0., 0., 0.])
        self.default_offset = jnp.zeros(shape=(500, 2))

        # Init GP objects
        self.all_corrected_descriptors = self.all_descriptors
        self.uncertainties = jnp.zeros(shape=(self.all_descriptors.shape[0], 1))
        self.kernel = RBF(active_dims=[_ for _ in range(3)])
        self.kernel = self.kernel.replace(
            lengthscale=self.default_lengthscale,
            variance=self.default_variance,
        )
        self.kernel_x = np.zeros(shape=(1, 1))
        self.x_opt_posterior = None
        self.xy_learned_params = {"mean_function": {}}

        # self.log_model = LogisticRegression(random_state=0)
        # Call functions to trigger jiting
        self.update(jiting_datapoints)
        _, _ = self._train_model()

        self.dataset = self.dataset_reset_fn(self.dataset)
        



    def _to_cpu(self, array: jnp.ndarray):
        """Get a dataset array, tranfer it to numpy and remove OUT_OF_BOUND."""
        cpu_array = np.asarray(array)
        cpu_array = cpu_array[~((cpu_array == OUT_OF_BOUND).any(axis=1)), ...]
        return cpu_array

    def reset(self) -> None:
        """Reset the controller adaptation."""

        self._logger.debug(f"RESETING GP" * 20)

        # Reset dataset
        self.dataset = self.dataset_reset_fn(self.dataset)

        # Reset minibatch
        self.minibatch_dataset = self.minibatch_dataset_reset_fn(
            self.minibatch_dataset
        )

        # Reset counters
        self.total_position = 0
        self.minibatch_total_position = 0
        self.latest_train_total_position = 0
        self.mean_intent_error_buffer = []
        self.last_reset_t = time.time()

        # Reset GP
        self.all_corrected_descriptors = self.all_descriptors
        self.uncertainties = jnp.zeros(shape=(self.all_descriptors.shape[0], 1))
        self.kernel_x = np.zeros(shape=(1, 1))

        # Reset xy_learned_params
        self.xy_learned_params["mean_function"]["rotation"] = self.default_rotation
        self.xy_learned_params["mean_function"][
            "offset"
        ] = self.x_opt_posterior.prior.mean_function.offset
        self.xy_learned_params["mean_function"][
            "obs_noise"
        ] = self.x_opt_posterior.likelihood.obs_noise
        self.p_raw = np.zeros(shape=(5,))

    def gp_update(self) -> Tuple[bool, float, float]:
        """Update the GP model based on its dataset."""

        # Train the GP
        if (
            self.total_position
            > self.latest_train_total_position + self.min_diff_datapoint
        ):
            # self._logger.debug("Start Training")
            opt_time, gp_fit_time = self._train_model()
            self.latest_train_total_position = self.total_position
            # self._logger.debug("Done Training")
            return True, opt_time, gp_fit_time

        return False, 0.0, 0.0

    def update(self, datapoint: DataPoints) -> None:
        """Update the GP dataset, adding the latest datapoints."""

        # Update the dataset
        self.dataset = self.dataset_add_fn(
            dataset=self.dataset, datapoint=datapoint
        )

        # Update N attribute for later calls
        self.total_position = self.total_position + jnp.sum(
            datapoint.array_point_id > -1
        )

        # Update the minibatch dataset
        self.minibatch_dataset = self.minibatch_dataset_add_fn(
            dataset=self.minibatch_dataset, datapoint=datapoint
        )

        # Update N attribute for later calls
        self.minibatch_total_position = self.minibatch_total_position + jnp.sum(
            datapoint.array_point_id > -1
        )

        # self._logger.debug("Ending update")


    def auto_reset(self) -> Tuple[bool, float]:
        """Test the reset condition, reset the dataset if necessary.

        Return:
            auto_reset: True if auto-reset needed
            error: error value to send to Grafana
        """

        # If it has been less than i seconds since last reset, do nothing
        #if time.time() - self.last_reset_t < 5.0:
        #    return False, 0.0, 0.0

        # If there is not enough data in the minibatch, do nothing
        if self.minibatch_total_position < self.minibatch_size:
            return False, 0.0, 0.0

        # Extract command sent to robot from dataset
        cmd_x = self._to_cpu(self.minibatch_dataset.command_x)
        cmd_y = self._to_cpu(self.minibatch_dataset.command_y)

        # Extract genotype predicted by GP from dataset
        gp_pred_x = self._to_cpu(self.minibatch_dataset.gp_prediction_x)
        gp_pred_y = self._to_cpu(self.minibatch_dataset.gp_prediction_y)

        # Extract user intent from dataset
        intent_x = self._to_cpu(self.minibatch_dataset.intent_x)
        intent_y = self._to_cpu(self.minibatch_dataset.intent_y)

        # Extract sensor reading from dataset
        sensor_x = self._to_cpu(self.minibatch_dataset.sensor_x)
        sensor_y = self._to_cpu(self.minibatch_dataset.sensor_y)

        # Compute all the corresponding errors
        intent_gp_error_x = np.abs(intent_x - gp_pred_x)
        intent_gp_error_y = np.abs(intent_y - gp_pred_y) * self.auto_reset_angular_rot_weight

        cmd_sensor_error_x = np.abs(cmd_x - sensor_x)
        cmd_sensor_error_y = np.abs(cmd_y - sensor_y) * self.auto_reset_angular_rot_weight

        error_distance_x = np.abs(intent_gp_error_x - cmd_sensor_error_x)
        error_distance_y = np.abs(intent_gp_error_y - cmd_sensor_error_y)

        # Old Error - Compute all the corresponding errors
        # intent_gp_error_x = np.abs(intent_x - sensor_x)
        # intent_gp_error_y = np.abs(intent_y - sensor_y) * self.auto_reset_angular_rot_weight
        # error_distance_x = np.abs(intent_gp_error_x)
        # error_distance_y = np.abs(intent_gp_error_y)

        # Mean the error across the datapoint in the minibatch
        mean_error_distance_x = np.mean(error_distance_x)
        mean_error_distance_y = np.mean(error_distance_y)

        # Average the errors to collapse two dimensions in one
        final_error_distance_norm = np.sqrt(
            mean_error_distance_x**2 + mean_error_distance_y**2
        )
        mean_intent_error = np.mean(final_error_distance_norm)

        # Just for debugging, compute the mean_error_distance_x one
        mean_intent_error_x = np.mean(np.abs(mean_error_distance_x))

        # If the error is bigger than threshold, reset
        if mean_intent_error > self.auto_reset_threshold:
            self.reset()
            self.last_reset_t = time.time()
            return True, mean_intent_error_x, mean_intent_error

        # PREVIOUS VERSION

        ## Store the current error in the error buffer
        #self.mean_intent_error_buffer.append(mean_intent_error)

        ## If the buffer is not filled, do not auto-reset
        #if len(self.mean_intent_error_buffer) < self.auto_reset_error_buffer_size:
        #    return False, 0.0, mean_intent_error

        ## Else, just select the self.auto_reset_error_buffer_size latest errors
        #self.mean_intent_error_buffer = self.mean_intent_error_buffer[
        #    -self.auto_reset_error_buffer_size:
        #]

        ## If there is an increase in the error in the buffer, reset
        #amplitude_increase = max(
        #    np.max(np.diff(self.mean_intent_error_buffer, n=1)),
        #    np.max(np.diff(self.mean_intent_error_buffer, n=2)),
        #)
        #if amplitude_increase > self.auto_reset_threshold:
        #    # Reset the GP, dataset and minibatch
        #    self.reset()
        #    return True, amplitude_increase, mean_intent_error

        return False, mean_intent_error_x, mean_intent_error


    def _train_model(self) -> Tuple[int, int]:
        """Train the model."""

        # Create a new GP
        # TODO: needed ? could be done in __init__
        self.kernel = self.kernel.replace(
            lengthscale=self.default_lengthscale,
            variance=self.default_variance,
        )
        mean_fn_x = MAPMean(chosen_bd=0).replace(
            rotation=self.default_rotation,
            offset=self.default_offset,
        )

        if self.x_opt_posterior is None:

            # Prior, likelihood and posterior for X
            x_prior = MAPPrior(kernel=self.kernel, mean_function=mean_fn_x)
            x_likelihood = gpx.Gaussian(num_datapoints=self.N).replace(
                obs_noise=self.default_obs_noise,
            )
            x_posterior = x_prior * x_likelihood
            self.x_opt_posterior = x_posterior

            # Create learned_params
            self.xy_learned_params["mean_function"]["rotation"] = self.default_rotation
            self.xy_learned_params["mean_function"][
                "offset"
            ] = self.x_opt_posterior.prior.mean_function.offset
            self.xy_learned_params["mean_function"][
                "obs_noise"
            ] = self.x_opt_posterior.likelihood.obs_noise

        # TODO: NO_STATE version has the spread condition here, should we have it too?

        ####### TIMED #########

        # Call optimisation
        # self._logger.debug(f"Entering p optimisation")
        start_t = time.time()
        if self.multi_function:

            # Multi function computation
            p_state, _ = AdaptiveGP.multi_optimisation(
                state_dim=self.state_dim,
                state_min_opt_clip=self.state_min_opt_clip,
                state_max_opt_clip=self.state_max_opt_clip,
                robot_width=self.robot_width,
                filtered_command_x=self._to_cpu(self.dataset.command_x),
                filtered_command_y=self._to_cpu(self.dataset.command_y),
                filtered_sensor_x=self._to_cpu(self.dataset.sensor_x),
                filtered_sensor_y=self._to_cpu(self.dataset.sensor_y),
                filtered_states=self._to_cpu(self.dataset.state),
                remove_offset = self.remove_offset,
                p0_min=-self.max_p_value,
                p0_max=self.max_p_value,
                p1_min=self.p1_min,
                p1_max=self.p1_max,
                p2_min=self.p2_min,
                p2_max=self.p2_max,
                p3_min=self.p3_min,
                p3_max=self.p3_max,
            )

            # Offset
            if self.remove_offset:
                p_state[-1] = 0

            # Final value
            p = 0
            p_state[0] = np.clip(p_state[0], -self.max_p_value, self.max_p_value)
            p_state[1] = np.clip(p_state[1], self.p1_min, self.p1_max)
            p_state[2] = np.clip(p_state[2], self.p2_min, self.p2_max)
            p_state[3] = np.clip(p_state[3], self.p3_min, self.p3_max)

        else:
            # Single function computation
            p_state, _, offset = AdaptiveGP.single_optimisation(
                state_dim=self.state_dim,
                state_min_opt_clip=self.state_min_opt_clip,
                state_max_opt_clip=self.state_max_opt_clip,
                robot_width=self.robot_width,
                filtered_command_x=self._to_cpu(self.dataset.command_x),
                filtered_command_y=self._to_cpu(self.dataset.command_y),
                filtered_sensor_x=self._to_cpu(self.dataset.sensor_x),
                filtered_sensor_y=self._to_cpu(self.dataset.sensor_y),
                filtered_states=self._to_cpu(self.dataset.state),
            )

            # Offset
            if self.remove_offset:
                offset = 0

            # Final value
            p_state = np.asarray([0, p_state, 0, 0, offset])
            p = p_state[0]


        ## Soft Update
        previous_p = self.p_raw
        p_mask = (p_state != 0)
        p_state = previous_p - np.clip(previous_p - p_state, a_min=-0.3, a_max=0.3)
        p_state = np.where(p_mask, p_state, 0)
        self.p_raw = p_state


        self._logger.debug(f"New introspection:: {p_state}")
        optimization_time = time.time() - start_t
        # self._logger.debug(f"Done p optimisation")

        ######### end of TIMED ##########

        # Enable Training
        mean_fn_x = MAPMean(chosen_bd=0).replace(
            rotation=jnp.asarray([p]), offset=p_state
        )
        x_prior = MAPPrior(kernel=self.kernel, mean_function=mean_fn_x)
        self.x_opt_posterior = self.x_opt_posterior.replace(prior=x_prior)
        self.xy_learned_params["mean_function"]["rotation"] = jnp.concatenate(
            [
                jnp.asarray(
                    [
                        jnp.min(jnp.asarray([1, 1 - p_state[-1]])),
                        jnp.min(jnp.asarray([1, 1 + p_state[-1]])),
                    ]
                ),
                p_state,
            ],
            axis=-1,
        )

        ####### TIMED #########

        # Call fit
        # self._logger.debug(f"Entering fit")
        start_t = time.time()
        (
            self.all_corrected_descriptors,
            self.uncertainties,
            self.xy_learned_params["mean_function"]["offset"],
            self.kernel_x,
            self.random_key,
        ) = AdaptiveGP._fit(
            x_posterior=self.x_opt_posterior,
            all_descriptors=self.all_descriptors,
            dataset=self.dataset,
            state_dim=self.state_dim,
            random_key=self.random_key,
        )
        gp_fit_time = time.time() - start_t
        # self._logger.debug(f"Done fit")

        ######### end of TIMED ##########

        return optimization_time, gp_fit_time

    @staticmethod
    @jax.jit
    def _fit(
        x_posterior: Any,
        all_descriptors: jnp.ndarray,
        dataset: StateFIFODataset,
        state_dim: int,
        random_key: jax.random.KeyArray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jax.random.KeyArray, float, float]:
        """Fit the GP to the current dataset."""

        @jax.jit
        def get_all_corrected_descriptors(
            x_posterior: Any,
            test_inputs: jnp.ndarray,
            training_data: Dataset,
            weights_noise: jnp.ndarray,
            state_train: jnp.ndarray,
            state_test: jnp.ndarray,
        ) -> jnp.ndarray:
            """Get the descriptor with the additional error computed by the GP model."""

            # Compute predictive mean and std
            predictive_mean, predictive_std, cov_term, kernel_x = jax.jit(
                x_posterior.predict_all_state
            )(
                test_inputs,
                train_data=training_data,
                weights_noise=weights_noise,
                state_train=state_train,
                state_test=state_test,
            )

            x_predictive_mean, x_predictive_std = predictive_mean[:, 0], predictive_std
            y_predictive_mean, y_predictive_std = predictive_mean[:, 1], predictive_std

            # Concatenate x and y predictive_mean
            predictive_mean = jnp.concatenate(
                [
                    jnp.expand_dims(x_predictive_mean, axis=1),
                    jnp.expand_dims(y_predictive_mean, axis=1),
                ],
                axis=1,
            )

            # Compute uncertainties
            uncertainties = jnp.expand_dims(x_predictive_std, axis=1) + jnp.expand_dims(
                y_predictive_std, axis=1
            )

            return predictive_mean, uncertainties, cov_term, kernel_x

        # Get dataset
        observations_dataset = Dataset(
            X=jnp.concatenate([dataset.command_x, dataset.command_y], axis=-1),
            y=jnp.concatenate([dataset.sensor_x, dataset.sensor_y], axis=-1),
        )

        weights_noise = jnp.ones((dataset.command_x.shape[0]))
        # Apply weight for accumulated datapoints
        #weights_noise = jnp.squeeze(1 / dataset.num_accumulate)
        #if len(dataset.num_accumulate) == 1:
        #    weights_noise = jnp.expand_dims(weights_noise, axis=0)

        # Get state
        state_train = dataset.state[:, [state_dim]]
        state_test = dataset.state[:, [state_dim]].at[[-1], ...].get()
        state_test = jnp.broadcast_to(
            state_test, (all_descriptors.shape[0],) + state_test.shape[1:]
        )

        # Compute corrected bd
        start = time.time()
        (
            all_corrected_descriptors,
            uncertainties,
            offset,
            kernel_x,
        ) = get_all_corrected_descriptors(
            x_posterior,
            test_inputs=all_descriptors,
            training_data=observations_dataset,
            weights_noise=weights_noise,
            state_train=state_train,
            state_test=state_test,
        )

        return (
            all_corrected_descriptors,
            uncertainties,
            offset,
            kernel_x,
            random_key,
        )

    @staticmethod
    def single_optimisation(
        state_dim: int,
        state_min_opt_clip: float,
        state_max_opt_clip: float,
        robot_width: float,
        filtered_command_x: np.ndarray,
        filtered_command_y: np.ndarray,
        filtered_sensor_x: np.ndarray,
        filtered_sensor_y: np.ndarray,
        filtered_states: np.ndarray,
    ) -> Tuple:
        """Find the best prior in single-function case."""

        # Concatenate to have the good shape
        dataset_x = np.concatenate([filtered_command_x, filtered_command_y], axis=-1)
        dataset_y = np.concatenate([filtered_sensor_x, filtered_sensor_y], axis=-1)

        # Get the state
        selected_state = filtered_states[:, [state_dim]]
        signs = np.sign(selected_state)
        selected_state = np.repeat(selected_state.reshape(-1, 1), 2, axis=0)
        selected_state = np.clip(
            selected_state, a_min=state_min_opt_clip, a_max=state_max_opt_clip
        )

        # Get the matrices for the least square
        A = (dataset_y - dataset_x).reshape(-1, 1)

        offset = np.concatenate(
            [np.ones(shape=signs.shape), np.zeros(shape=signs.shape)], axis=1
        ).reshape(-1, 1)
        a = np.concatenate(
            [-signs * 0.5, np.ones(shape=signs.shape) * robot_width / 4], axis=1
        )
        b = np.concatenate(
            [np.ones(shape=signs.shape) * 1 / robot_width, -signs * 0.5], axis=1
        )
        B = np.stack(
            [
                np.multiply(dataset_x, a).sum(axis=1),
                np.multiply(dataset_x, b).sum(axis=1),
            ],
            axis=-1,
        ).reshape(-1, 1)
        B = np.multiply(B, selected_state)
        B = np.concatenate([B, offset], axis=1)

        # Apply the least square
        p_choice, rest, _, _ = np.linalg.lstsq(B, A)

        # Return result
        p = p_choice.squeeze()[0]
        p = np.clip(p, -1.0, 1.0)
        offset = p_choice.squeeze()[1]

        return p, rest, offset

    @staticmethod
    def multi_optimisation(
        state_dim: int,
        state_min_opt_clip: float,
        state_max_opt_clip: float,
        robot_width: float,
        filtered_command_x: np.ndarray,
        filtered_command_y: np.ndarray,
        filtered_sensor_x: np.ndarray,
        filtered_sensor_y: np.ndarray,
        filtered_states: np.ndarray,
        remove_offset: bool,
        p0_min: float,
        p0_max: float,
        p1_min: float,
        p1_max: float,
        p2_min: float,
        p2_max: float,
        p3_min: float,
        p3_max: float,
    ) -> Tuple:
        """Find the best prior in multi-function case."""

        #For the treadmill we adjust for the angle
        #yaw_angle = filtered_states[:, [5]]
        #filtered_command_x = np.multiply(filtered_command_x,np.cos(yaw_angle))

        # Build the datasets
        dataset_x = np.concatenate([filtered_command_x, filtered_command_y], axis=-1)
        dataset_y = np.concatenate([filtered_sensor_x, filtered_sensor_y], axis=-1)
        selected_state = filtered_states[:, [state_dim]]
        
        # Clip and reshape the state
        signs = np.sign(selected_state)
        selected_state = np.repeat(selected_state.reshape(-1, 1), 2, axis=0)
        selected_state = np.clip(
            selected_state, a_min=state_min_opt_clip, a_max=state_max_opt_clip
        )

        # Compute the matrices for the least square
        A = (dataset_y - dataset_x).reshape(-1, 1)
        offset = np.concatenate(
            [np.ones(shape=signs.shape), np.zeros(shape=signs.shape)], axis=1
        ).reshape(-1, 1)

        ## No Offset
        if remove_offset:
            offset =np.zeros(shape=offset.shape)

        a = np.concatenate(
            [-signs * 0.5, np.ones(shape=signs.shape) * robot_width / 4], axis=1
        )
        b = np.concatenate(
            [np.ones(shape=signs.shape) * 1 / robot_width, -signs * 0.5], axis=1
        )
        B = np.stack(
            [
                np.multiply(dataset_x, a).sum(axis=1),
                np.multiply(dataset_x, b).sum(axis=1),
            ],
            axis=-1,
        ).reshape(-1, 1)

        # Make a list of functions to attempt to regress to
        functions_list = [
            np.ones(shape=(len(selected_state), 1)),  # constant
            selected_state,  # order 1 poly
            np.sign(selected_state) * selected_state**2,  # order 2 poly
            selected_state**3,  # order 3 poly
        ]

        # Compute regression using least squares for all the functions in function list
        lstsq_rests = np.ones(len(functions_list)) * np.inf # Default to inf
        lstsq_coeffs = np.zeros(len(functions_list))
        lstsq_offsets = np.zeros(len(functions_list))
        mins = [p0_min, p1_min, p2_min, p3_min]
        maxs = [p0_max, p1_max, p2_max, p3_max]
        for function_index in range (len(functions_list)):

            # Compute the least squares
            B_new = np.multiply(B, functions_list[function_index])
            # B_new = np.concatenate([B_new, offset], axis=1)
            # coeff, rest, _, _ = np.linalg.lstsq(B_new, A)
            res = lsq_linear(B_new,A.squeeze(), bounds=([mins[function_index]], [maxs[function_index]]))
            coeff = res.x
            rest = res.cost

            # Store the resulting coeff and rests
            lstsq_coeffs[function_index] = coeff[0]
            # lstsq_offsets[function_index] = coeff[1]
            if rest:
                lstsq_rests[function_index] = rest

        # Get best function to fit too using the rests values
        best_func = np.argmin(lstsq_rests)
        #print(lstsq_coeffs)
        #print(lstsq_rests, flush=True)
        # self.log_model.fit()

        # Keep coeffs of all functions zero except the best function
        p_choice = np.zeros(shape=(len(functions_list) + 1))  # +1 for offset
        p_choice[best_func] = lstsq_coeffs[best_func]
        p_choice[-1] = lstsq_offsets[best_func]


        return p_choice, lstsq_rests[best_func]
