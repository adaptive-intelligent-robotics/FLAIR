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

import gpjax as gpx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from functionality_controller.datapoint import DataPoints
from functionality_controller.dataset_fifo import FIFODataset
from functionality_controller.dataset_grid import OUT_OF_BOUND, InputGridFIFODataset
from functionality_controller.gp.gpjax_map_v2 import MAPMean, MAPPrior
from gpjax.base import meta_leaves, meta_map
from gpjax.dataset import Dataset
from gpjax.fit import fit
from gpjax.kernels import RBF, Matern52


class AdaptiveGP:
    """GP class used by NO_STATE_FAST and NO_STATE_FAST_RESET versions of the GP."""

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
        p_soft_update_size: float,
        min_spread: float,
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
            p_soft_update_size: smooth update of prior value via soft update.
            min_spread: minimum datapoint spread along x-command-axis to start training.
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
        self.p_soft_update_size = p_soft_update_size
        self.min_spread = min_spread
        self.minibatch_size = minibatch_size
        self.auto_reset_error_buffer_size = auto_reset_error_buffer_size
        self.auto_reset_angular_rot_weight = auto_reset_angular_rot_weight
        self.auto_reset_threshold = auto_reset_threshold

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
            self.dataset = InputGridFIFODataset.create(
                size=dataset_size,
                min_command=min_command,
                max_command=max_command,
                cell_depth=dataset_grid_cell_size,
                input_bins=grid_resolution,
                prop_neighbours=dataset_grid_neighbours,
                novelty_threshold=dataset_grid_novelty_threshold,
                datapoint_batch_size=datapoint_batch_size,
            )
            self.dataset_add_fn = InputGridFIFODataset.add
            self.dataset_reset_fn = InputGridFIFODataset.reset
        else:
            # FIFO dataset
            self.dataset = FIFODataset.create(dataset_size)
            self.dataset_add_fn = FIFODataset.add
            self.dataset_reset_fn = FIFODataset.reset

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

        # Default empty parameters
        self.default_rotation = jnp.asarray([1.0, 1.0])
        self.default_offset = jnp.zeros(shape=(2,))

        # Init GP objects
        self.all_corrected_descriptors = self.all_descriptors
        self.uncertainties = jnp.zeros(shape=(self.all_descriptors.shape[0], 1))
        self.kernel = Matern52(active_dims=[_ for _ in range(2)])
        self.kernel = self.kernel.replace(
            lengthscale=self.default_lengthscale,
            variance=self.default_variance,
        )
        self.kernel_x = np.zeros(shape=(1, 1))
        self.x_opt_posterior = None
        self.xy_learned_params = {"mean_function": {}}

        # Init P parameter
        self.p = 0.0
        self.raw_p = 0.0

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
        self.minibatch_dataset = self.minibatch_dataset_reset_fn(self.minibatch_dataset)

        # Reset counters
        self.total_position = 0
        self.minibatch_total_position = 0
        self.latest_train_total_position = 0
        self.mean_intent_error_buffer = []

        # Reset GP
        self.all_corrected_descriptors = self.all_descriptors
        self.uncertainties = jnp.zeros(shape=(self.all_descriptors.shape[0], 1))

        # Init P parameter
        self.p = 0.0
        self.raw_p = 0.0

        # Reset xy_learned_params
        self.xy_learned_params["mean_function"]["rotation"] = self.default_rotation
        self.xy_learned_params["mean_function"][
            "offset"
        ] = self.x_opt_posterior.prior.mean_function.offset
        self.xy_learned_params["mean_function"][
            "obs_noise"
        ] = self.x_opt_posterior.likelihood.obs_noise

    def update(self, datapoint: DataPoints) -> None:
        """Update the GP dataset, adding the latest datapoints."""

        # self._logger.debug("Entering update")

        # Update the dataset
        self.dataset = self.dataset_add_fn(dataset=self.dataset, datapoint=datapoint)

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

    def gp_update(self) -> Tuple[bool, float, float]:
        """Update the GP model based on its dataset.
        Return:
            a boolean indicating if the model has been trained.
            a timer for the optimisation.
            a timer for the fit.
        """

        # If enough datapoints, train the model
        if (
            self.total_position
            > self.latest_train_total_position + self.min_diff_datapoint
        ):
            # self._logger.debug("Start Training")
            opt_time, gp_fit_time = self._train_model()
            self.latest_train_total_position = self.total_position
            # self._logger.debug("Done Training")

            return True, opt_time, gp_fit_time

        # If not enough datapoints, do not train
        return False, 0.0, 0.0

    def auto_reset(self) -> Tuple[bool, float]:
        """Test the reset condition, reset the dataset if necessary.

        Return:
            auto_reset: True if auto-reset needed
            error: error value to send to Grafana
        """

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
        intent_gp_error_y = (
            np.abs(intent_y - gp_pred_y) * self.auto_reset_angular_rot_weight
        )

        cmd_sensor_error_x = np.abs(cmd_x - sensor_x)
        cmd_sensor_error_y = (
            np.abs(cmd_y - sensor_y) * self.auto_reset_angular_rot_weight
        )

        error_distance_x = np.abs(intent_gp_error_x - cmd_sensor_error_x)
        error_distance_y = np.abs(intent_gp_error_y - cmd_sensor_error_y)

        # Mean the error across the datapoint in the minibatch
        mean_error_distance_x = np.mean(error_distance_x)
        mean_error_distance_y = np.mean(error_distance_y)

        # Average the errors to collapse two dimensions in one
        final_error_distance_norm = np.sqrt(
            mean_error_distance_x**2 + mean_error_distance_y**2
        )

        # Store the current error in the error buffer
        mean_intent_error = np.mean(final_error_distance_norm)
        self.mean_intent_error_buffer.append(mean_intent_error)

        # If the buffer is not filled, do not auto-reset
        if len(self.mean_intent_error_buffer) < self.auto_reset_error_buffer_size:
            return False, 0.0, mean_intent_error

        # Else, just select the self.auto_reset_error_buffer_size latest errors
        self.mean_intent_error_buffer = self.mean_intent_error_buffer[
            -self.auto_reset_error_buffer_size :
        ]

        # If there is an increase in the error in the buffer, reset
        amplitude_increase = np.max(self.mean_intent_error_buffer) - np.min(
            self.mean_intent_error_buffer
        )
        if amplitude_increase > self.auto_reset_threshold and (
            np.argmax(self.mean_intent_error_buffer)
            > np.argmin(self.mean_intent_error_buffer)
        ):
            # Reset the GP, dataset and minibatch
            self.reset()
            return True, amplitude_increase, mean_intent_error

        return False, amplitude_increase, mean_intent_error

    def _train_model(self) -> Tuple[int, int]:
        """Train the model."""

        # Create a new GP
        # TODO: needed ? could be done in __init__
        self.kernel = self.kernel.replace(
            lengthscale=self.default_lengthscale,
            variance=self.default_variance,
        )
        self.mean_fn_x = MAPMean(chosen_bd=0).replace(
            rotation=self.default_rotation,
            offset=self.default_offset,
        )

        if self.x_opt_posterior is None:

            # Prior, likelihood and posterior for X
            x_prior = MAPPrior(kernel=self.kernel, mean_function=self.mean_fn_x)
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

        # Compute the variance and ensure it is high enough to avoid collapse
        nan_command = jnp.concatenate(
            [
                jnp.where(
                    self.dataset.command_x == OUT_OF_BOUND,
                    jnp.nan,
                    self.dataset.command_x,
                ),
                jnp.where(
                    self.dataset.command_y == OUT_OF_BOUND,
                    jnp.nan,
                    self.dataset.command_y,
                ),
            ],
            axis=-1,
        )
        spread = jnp.abs(
            jnp.nanmax(nan_command, axis=0) - jnp.nanmin(nan_command, axis=0)
        )
        if spread[0] < self.min_spread:
            self._logger.debug(
                f"WARNING spread is too low {spread}, NOT TRAINING THE GP."
            )
            return 0.0, 0.0

        ####### TIMED #########

        # Call optimisation
        # self._logger.debug(f"Entering p optimisation")
        start_t = time.time()
        p = AdaptiveGP._p_optimisation(
            filtered_command_x=self._to_cpu(self.dataset.command_x),
            filtered_command_y=self._to_cpu(self.dataset.command_y),
            filtered_sensor_x=self._to_cpu(self.dataset.sensor_x),
            filtered_sensor_y=self._to_cpu(self.dataset.sensor_y),
            robot_width=self.robot_width,
        )
        # self._logger.debug(f"Done p optimisation")

        # Enforce p clipping
        p = np.clip(p, -self.max_p_value, self.max_p_value)

        # Enforce soft Update of P to avoid sudden jump
        self.raw_p = p
        self.p = self.p - np.clip(
            self.p - p, a_min=-self.p_soft_update_size, a_max=self.p_soft_update_size
        )

        optimization_time = time.time() - start_t

        ######### end of TIMED ##########

        # Enable Training
        mean_fn_x = MAPMean(chosen_bd=0).replace(rotation=jnp.asarray([self.p]))
        x_prior = MAPPrior(kernel=self.kernel, mean_function=mean_fn_x)
        self.x_opt_posterior = self.x_opt_posterior.replace(prior=x_prior)
        self.xy_learned_params["mean_function"]["rotation"] = jnp.asarray(
            [
                jnp.min(jnp.asarray([1, 1 - self.p])),
                jnp.min(jnp.asarray([1, 1 + self.p])),
            ]
        )

        self._logger.debug(
            f"New introspection: p1={self.xy_learned_params['mean_function']['rotation'][0]}, p2={self.xy_learned_params['mean_function']['rotation'][1]}"
        )

        ####### TIMED #########

        # Call fit
        # self._logger.debug(f"Entering fit")
        start_t = time.time()
        (
            self.all_corrected_descriptors,
            self.uncertainties,
            self.random_key,
        ) = AdaptiveGP._fit(
            x_posterior=self.x_opt_posterior,
            all_descriptors=self.all_descriptors,
            dataset=self.dataset,
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
        dataset: FIFODataset,
        random_key: jax.random.KeyArray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jax.random.KeyArray, float, float]:
        """Fit the GP to the current dataset."""

        @jax.jit
        def get_all_corrected_descriptors(
            x_posterior: Any,
            test_inputs: jnp.ndarray,
            training_data: Dataset,
            weights_noise: jnp.ndarray,
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            """
            Get the descriptor with the additional error computed by the GP model.
            """

            # Compute predictive mean and std
            predictive_mean, predictive_std, cov_term = jax.jit(
                x_posterior.predict_all
            )(test_inputs, train_data=training_data, weights_noise=weights_noise)

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

            # Return
            return predictive_mean, uncertainties, cov_term

        # Get dataset
        observations_dataset = Dataset(
            X=jnp.concatenate([dataset.command_x, dataset.command_y], axis=-1),
            y=jnp.concatenate([dataset.sensor_x, dataset.sensor_y], axis=-1),
        )

        weights_noise = jnp.ones((dataset.command_x.shape[0]))
        # Apply weight for accumulated datapoints
        # weights_noise = jnp.squeeze(1 / dataset.num_accumulate)
        # if len(dataset.num_accumulate) == 1:
        #    weights_noise = jnp.expand_dims(weights_noise, axis=0)

        # Compute corrected bd
        (all_corrected_descriptors, uncertainties, L,) = get_all_corrected_descriptors(
            x_posterior,
            test_inputs=all_descriptors,
            training_data=observations_dataset,
            weights_noise=weights_noise,
        )

        return (
            all_corrected_descriptors,
            uncertainties,
            random_key,
        )

    @staticmethod
    def _p_optimisation(
        filtered_command_x: np.ndarray,
        filtered_command_y: np.ndarray,
        filtered_sensor_x: np.ndarray,
        filtered_sensor_y: np.ndarray,
        robot_width: float,
    ) -> float:
        """Find the best prior."""

        # @jax.jit
        def lst_sq(dataset_x: np.ndarray, dataset_y: np.ndarray) -> np.ndarray:
            """Compute least squares to find the p-values."""

            A = (dataset_y - dataset_x).reshape(-1, 1)

            sign = -1
            transform = np.asarray(
                [[1 / 2, (robot_width / 4)], [(1 / robot_width), 1 / 2]]
            ).T
            B = (dataset_x @ transform).reshape(-1, 1)
            p_choice_1, rest1, _, _ = np.linalg.lstsq(B, A)

            transform = np.asarray(
                [[-1 / 2, (robot_width / 4)], [(1 / robot_width), -1 / 2]]
            ).T
            C = (dataset_x @ transform).reshape(-1, 1)
            p_choice_2, rest2, _, _ = np.linalg.lstsq(C, A)

            if np.abs(rest1) < np.abs(rest2):
                p = p_choice_1.squeeze()
                p1 = 1
                p2 = 1 + p
            else:
                p = p_choice_2.squeeze()
                p1 = 1 - p
                p2 = 1

            return p

        # Check the length
        dataset_n = filtered_sensor_y.shape[0]
        if dataset_n < 2:
            return 0.0

        # Concatenate to have the good shape
        dataset_x = np.concatenate([filtered_command_x, filtered_command_y], axis=-1)
        dataset_y = np.concatenate([filtered_sensor_x, filtered_sensor_y], axis=-1)

        # Compute leaast square
        p = lst_sq(dataset_x, dataset_y)

        return p
