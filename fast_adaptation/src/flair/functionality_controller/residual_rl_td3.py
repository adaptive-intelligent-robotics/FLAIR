# Written or Adapted by the Imperial College London Team for the FLAIR project, 2023
# Authors for this file:
# Maxime Allard
# Manon Flageat
# Bryan Lim
# Antoine Cully

from typing import Any, Dict, Tuple, Callable

from jax.config import config

config.update("jax_enable_x64", True)
import copy
import time
from functools import partial

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from functionality_controller.datapoint import DataPoints
from functionality_controller.dataset_fifo import FIFODataset, StateFIFODataset

class TrainState(TrainState):
        target_params: flax.core.FrozenDict

class QNetwork(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class Actor(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim, kernel_init=jax.nn.initializers.zeros, bias_init=jax.nn.initializers.zeros)(x)
        x = nn.tanh(x)
        x = x * self.action_scale + self.action_bias
        return x

@partial(jax.jit, static_argnames=("actor_apply", "qf_apply", ))
def update_critic(
    actor_state: TrainState,
    qf1_state: TrainState,
    qf2_state: TrainState,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    rewards: jnp.ndarray,
    random_key: jnp.ndarray,
    gamma: jnp.ndarray,
    min_command: jnp.ndarray,
    max_command: jnp.ndarray,
    policy_noise: jnp.ndarray,
    noise_clip: jnp.ndarray,
    action_scale: jnp.ndarray,
    actor_apply: Callable,
    qf_apply: Callable,
):
    """Adapted from Clean RL: https://github.com/vwxyzjn/cleanrl."""

    random_key, noise_random_key = jax.random.split(random_key, 2)
    clipped_noise = (
        jnp.clip(
            (jax.random.normal(noise_random_key, actions.shape) * policy_noise),
            -noise_clip,
            noise_clip,
        )
        * action_scale
    )
    next_state_actions = jnp.clip(
        actor_apply(actor_state.target_params, next_observations)
        + clipped_noise,
        min_command,
        max_command,
    )
    qf1_next_target = qf_apply(
        qf1_state.target_params, next_observations, next_state_actions
    ).reshape(-1)
    qf2_next_target = qf_apply(
        qf2_state.target_params, next_observations, next_state_actions
    ).reshape(-1)
    min_qf_next_target = jnp.minimum(qf1_next_target, qf2_next_target)
    next_q_value = (rewards + gamma * (min_qf_next_target)).reshape(-1)

    @jax.jit
    def mse_loss(params):
        qf_a_values = qf_apply(params, observations, actions).squeeze()
        return ((qf_a_values - next_q_value) ** 2).mean(), qf_a_values.mean()

    (qf1_loss_value, qf1_a_values), grads1 = jax.value_and_grad(mse_loss, has_aux=True)(qf1_state.params)
    (qf2_loss_value, qf2_a_values), grads2 = jax.value_and_grad(mse_loss, has_aux=True)(qf2_state.params)
    qf1_state = qf1_state.apply_gradients(grads=grads1)
    qf2_state = qf2_state.apply_gradients(grads=grads2)

    return (
        (qf1_state, qf2_state),
        (qf1_loss_value, qf2_loss_value),
        (qf1_a_values, qf2_a_values),
        random_key,
    )

@partial(jax.jit, static_argnames=("actor_apply", "qf_apply", ))
def update_actor(
    actor_state: TrainState,
    qf1_state: TrainState,
    qf2_state: TrainState,
    observations: jnp.ndarray,
    tau: jnp.ndarray,
    actor_apply: Callable,
    qf_apply: Callable,
):
    """Adapted from Clean RL: https://github.com/vwxyzjn/cleanrl."""

    @jax.jit
    def actor_loss(params):
        return -qf_apply(qf1_state.params, observations, actor_apply(params, observations)).mean()

    actor_loss_value, grads = jax.value_and_grad(actor_loss)(actor_state.params)
    actor_state = actor_state.apply_gradients(grads=grads)
    actor_state = actor_state.replace(
        target_params=optax.incremental_update(
            actor_state.params, actor_state.target_params, tau
        )
    )

    qf1_state = qf1_state.replace(
        target_params=optax.incremental_update(
            qf1_state.params, qf1_state.target_params, tau
        )
    )
    qf2_state = qf2_state.replace(
        target_params=optax.incremental_update(
            qf2_state.params, qf2_state.target_params, tau
        )
    )
    return actor_state, (qf1_state, qf2_state), actor_loss_value


class ResidualTD3:
    def __init__(
        self,
        logger: Any,
        jiting_datapoints: DataPoints,
        min_command: int,
        max_command: int,
        learning_rate: float,
        policy_noise: float,
        noise_clip: float,
        gamma: float,
        tau: float,
        batch_size: int,
        policy_frequency: int,
        min_diff_datapoint: int,
        dataset_size: int,
        datapoint_batch_size: int,
        minibatch_size: int,
        auto_reset_error_buffer_size: int,
        auto_reset_angular_rot_weight: float,
        auto_reset_threshold: float,
    ) -> None:
        """
        Args:
            logger
            jiting_datapoints: empty DataPoints uses to trigger jitting of the Dataset.
            default_obs_noise: property of the robot.
            default_lengthscale: property of the robot.
            default_variance: property of the robot.
            min_diff_datapoint: minimum number of datapoint diff to train.
            dataset_size: temporal FIFO size.
            datapoint_batch_size: size of datapoints, used bu dataset.
            minibatch_size: size of the minibatch used for auto-reset.
            auto_reset_error_buffer_size: size of tge error buffer used for auto-reset.
            auto_reset_angular_rot_weight: weight of the angular-command in the auto-reset formula.
            auto_reset_threshold: threshold to trigger auto-reset.
        """

        # Store attributes
        self._logger = logger
        self.min_command = min_command
        self.max_command = max_command
        self.learning_rate = learning_rate
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.policy_frequency = policy_frequency
        self.min_diff_datapoint = min_diff_datapoint
        self.minibatch_size = minibatch_size
        self.auto_reset_error_buffer_size = auto_reset_error_buffer_size
        self.auto_reset_angular_rot_weight = auto_reset_angular_rot_weight
        self.auto_reset_threshold = auto_reset_threshold

        # Set random key
        self.random_key = jax.random.PRNGKey(0)

        # Create the jited dataset handling addition
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

        # Init an empty observation and action just for network initialisation
        self.random_key, obs_key, action_key = jax.random.split(self.random_key, 3)
        self.dummy_observations = jax.random.uniform(obs_key, (10,), minval=0.0, maxval=1.0)
        self.dummy_actions = jax.random.uniform(action_key, (2,), minval=0.0, maxval=1.0)

        # Init RL networks
        self.action_scale = jnp.array((max_command - min_command) / 2.0)
        self.action_bias = jnp.array((max_command + min_command) / 2.0)
        self.actor = Actor(
            action_dim=2,
            action_scale=self.action_scale,
            action_bias=self.action_bias,
        )
        self.qf = QNetwork()
        self.actor_apply = jax.jit(self.actor.apply)
        self.qf_apply = jax.jit(self.qf.apply)

        # Init RL params with last layers to 0 for residual
        self.random_key, actor_key = jax.random.split(self.random_key, 2)
        default_actor_params = self.actor.init(actor_key, self.dummy_observations)
        self.random_key, qf1_key = jax.random.split(self.random_key, 2)
        default_qf1_params = self.qf.init(qf1_key, self.dummy_observations, self.dummy_actions)
        self.random_key, qf2_key = jax.random.split(self.random_key, 2)
        default_qf2_params = self.qf.init(qf2_key, self.dummy_observations, self.dummy_actions)

        # Init RL training states
        self.actor_state = TrainState.create(
            apply_fn=self.actor_apply,
            params=default_actor_params,
            target_params=default_actor_params,
            tx=optax.adam(learning_rate=self.learning_rate),
        )
        self.qf1_state = TrainState.create(
            apply_fn=self.qf_apply,
            params=default_qf1_params,
            target_params=default_qf1_params,
            tx=optax.adam(learning_rate=self.learning_rate),
        )
        self.qf2_state = TrainState.create(
            apply_fn=self.qf_apply,
            params=default_qf2_params,
            target_params=default_qf2_params,
            tx=optax.adam(learning_rate=self.learning_rate),
        )
        self.last_trained_actor = 0

        # Call functions to trigger jiting
        self.update(jiting_datapoints)
        _ = self._train_model()
        self.dataset = self.dataset_reset_fn(self.dataset)

    def _to_cpu(self, array: jnp.ndarray):
        """Get a dataset array, tranfer it to numpy and remove OUT_OF_BOUND."""
        cpu_array = np.asarray(array)
        cpu_array = cpu_array[~((cpu_array == OUT_OF_BOUND).any(axis=1)), ...]
        return cpu_array

    def reset(self) -> None:
        """Reset the controller adaptation."""

        self._logger.debug(f"RESETING RL" * 20)

        # Reset dataset
        self.dataset = self.dataset_reset_fn(self.dataset)

        # Reset minibatch
        self.minibatch_dataset = self.minibatch_dataset_reset_fn(self.minibatch_dataset)

        # Reset counters
        self.total_position = 0
        self.minibatch_total_position = 0
        self.latest_train_total_position = 0

        # Init RL params with last layers to 0 for residual
        self.random_key, actor_key = jax.random.split(self.random_key, 2)
        default_actor_params = self.actor.init(actor_key, self.dummy_observations)
        self.random_key, qf1_key = jax.random.split(self.random_key, 2)
        default_qf1_params = self.qf.init(qf1_key, self.dummy_observations, self.dummy_actions)
        self.random_key, qf2_key = jax.random.split(self.random_key, 2)
        default_qf2_params = self.qf.init(qf2_key, self.dummy_observations, self.dummy_actions)

        # Init RL training states
        self.actor_state = TrainState.create(
            apply_fn=self.actor_apply,
            params=default_actor_params,
            target_params=default_actor_params,
            tx=optax.adam(learning_rate=self.learning_rate),
        )
        self.qf1_state = TrainState.create(
            apply_fn=self.qf_apply,
            params=default_qf1_params,
            target_params=default_qf1_params,
            tx=optax.adam(learning_rate=self.learning_rate),
        )
        self.qf2_state = TrainState.create(
            apply_fn=self.qf_apply,
            params=default_qf2_params,
            target_params=default_qf2_params,
            tx=optax.adam(learning_rate=self.learning_rate),
        )
        self.last_trained_actor = 0

    def update(self, datapoint: DataPoints) -> None:
        """Update the RL dataset, adding the latest datapoints."""

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

    def rl_update(self) -> Tuple[bool, float, float]:
        """Update the RL model based on its dataset.
        Return:
            a boolean indicating if the model has been trained.
            a timer for the optimisation.
            a timer for the fit.
        """

        # If enough datapoints, train the model
        if (
            self.total_position
            > self.latest_train_total_position + self.min_diff_datapoint
            and self.total_position > 2 * self.batch_size
        ):
            self._logger.debug("Start Training")
            self._train_model()
            self.latest_train_total_position = self.total_position
            self._logger.debug("Done Training")

            return True

        # If not enough datapoints, do not train
        return False

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
        mean_intent_error = np.mean(final_error_distance_norm)

        # Just for debugging, compute the mean_error_distance_x one
        mean_intent_error_x = np.mean(np.abs(mean_error_distance_x))

        # If the error is bigger than threshold, reset
        if mean_intent_error > self.auto_reset_threshold:
            self.reset()
            self.last_reset_t = time.time()
            return True, mean_intent_error_x, mean_intent_error

        return False, mean_intent_error_x, mean_intent_error

    def _train_model(self) -> Tuple[int, int]:
        """Train the model."""

        # Do as many training steps as new datapoints
        for step in range(self.total_position - self.latest_train_total_position):

            # Sample from dataset
            (
                samples_command_x,
                samples_command_y,
                samples_gp_prediction_x,
                samples_gp_prediction_y,
                samples_intent_x,
                samples_intent_y,
                samples_sensor_x,
                samples_sensor_y,
                samples_state,
                next_samples_command_x,
                next_samples_command_y,
                next_samples_gp_prediction_x,
                next_samples_gp_prediction_y,
                next_samples_intent_x,
                next_samples_intent_y,
                next_samples_sensor_x,
                next_samples_sensor_y,
                next_samples_state,
                self.random_key,
            ) = self.dataset.sample_with_next(self.dataset, self.batch_size, self.random_key)

            # Transform samples into RL data
            observations = jnp.concatenate(
                [
                    jnp.expand_dims(samples_intent_x, axis=1),
                    jnp.expand_dims(samples_intent_y, axis=1),
                    samples_state,
                ],
                axis=1,
            )
            next_observations = jnp.concatenate(
                [
                    jnp.expand_dims(next_samples_intent_x, axis=1),
                    jnp.expand_dims(next_samples_intent_y, axis=1),
                    next_samples_state,
                ],
                axis=1,
            )

            actions = jnp.concatenate(
                [
                    jnp.expand_dims(samples_command_x - samples_intent_x, axis=1),
                    jnp.expand_dims(samples_command_y - samples_intent_y, axis=1),
                ],
                axis=1,
            )

            # Compute a reward
            rewards = -jnp.linalg.norm(jnp.concatenate(
                [
                    jnp.expand_dims(samples_intent_x - samples_sensor_x, axis=1),
                    jnp.expand_dims(samples_intent_y - samples_sensor_y, axis=1),
                ],
                axis=1,
            ), axis=1)

            # Train critic
            (
                (self.qf1_state, self.qf2_state),
                (qf1_loss_value, qf2_loss_value),
                (qf1_a_values, qf2_a_values),
                self.random_key,
            ) = update_critic(
                actor_state=self.actor_state,
                qf1_state=self.qf1_state,
                qf2_state=self.qf2_state,
                observations=observations,
                actions=actions,
                next_observations=next_observations,
                rewards=rewards,
                random_key=self.random_key,
                gamma=self.gamma,
                min_command=self.min_command,
                max_command=self.max_command,
                policy_noise=self.policy_noise,
                noise_clip=self.noise_clip,
                action_scale=self.action_scale,
                actor_apply=self.actor_apply,
                qf_apply=self.qf_apply,
            )

            # Train actor
            if (self.latest_train_total_position + step) > (
                self.last_trained_actor + self.policy_frequency
            ):
                (
                    self.actor_state,
                    (self.qf1_state, self.qf2_state),
                    actor_loss_value,
                ) = update_actor(
                    actor_state=self.actor_state,
                    qf1_state=self.qf1_state,
                    qf2_state=self.qf2_state,
                    observations=observations,
                    tau=self.tau,
                    actor_apply=self.actor_apply,
                    qf_apply=self.qf_apply,
                )
                self.last_trained_actor = self.latest_train_total_position + step

        return 


