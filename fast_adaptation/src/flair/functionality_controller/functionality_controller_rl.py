# Written or Adapted by the Imperial College London Team for the FLAIR project, 2023
# Authors for this file:
# Maxime Allard
# Manon Flageat
# Bryan Lim
# Antoine Cully

from functools import partial
from typing import Any, Tuple

from jax.config import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import optax
import numpy as np
from functionality_controller.residual_rl_td3 import Actor, TrainState


class FunctionalityControllerRL:
    """Main FunctionalityController class, call at each lopp."""

    def __init__(
        self,
        min_command: int,
        max_command: int,
        learning_rate: float,
    ) -> None:
        """
        Args:
            grid_resolution: resolution of the command grid, used by all processes across FC.
            min_command: min value for the command grid, used by all processes across FC.
            max_command: max value for the command grid, used by all processes across FC.
            state_dim: which dimension to use for the state.
            state_min_opt_clip: min to clip the state before prior computation.
            state_max_opt_clip: max to clip the state before prior computation.
            robot_width: parameter of the robot used for prior computation.
        """

        # Init an empty observation just for network initialisation
        self.dummy_observations = jnp.zeros((10,))

        # Init RL components
        self.learning_rate = learning_rate
        self.actor_key = jax.random.PRNGKey(0)
        self.actor = Actor(
            action_dim=2,
            action_scale=jnp.array(
                (max_command - min_command) / 2.0
            ),
            action_bias=jnp.array(
                (max_command + min_command) / 2.0
            ),
        )
        self.params=self.actor.init(self.actor_key, self.dummy_observations)

        # Create all adaptation objects
        self.reset()

    def reset(self):
        """Reset the FC."""

        # Reset RL components
        self.params=self.actor.init(self.actor_key, self.dummy_observations)

    def get_command(
        self,
        joystick_linear_x_human_command: float,
        joystick_angular_z_human_command: float,
        flipper_angular_x_human_command: float,
        state: np.array,
        sensor_x: float,
        sensor_y: float,
    ) -> Tuple[float, float, float, float, float, float, float]:
        """Main function called by the pipeline to get the command to execute.

        Args:
            joystick_linear_x_human_command
            joystick_angular_z_human_command
            flipper_angular_x_human_command
            use_state: indicate if state-dependent prior only
            use_state_gp: indicate if state-dependent prior with GP

        Returns:
            the new command to forward to the safety controller in form of a msg.
        """
        
        if state is None or sensor_x is None or sensor_y is None:
            return (
                joystick_linear_x_human_command, 
                joystick_angular_z_human_command, 
                flipper_angular_x_human_command,
            )

        # Scale joystick signal
        intent_x = joystick_linear_x_human_command
        intent_y = joystick_angular_z_human_command

        # Build observation
        observations = jnp.concatenate(
            [
                jnp.asarray([intent_x, intent_y]),
                jnp.asarray(state),
            ],
            axis=0,
        )

        # Get actor prediction
        residual = self.actor.apply(self.params, observations)

        # Return action to be published
        linear_velocity = intent_x + float(residual[0])
        angular_velocity = intent_y + float(residual[1])

        return (
            linear_velocity,
            angular_velocity,
            flipper_angular_x_human_command,
        )
