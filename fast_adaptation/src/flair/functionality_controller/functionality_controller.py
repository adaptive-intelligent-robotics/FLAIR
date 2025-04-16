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
import numpy as np
from gpjax.kernels import RBF
from gpjax.kernels.stationary.utils import squared_distance


class FunctionalityController:
    """Main FunctionalityController class, call at each lopp."""

    def __init__(
        self,
        grid_resolution: int,
        min_command: int,
        max_command: int,
        state_dim: int,
        state_min_opt_clip: float,
        state_max_opt_clip: float,
        robot_width: float,
        max_p_value: float,
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

        # Store the attributes
        self.min_command = min_command
        self.max_command = max_command
        self.state_dim = state_dim
        self.state_min_opt_clip = state_min_opt_clip
        self.state_max_opt_clip = state_max_opt_clip
        self.robot_width = robot_width
        self.max_p_value = max_p_value

        # Create the grid
        x_axis = jnp.linspace(self.min_command, self.max_command, num=grid_resolution)
        self.all_descriptors = jnp.asarray(jnp.meshgrid(x_axis, x_axis)).T.reshape(
            -1, 2
        )
        self.all_genotypes = jnp.concatenate(
            [self.all_descriptors, jnp.zeros(shape=(self.all_descriptors.shape[0], 1))],
            axis=-1,
        )

        # Create all adaptation objects
        self.reset()

    def reset(self):
        """Reset the FC."""
        self.all_corrected_descriptors = self.all_descriptors
        self.uncertainties = jnp.zeros(shape=(self.all_descriptors.shape[0], 1))
        self.learned_params = jnp.asarray([1.0, 1.0, 0.0, 0.0])
        self.cov_alpha = jnp.zeros(shape=(2, 2))
        self.state = None
        self.max_x = self.max_command
        self.max_y = self.max_command
        self.dataset_state = None
        self.kernel_x = None
        self.base_kernel = RBF(active_dims=[0]).replace(
            lengthscale=jnp.array([10.0]),
            variance=jnp.array([1.0]),  # HACK, needs to be 1
        )

    @staticmethod
    @jax.jit
    def _get_gp_only_prior(
        all_corrected_descriptors: jnp.ndarray,
        uncertainties: jnp.ndarray,
        all_genotypes: jnp.ndarray,
        descriptor: jnp.ndarray,
    ) -> jnp.ndarray:
        """Find the genotype to execute as the genotypes giving the
        closest corrected descriptors.
        """

        # Compute distance to corrected descriptors
        @jax.jit
        def distance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            return jnp.sqrt(jnp.sum(jnp.square(x - y)))

        distances = jax.vmap(partial(distance, y=descriptor))(all_corrected_descriptors)

        # Return closest genotype
        indice = jnp.argmax(-distances)
        return (
            all_genotypes.at[indice.squeeze()].get(),
            jnp.expand_dims(all_corrected_descriptors.at[indice].get(), axis=0),
            jnp.expand_dims(uncertainties.at[indice].get(), axis=0),
        )

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=(
            "min_command",
            "max_command",
            "state_dim",
            "state_min_opt_clip",
            "state_max_opt_clip",
            "robot_width",
        ),
    )
    def _get_state_gp_prior(
        max_p_value: float,
        min_command: float,
        max_command: float,
        state_dim: int,
        state_min_opt_clip: float,
        state_max_opt_clip: float,
        robot_width: float,
        learned_params: Any,
        state: jnp.ndarray,
        all_descriptors: jnp.ndarray,
        cov_alpha: jnp.ndarray,
        kernel: Any,
        dataset_state: jnp.ndarray,
        Kxx: jnp.ndarray,
    ) -> jnp.ndarray:

        # Get the polynomial coefficients: a + b*x + c*x**2 + d*x**3
        a = learned_params[2]
        b = learned_params[3]
        c = learned_params[4]
        d = learned_params[5]
        offset = learned_params[6]

        # Compute the polynomial
        clip_state = jnp.clip(
            state[state_dim], a_min=state_min_opt_clip, a_max=state_max_opt_clip
        )
        new_p = a + b * clip_state + c * clip_state**2 + d * clip_state**3
        new_p = jnp.clip(new_p, a_min=-max_p_value, a_max=max_p_value)

        # Get corresponding prior
        p1, p2 = jnp.min(jnp.asarray([1, 1 - new_p])), jnp.min(
            jnp.asarray([1, 1 + new_p])
        )
        diag = (p1 + p2) / 2
        scaling = jnp.asarray(
            [[diag, (p2 - p1) * robot_width / 4], [(p2 - p1) / robot_width, diag]]
        ).T

        # Get corresponding GP mean and kernel
        new_map_mean = jnp.matmul(all_descriptors, scaling) + jnp.asarray([offset, 0.0])
        Kxt_state = jax.vmap(
            lambda x: jax.vmap(lambda y: kernel(x, y))(
                jnp.broadcast_to(state[state_dim], shape=(all_descriptors.shape[0], 1))
            )
        )(dataset_state[:, [state_dim]])
        Kxt_full = jnp.multiply(Kxx, Kxt_state.squeeze())
        cov_offset = jnp.matmul(jnp.transpose(Kxt_full), cov_alpha)

        # Get final all corrected descriptor from mean and covariance
        all_corrected_descriptors = new_map_mean + cov_offset

        return all_corrected_descriptors

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=(
            "min_command",
            "max_command",
            "state_dim",
            "state_min_opt_clip",
            "state_max_opt_clip",
            "robot_width",
        ),
    )
    def _get_state_only_prior(
        max_p_value: float,
        min_command: float,
        max_command: float,
        state_dim: int,
        state_min_opt_clip: float,
        state_max_opt_clip: float,
        robot_width: float,
        learned_params: Any,
        state: jnp.ndarray,
        descriptor: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

        # Get the polynomial coefficients: a + b*x + c*x**2 + d*x**3
        a = learned_params[2]
        b = learned_params[3]
        c = learned_params[4]
        d = learned_params[5]
        offset = learned_params[6]

        # Compute the polynomial
        clip_state = jnp.clip(
            state[state_dim], a_min=state_min_opt_clip, a_max=state_max_opt_clip
        )
        new_p = a + b * clip_state + c * clip_state**2 + d * clip_state**3
        new_p = jnp.clip(new_p, a_min=-max_p_value, a_max=max_p_value)
        # Get corresponding prior
        p1, p2 = jnp.min(jnp.asarray([1, 1 - new_p])), jnp.min(
            jnp.asarray([1, 1 + new_p])
        )
        diag = (p1 + p2) / 2
        scaling = jnp.asarray(
            [[diag, (p2 - p1) * robot_width / 4], [(p2 - p1) / robot_width, diag]]
        ).T
        inv_s = jnp.linalg.inv(scaling)

        # Final command
        command = jnp.matmul((descriptor - jnp.asarray([offset, 0.0])), inv_s)
        command = jnp.clip(command, a_min=min_command, a_max=max_command)

        return command, descriptor, jnp.asarray([[0.0, 0.0]])

    def get_command(
        self,
        joystick_linear_x_human_command: float,
        joystick_angular_z_human_command: float,
        flipper_angular_x_human_command: float,
        use_state: bool = False,
        use_state_gp: bool = False,
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

        # Scale joystick signal
        x = joystick_linear_x_human_command
        y = joystick_angular_z_human_command
        # x = jnp.clip(x, a_min=-self.max_x, a_max=self.max_x)
        # y = jnp.clip(y, a_min=-self.max_y, a_max=self.max_y)
        cartesian_descriptor = jnp.asarray([[x, y]])

        # If using state prior and GP
        if (
            use_state_gp
            and self.state is not None
            and self.dataset_state is not None
            and self.kernel_x is not None
        ):
            self.all_corrected_descriptors = (
                FunctionalityController._get_state_gp_prior(
                    max_p_value=self.max_p_value,
                    min_command=self.min_command,
                    max_command=self.max_command,
                    state_dim=self.state_dim,
                    state_min_opt_clip=self.state_min_opt_clip,
                    state_max_opt_clip=self.state_max_opt_clip,
                    robot_width=self.robot_width,
                    learned_params=self.learned_params,
                    state=self.state,
                    all_descriptors=self.all_descriptors,
                    cov_alpha=self.cov_alpha,
                    kernel=self.base_kernel,
                    dataset_state=self.dataset_state,
                    Kxx=self.kernel_x,
                )
            )
            (
                action_to_be_executed,
                chosen_descriptor_gp,
                chosen_uncertainties,
            ) = FunctionalityController._get_gp_only_prior(
                self.all_corrected_descriptors,
                self.uncertainties,
                self.all_genotypes,
                cartesian_descriptor,
            )
            action_to_be_executed = jnp.expand_dims(action_to_be_executed, axis=0)

        # If using state prior only
        elif use_state and self.state is not None:
            (
                action_to_be_executed,
                chosen_descriptor_gp,
                chosen_uncertainties,
            ) = FunctionalityController._get_state_only_prior(
                max_p_value=self.max_p_value,
                min_command=self.min_command,
                max_command=self.max_command,
                state_dim=self.state_dim,
                state_min_opt_clip=self.state_min_opt_clip,
                state_max_opt_clip=self.state_max_opt_clip,
                robot_width=self.robot_width,
                learned_params=self.learned_params,
                state=self.state,
                descriptor=cartesian_descriptor,
            )

        # If not using any state prior
        else:
            (
                action_to_be_executed,
                chosen_descriptor_gp,
                chosen_uncertainties,
            ) = FunctionalityController._get_gp_only_prior(
                self.all_corrected_descriptors,
                self.uncertainties,
                self.all_genotypes,
                cartesian_descriptor,
            )
            action_to_be_executed = jnp.expand_dims(action_to_be_executed, axis=0)

        # Concatenate before memory transfert to speed up x10 FC
        results = jnp.concatenate(
            [
                action_to_be_executed.squeeze(axis=0),
                chosen_descriptor_gp.squeeze(axis=0),
                chosen_uncertainties.squeeze(axis=0),
            ],
            axis=0,
        )

        # Return action to be published
        results = np.asarray(results)
        linear_velocity = results[0]
        angular_velocity = results[1]
        descriptor_x = results[2]
        descriptor_z = results[3]
        uncertainty_x = results[4]
        uncertainty_z = results[5]

        return (
            linear_velocity,
            angular_velocity,
            flipper_angular_x_human_command,
            descriptor_x,
            descriptor_z,
            uncertainty_x,
            uncertainty_z,
        )
