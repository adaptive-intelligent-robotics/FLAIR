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
import numpy as np
from scipy.linalg import solve_continuous_are


class FunctionalityControllerLQR:
    """Main FunctionalityController class, call at each lopp."""

    def __init__(
        self,
        min_command: int,
        max_command: int,
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
        self.delta_t = 0.1
        self.z_v = 0
        self.z_omega = 0

        # Define the system matrices for velocity control
        # self.A = np.array([[0, 0], 
        #             [0, 0]])
        self.A = np.array([[0.0, 0], 
                    [0, 0.0]])

        self.B = np.array([[1, 0], 
                    [0, 1]])


        self.A_aug = np.array([
            [0, 0, 0, 0],    # Dynamics for linear velocity error
            [0, 0, 0, 0],    # Dynamics for angular velocity error
            [1, 0, 0, 0],    # Accumulated linear velocity error depends on linear error
            [0, 1, 0, 0]     # Accumulated angular velocity error depends on angular error
        ])

        # self.A_aug = np.block([[self.A, np.zeros((2, 2))],
        #               [np.asarray([[1, 0], [0, 1]]), np.zeros((2, 2))]])

        self.B_aug = np.array([
            [1, 0],          # control input for linear velocity
            [0, 1],          # control input for angular velocity
            [0, 0],          # no direct control impact on integral states
            [0, 0]
        ])

        # Define cost matrices
        q_v = 0.3      # Penalty for linear velocity error
        q_omega = 0.08 # Penalty for angular velocity error
        self.Q = np.diag([q_v, q_omega])

        r_v = 1.0 #0.5     # Penalty for control effort (linear acceleration)
        r_omega = 1.0# 0.2 # Penalty for control effort (angular acceleration)
        self.R = np.diag([r_v, r_omega])

        q_z_v = 0.1        # Penalty for accumulated linear velocity error
        q_z_omega = 0.1    # Penalty for accumulated angular velocity error
        self.Q_aug = np.diag([q_v, q_omega, q_z_v, q_z_omega])
        
        # # # # Solve the continuous-time algebraic Riccati equation for the augmented system
        # self.P_aug = solve_continuous_are(self.A_aug, self.B_aug, self.Q_aug, self.R)

        # # # # # Compute the LQI gain matrix
        # self.K = np.linalg.inv(self.R) @ self.B_aug.T @ self.P_aug

        # Solve the continuous-time algebraic Riccati equation
        self.P = solve_continuous_are(self.A, self.B, self.Q, self.R)

        # Compute the LQR gain matrix
        self.K = np.linalg.inv(self.R) @ self.B.T @ self.P

        # Create all adaptation objects
        self.reset()

    def reset(self):
        """Reset the FC."""
        pass


    def lqr_velocity_control(self,v, omega, v_ref, omega_ref):
        """
        Compute the LQR control input for velocity tracking.
        
        :param v: Current linear velocity
        :param omega: Current angular velocity
        :param v_ref: Desired linear velocity
        :param omega_ref: Desired angular velocity
        :return: Control input [u_v, u_omega] (linear and angular accelerations)
        """
        # Current state (velocities)

        x = np.array([v, omega])
        
        # Reference state (desired velocities)
        x_ref = np.array([v_ref, omega_ref])
        
        # State error
        x_error = x - x_ref
        
        #  # Integrate errors over time
        self.z_v += x_error[0] * self.delta_t
        self.z_omega += x_error[1] * self.delta_t

        # x_error_aug = np.array([x_error[0], x_error[1], self.z_v, self.z_omega])

        # LQR control law
        u = x_ref-(self.K @ x_error)#x_error
        u = np.clip(u, a_min=self.min_command, a_max=self.max_command)

        return u

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

        # Scale joystick signal
        x = joystick_linear_x_human_command
        y = joystick_angular_z_human_command

        # Define the LQR control law for velocity tracking
        if state is None or sensor_x is None or sensor_y is None:
            return (
                joystick_linear_x_human_command, 
                joystick_angular_z_human_command, 
                flipper_angular_x_human_command,
            )
        linear_velocity, angular_velocity = self.lqr_velocity_control(sensor_x,sensor_y, x, y)
        return (
            linear_velocity,
            angular_velocity,
            flipper_angular_x_human_command,
        )
