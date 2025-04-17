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


class FunctionalityControllerL1:
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

        # Robot physical parameters
        self.mass = 10.0  # kg
        self.inertia = 1.0  # kg*m^2
        self.wheel_radius = 0.1  # m
        self.wheel_distance = 0.5  # m
        

        # Define the system matrices for velocity control
        # Here, A includes a small damping term (e.g., friction)
        # self.A = np.array([[-0.1, 0.0],
        #                    [ 0.0, -0.1]])

        self.A = np.array([[-0.1, 0], 
                    [0, -0.1]])

        self.B = np.array([[1, 0], 
                    [0, 1]])


        # L1 adaptive control parameters
        self.Gamma = 1.0 # Adaptation gain (higher means faster adaptation)
        self.P = np.array([
            [1.0, 0],
            [0, 1.0]
        ])  # Lyapunov equation solution
        
        # Low pass filter coefficients
        self.omega_c = 1.0  # Filter bandwidth
        # We'll compute alpha dynamically: alpha = 1 - exp(-omega_c*delta_t)

        ##Pre-Commputed Version
        self.C = self.omega_c / (self.omega_c + 1)
        
        # Saturation bound for uncertainty estimate for numerical stability
        self.max_sigma = 100.0
        
        # State variables
        self.state = np.zeros(2)  # Current state [v, ω]
        self.x_ref = np.zeros(2)  # Reference state
        self.x_hat = np.zeros(2)  # State estimate
        self.sigma_hat = np.zeros(2)  # Uncertainty estimate
        self.u = np.zeros(2)  # Control input [linear_cmd, angular_cmd]
        self.u_ad = np.zeros(2)  # Adaptive control input
        
        # Robot pose [x, y, θ]
        self.pose = np.zeros(3)
        

        # Create all adaptation objects
        self.reset()

    def reset(self):
        """Reset the controller states"""
        self.state = np.zeros(2)
        self.x_hat = np.zeros(2)
        self.sigma_hat = np.zeros(2)
        self.u = np.zeros(2)
        self.u_ad = np.zeros(2)
        self.pose = np.zeros(3)

    def update_reference(self, v_ref, omega_ref):
        """Update reference velocities"""
        self.x_ref = np.array([v_ref, omega_ref])

    def update_state(self,v,omega):
        """Update current state velocities"""
        self.state = np.array([v,omega])

    def state_predictor(self):
        """
        Update the state predictor (state estimator)
        """
        # Nominal dynamics
        x_dot_nominal = np.dot(self.A, self.x_hat) + np.dot(self.B, self.u)
        
        # Add estimated uncertainties
        x_dot_pred = x_dot_nominal + np.dot(self.B, self.sigma_hat)
        
        # Euler integration
        # self.x_hat += x_dot_pred * self.delta_t
        self.x_hat = x_dot_pred
    
    def adaptation_law(self):
        """
        Update the uncertainty estimate using adaptive law
        """
        # Error between predicted and actual state
        error = self.state - self.x_hat
        
        # Update sigma_hat based on adaptation law
        sigma_dot = self.Gamma * np.dot(self.P, error)
        # self.sigma_hat += sigma_dot * self.delta_t
        self.sigma_hat = sigma_dot

        self.sigma_hat = np.clip(self.sigma_hat, -self.max_sigma, self.max_sigma)
    
    def control_law(self):
        """
        Compute the L1 adaptive control input
        """
        # Compute reference control input
        u_ref = np.dot(np.linalg.inv(self.B), self.x_ref - np.dot(self.A, (self.x_hat)))
        # u_ref = self.x_ref
        
        # Compute adaptive control to cancel uncertainties
        self.u_ad = -self.C * self.sigma_hat

        ## Adaptive Filtering
        # Compute the filter coefficient dynamically
        # alpha = 1 - np.exp(-self.omega_c * self.delta_t)
        # # Exponential smoothing filter on adaptive control input:
        # self.u_ad = (1 - alpha) * self.u_ad - alpha * self.sigma_hat
        
        # Total control input
        self.u = u_ref + self.u_ad

        ## Clipping
        self.u = np.clip(self.u, self.min_command, self.max_command)
        
        return self.u
    
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

        # Define the L1 control law for velocity tracking
        if state is None or sensor_x is None or sensor_y is None:
            return (
                joystick_linear_x_human_command, 
                joystick_angular_z_human_command, 
                flipper_angular_x_human_command,
            )

        self.state_predictor()
        self.update_state(sensor_x,sensor_y)
        self.adaptation_law()
        self.update_reference(x,y)
        self.control_law()

        linear_velocity, angular_velocity = self.u

        return (
            linear_velocity,
            angular_velocity,
            flipper_angular_x_human_command,
        )
