import numpy as np
from scipy.linalg import solve_continuous_are

# Define the system matrices for velocity control
A = np.array([[0, 0], 
              [0, 0]])

B = np.array([[1, 0], 
              [0, 1]])

# Define cost matrices
q_v = 1.0      # Penalty for linear velocity error
q_omega = 1.0  # Penalty for angular velocity error
Q = np.diag([q_v, q_omega])

r_v = 0.01     # Penalty for control effort (linear acceleration)
r_omega = 0.01 # Penalty for control effort (angular acceleration)
R = np.diag([r_v, r_omega])

# Solve the continuous-time algebraic Riccati equation
P = solve_continuous_are(A, B, Q, R)

# Compute the LQR gain matrix
K = np.linalg.inv(R) @ B.T @ P

# Define the LQR control law for velocity tracking
def lqr_velocity_control(v, omega, v_ref, omega_ref):
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
    
    # LQR control law
    u = -K @ x_error
    return u

# Example usage
v = 1.0          # Current linear velocity (m/s)
omega = 0.1      # Current angular velocity (rad/s)
v_ref = 2.0      # Desired linear velocity (m/s)
omega_ref = 0.0  # Desired angular velocity (rad/s)

# Compute the control input (accelerations)
u_v, u_omega = lqr_velocity_control(v, omega, v_ref, omega_ref)
print("Control input (u_v, u_omega):", u_v, u_omega)
