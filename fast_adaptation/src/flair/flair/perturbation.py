from typing import Any, Tuple

import time
import numpy as np
from FLAIR_msg.msg import FunctionalityControllerControl, Perturbation


class PerturbationTransform:
    def __init__(
        self,
        wheel_base: float,
        wheel_radius: float,
        wheel_max_velocity: float,
        logger: Any,
    ):

        self._logger = logger
        self.wheel_base = wheel_base
        self.wheel_radius = wheel_radius
        self.wheel_max_velocity = wheel_max_velocity
        self._logger.info("Perturbation: Starting")

        # Initialise perturbation values
        self.dynamic_scaling = False
        self.track_scaling = False
        self.track_offset = False
        self.wind = False
        self.bernoulli = False

        # Initialise all needed attributes
        self.init_pos = [0.0, 0.0]
        self.init_angle = 0.0
        self.left_track_scaling = 1.0
        self.right_track_scaling = 1.0
        self.left_track_offset = 0.0
        self.right_track_offset = 0.0
        self.dynamic_scale_interval = 1.0
        self.dynamic_scale_amplitude = 1.0
        self.dynamic_current_left_track_scaling = 1.0
        self.dynamic_current_right_track_scaling = 1.0
        self.dynamic_change_time = time.time()

        self._logger.info("Perturbation: Completed __init__")

    def new_perturbation(self, msg: Perturbation, current_state: Tuple) -> None:
        """Update the perturbation to apply."""

        if current_state is None:
            # self._logger.debug("Current state is None, no damage.")
            return

        # Update state
        self.position = [current_state[3], current_state[4]]  # tx, ty
        self.angle = current_state[5]  # yaw

        # Detect scaling change
        new_left_track_scaling = np.clip(msg.left_scale, 0.5, 1)
        new_right_track_scaling = np.clip(msg.right_scale, 0.5, 1)
        if (
            self.left_track_scaling != new_left_track_scaling
            or self.right_track_scaling != new_right_track_scaling
        ):

            self.left_track_scaling = new_left_track_scaling
            self.right_track_scaling = new_right_track_scaling
            if self.left_track_scaling != 1 or self.right_track_scaling != 1:
                self._logger.info(f"Starting scaling perturbation.")
                self.track_scaling = True
            else:
                self._logger.info(f"Stopping scaling perturbation.")
                self.track_scaling = False

        # Detect offset change
        new_left_track_offset = np.clip(msg.left_offset, -2.0, 2.0)
        new_right_track_offset = np.clip(msg.right_offset, -2.0, 2.0)
        if (
            self.left_track_offset != new_left_track_offset
            or self.right_track_offset != new_right_track_offset
        ):

            self.left_track_offset = new_left_track_offset
            self.right_track_offset = new_right_track_offset
            if self.left_track_offset != 0 or self.right_track_offset != 0:
                self._logger.info(f"Starting offset perturbation.")
                self.track_offset = True
            else:
                self._logger.info(f"Stopping offset perturbation.")
                self.track_offset = False

        # Detect dynamic scaling change
        new_dynamic_scale_amplitude = np.clip(msg.dynamic_scale_amplitude, 0.5, 1)
        if (
            self.dynamic_scale_amplitude != new_dynamic_scale_amplitude
            or msg.dynamic_scale_interval != self.dynamic_scale_interval
        ):
            self.dynamic_scale_amplitude = new_dynamic_scale_amplitude
            self.dynamic_scale_interval = msg.dynamic_scale_interval
            if self.dynamic_scale_amplitude != 1:
                self._logger.info(f"Starting dynamic scaling perturbation.")
                self.dynamic_scaling = True
                self.dynamic_current_left_track_scaling = 1.0
                self.dynamic_current_right_track_scaling = self.dynamic_scale_amplitude
                self.dynamic_change_time = time.time()
            else:
                self._logger.info(f"Stopping dynamic scaling perturbation.")
                self.dynamic_scaling = False
                self.dynamic_current_left_track_scaling = 1.0
                self.dynamic_current_right_track_scaling = 1.0
                self.dynamic_change_time = time.time()
            

        # Detect bernoulli change
        if self.bernoulli != msg.bernoulli:
            if msg.bernoulli:
                self._logger.info(f"Starting bernoulli perturbation.")
                self.init_pos = self.position
                self.init_angle = self.angle
            else:
                self._logger.info(f"Stopping bernoulli perturbation.")
        self.bernoulli = msg.bernoulli

        # Detect wind change
        if self.wind != msg.wind:
            if msg.wind:
                self._logger.info(f"Starting wind perturbation.")
            else:
                self._logger.info(f"Stopping wind perturbation.")
        self.wind = msg.wind

    def apply_perturbation(
        self, msg: FunctionalityControllerControl, current_state: Tuple
    ) -> FunctionalityControllerControl:
        """Apply the corruption due to any perturbation."""

        if current_state is None:
            # self._logger.debug("Current state is None, no damage.")
            return msg

        # Update state
        self.position = [current_state[3], current_state[4]]  # tx, ty
        self.angle = current_state[5]  # yaw

        # Apply perturbation
        if self.bernoulli:
            msg = self._corrupt_state_bernoulli(msg)
        elif self.wind:
            msg = self._corrupt_wind(msg)
        elif self.track_scaling or self.track_offset:
            msg = self._corrupt_state(msg)
        elif self.dynamic_scaling:
            msg = self._dynamic_corrupt_state(msg)
        return msg

    def _corrupt_state(
        self, msg: FunctionalityControllerControl
    ) -> FunctionalityControllerControl:
        """Apply scalling and offset perturbation."""
        lin = msg.joystick.linear.x
        ang = -msg.joystick.angular.z

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

        if self.left_track_offset == self.right_track_offset:
            linear_vel_corrupted += self.left_track_offset

        msg.joystick.linear.x = linear_vel_corrupted
        msg.joystick.angular.z = -angular_vel_corrupted

        return msg

    def _dynamic_corrupt_state(
        self, msg: FunctionalityControllerControl
    ) -> FunctionalityControllerControl:
        """Apply dynamic scalling perturbation."""

        # Update dynamic damage using interval
        current_time = time.time()
        if current_time - self.dynamic_change_time > self.dynamic_scale_interval:
            if self.dynamic_current_right_track_scaling == self.dynamic_scale_amplitude:
                self.dynamic_current_left_track_scaling = self.dynamic_scale_amplitude
                self.dynamic_current_right_track_scaling = 1.0
                self._logger.info(f"Changing dynamic scaling perturbation to left side.")
            else:
                self.dynamic_current_left_track_scaling = 1.0
                self.dynamic_current_right_track_scaling = self.dynamic_scale_amplitude
                self._logger.info(f"Changing dynamic scaling perturbation to right side.")
            self.dynamic_change_time = time.time()

        # Apply damage
        lin = msg.joystick.linear.x
        ang = -msg.joystick.angular.z

        left_vel = (lin - ang * self.wheel_base / 2) / self.wheel_radius
        right_vel = (lin + ang * self.wheel_base / 2) / self.wheel_radius

        left_vel *= self.dynamic_current_left_track_scaling
        right_vel *= self.dynamic_current_right_track_scaling

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

        msg.joystick.linear.x = linear_vel_corrupted
        msg.joystick.angular.z = -angular_vel_corrupted

        return msg

    def get_scaling(self, pos, sandia=True):
        left_vel_scaling = 1.0
        right_vel_scaling = 1.0

        if not sandia:
            val = ((1 - pos**2) - 0.8164) / (0.9964 - 0.8164) * 0.4 + 0.5
            val = np.clip(val, 0.5, 1.0)
            if pos > 0.05:
                left_vel_scaling = val
                right_vel_scaling = 1
                self._logger.debug(">0.1")
            elif pos < -0.05:
                right_vel_scaling = val
                left_vel_scaling = 1
                self._logger.debug("<0.1")
            return left_vel_scaling, right_vel_scaling
        else:
            bernoulli_scale = 2.5  # 1.0
            robot_length = 280  # [mm] Length of robot in Z direction
            robot_height = 100  # [mm] Height from the
            robot_width = 280  # [mm] Width of robot in X direction
            treadmill_width_x = 1190  # [mm] edge of treadmill to center of treamill,
            x_bound = 50  # [mm] invisible boundary layer
            pos_bound = x_bound
            neg_bound = -1 * x_bound
            center = treadmill_width_x / 2
            max_force = (
                50  # Max Motor influence from force we will allow (can go up to 100)
            )

            self._logger.debug(f"POS, {-pos}, {type(pos)}")
            dist = -pos * 1000  # negating due to Peraton's robot conventions
            dist_2 = round(center - abs(dist) - (robot_width / 2)) + 25
            if dist_2 < 0:
                self._logger.debug(
                    f"Robot over Edge. If not, then calibration or param variables are wrong {pos}"
                )

            force = self.forcecalc(dist_2, robot_length, robot_height, bernoulli_scale)

            # Checking force value and setting it to max safety limit set from observations
            force = min(force, max_force)
            # Boundary Layers dependent on X values
            if neg_bound < dist < pos_bound:
                # 0 enviromental effect zone when centroid is within bounds
                left_vel_scaling = 1.0
                right_vel_scaling = 1.0

            # Future Additions?:
            # Yaw can be incorporated to dictate a factor by which edge drag force grows.

            elif pos_bound <= dist or dist <= neg_bound:
                # Robot outside of mainstream on left
                if dist <= neg_bound:
                    left_vel_scaling = 1.0 - (force / 100.0)
                    right_vel_scaling = 1.0
                # Robot outside of mainstream on right
                elif dist >= pos_bound:
                    left_vel_scaling = 1.0
                    right_vel_scaling = 1.0 - (force / 100.0)
            return left_vel_scaling, right_vel_scaling

    def _corrupt_state_bernoulli(
        self, msg: FunctionalityControllerControl
    ) -> FunctionalityControllerControl:
        """Apply bernoulli perturbation."""
        lin = msg.joystick.linear.x
        ang = -msg.joystick.angular.z

        left_vel = (lin - ang * self.wheel_base / 2) / self.wheel_radius
        right_vel = (lin + ang * self.wheel_base / 2) / self.wheel_radius

        # Force ~ -K * [v_bot * [(x + x_0)/(x_0) - 1]]^2
        x = self.position[1]
        if not (np.isfinite(x)):
            self._logger.warning(f"Got nan as position. Using 0.0001.")
            x = 0.0001
        left_vel_scaling, right_vel_scaling = self.get_scaling(x)

        left_vel *= left_vel_scaling
        right_vel *= right_vel_scaling

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

        msg.joystick.linear.x = linear_vel_corrupted
        msg.joystick.angular.z = -angular_vel_corrupted

        return msg

    def _corrupt_wind(
        self, msg: FunctionalityControllerControl
    ) -> FunctionalityControllerControl:
        point = 0.0
        error = point - self.angle

        error *= 0.6/np.pi

        perturbation = np.clip(abs(error), 0, 0.6)

        if error < 0:
            old_perturbation = self.left_track_scaling
            self.left_track_scaling = 1.0 - perturbation
        else:
            old_perturbation = self.right_track_scaling
            self.right_track_scaling = 1.0 - perturbation

        msg = self._corrupt_state(msg)
        if error < 0:
            self.left_track_scaling = old_perturbation
        else:
            self.right_track_scaling = old_perturbation

        return msg
