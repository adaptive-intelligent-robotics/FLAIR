from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
import brax.v1 as brax
from brax.envs.base import Env, State, Wrapper

from qdax.environments.base_wrappers import QDEnv

# name of the center of gravity
COG_NAMES = {
    "hexapod_angle_diff": "base_link",
    "hexapod_no_reward": "base_link",
    "hexapod_control": "base_link",
}

VELOCITY_BOUNDS = {
    "hexapod_angle_diff": (jnp.array([-5.0, -5.0]), jnp.array([5.0, 5.0])),
    "hexapod_no_reward": (jnp.array([-5.0, -5.0]), jnp.array([5.0, 5.0])),
    "hexapod_control": (jnp.array([-5.0, -5.0]), jnp.array([5.0, 5.0])),
}

class XYawVelocityWrapper(QDEnv):
    def __init__(
        self,
        env: Env,
        env_name: str,
        minval: Optional[List[float]] = None,
        maxval: Optional[List[float]] = None,
    ):
        if env_name not in COG_NAMES.keys():
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")

        super().__init__(config=None)

        self.env = env
        self._env_name = env_name
        if hasattr(self.env, "sys"):
            self._cog_idx = self.env.sys.body.index[COG_NAMES[env_name]]
            self._bounds = VELOCITY_BOUNDS[env_name]
            self._dim = self._bounds[0].size
        else:
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")

        if minval is None:
            minval = jnp.ones((2,)) * (-jnp.inf)

        if maxval is None:
            maxval = jnp.ones((2,)) * jnp.inf

        if len(minval) == 2 and len(maxval) == 2:
            self._minval = jnp.array(minval)
            self._maxval = jnp.array(maxval)
        else:
            raise NotImplementedError(
                "Please make sure to give two values for each limits."
            )

    @property
    def state_descriptor_length(self) -> int:
        return 2

    @property
    def state_descriptor_name(self) -> str:
        return "xy_velocity"

    @property
    def state_descriptor_limits(self) -> Tuple[List[float], List[float]]:
        return self._minval, self._maxval

    @property
    def behavior_descriptor_length(self) -> int:
        return self.state_descriptor_length

    @property
    def behavior_descriptor_limits(self) -> Tuple[List[float], List[float]]:
        return self.state_descriptor_limits

    @property
    def name(self) -> str:
        return self._env_name

    def get_state_descriptor(self, state: State) -> jnp.ndarray:

        # Get the linear velocity (in world frame)
        linear_velocity_world = state.qp.vel[self._cog_idx]

        # Get the angular velocity (in world frame)
        angular_velocity_world = state.qp.ang[self._cog_idx]

        # Get the rotation (quaternion) of the robot's body in the world frame
        rot_world_to_body = state.qp.rot[self._cog_idx]

        # Inverse of the rotation quaternion (to convert from world to body frame)
        rot_world_to_body_inv = brax.math.quat_inv(rot_world_to_body)

        # Transform the linear velocity from the world frame to the body frame
        linear_velocity_body = brax.math.rotate(linear_velocity_world, rot_world_to_body_inv)

        # Transform the angular velocity from the world frame to the body frame
        angular_velocity_body = brax.math.rotate(angular_velocity_world, rot_world_to_body_inv)

        return jnp.concatenate([linear_velocity_body[0:1], angular_velocity_body[2:3]], axis=0)


    def reset(self, rng: jnp.ndarray) -> State:
        state = self.env.reset(rng)
        state.info["state_descriptor"] = self.get_state_descriptor(state)
        return state

    def step(self, state: State, action: jnp.ndarray) -> State:
        state = self.env.step(state, action)
        state.info["state_descriptor"] = self.get_state_descriptor(state)
        return state

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped

    def __getattr__(self, name: str) -> Any:
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)

