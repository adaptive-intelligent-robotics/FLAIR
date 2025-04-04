import functools
from typing import Any, Callable, List, Optional, Union

from brax.v1.envs import Env, _envs
from brax.v1.envs.wrappers import (
    AutoResetWrapper,
    EpisodeWrapper,
    EvalWrapper,
    VectorWrapper,
)
from qdax.environments.base_wrappers import QDEnv, StateDescriptorResetWrapper
from qdax.environments.bd_extractors import (
    get_feet_contact_proportion,
    get_final_xy_position,
)
from qdax.environments.exploration_wrappers import MazeWrapper, TrapWrapper
from qdax.environments.humanoidtrap import HumanoidTrap
from qdax.environments.init_state_wrapper import FixedInitialStateWrapper
from qdax.environments.locomotion_wrappers import (
    FeetContactWrapper,
    NoForwardRewardWrapper,
    XYPositionWrapper,
)
from qdax.environments.pointmaze import PointMaze
from qdax.environments.wrappers import CompletedEvalWrapper
from brax_environments.hexapod import Hexapod, HexapodAngleDiff, HexapodControl
from brax_environments.locomotion_wrappers import XYVelocityWrapper
from brax_environments.bd_extractors import get_velocity

# experimentally determinated offset
qd_offset = {
    "hexapod_velocity": 750,
    "hexapod_no_reward_velocity": 0.0,
    "hexapod_control_reward_velocity": 0.0,
}

behavior_descriptor_extractor = {
    "hexapod_velocity": get_velocity,
    "hexapod_no_reward_velocity": get_velocity,
    "hexapod_control_reward_velocity": get_velocity,
}

_qdax_custom_envs = {
    "hexapod_velocity": {
        "env": "hexapod_angle_diff",
        "wrappers": [XYVelocityWrapper],
        "kwargs": [{"minval": [-0.25, -0.25], "maxval": [0.25, 0.25]}, {}],
    },
    "hexapod_no_reward_velocity": {
        "env": "hexapod_no_reward",
        "wrappers": [XYVelocityWrapper],
        "kwargs": [{"minval": [-0.25, -0.25], "maxval": [0.25, 0.25]}, {}],
    },
    "hexapod_control_reward_velocity": {
        "env": "hexapod_control",
        "wrappers": [XYVelocityWrapper],
        "kwargs": [{"minval": [-0.25, -0.25], "maxval": [0.25, 0.25]}, {}],
    },
}


def create(
    env_name: str,
    episode_length: int = 1000,
    action_repeat: int = 1,
    auto_reset: bool = True,
    batch_size: Optional[int] = None,
    eval_metrics: bool = False,
    fixed_init_state: bool = False,
    qdax_wrappers_kwargs: Optional[List] = None,
    **kwargs: Any,
) -> Union[Env, QDEnv]:
    """Creates an Env with a specified brax system.
    Please use namespace to avoid confusion between this function and
    brax.envs.create.
    """
    print("    Using legacy_spring=False.")

    if env_name == "ant":
        env = _envs[env_name](legacy_spring=False, use_contact_forces=False, **kwargs)
    elif env_name in _envs.keys():
        env = _envs[env_name](legacy_spring=False, **kwargs)
    elif env_name in _qdax_custom_envs.keys():
        base_env_name = _qdax_custom_envs[env_name]["env"]
        if base_env_name == "ant":
            env = _envs[base_env_name](legacy_spring=False, use_contact_forces=False, **kwargs)
        elif base_env_name == "hexapod_no_reward":
            print("    Using local hexapod")
            env = Hexapod(**kwargs)
        elif base_env_name == "hexapod_angle_diff":
            print("    Using local hexapod")
            env = HexapodAngleDiff(**kwargs)
        elif base_env_name == "hexapod_control":
            print("    Using local hexapod")
            env = HexapodControl(**kwargs)
        elif base_env_name in _envs.keys():
            env = _envs[base_env_name](legacy_spring=False, **kwargs)
    else:
        raise NotImplementedError("This environment name does not exist!")

    if env_name in _qdax_custom_envs.keys():
        # roll with qdax wrappers
        wrappers = _qdax_custom_envs[env_name]["wrappers"]
        if qdax_wrappers_kwargs is None:
            kwargs_list = _qdax_custom_envs[env_name]["kwargs"]
        else:
            kwargs_list = qdax_wrappers_kwargs
        for wrapper, kwargs in zip(wrappers, kwargs_list):  # type: ignore
            env = wrapper(env, base_env_name, **kwargs)  # type: ignore

    if episode_length is not None:
        env = EpisodeWrapper(env, episode_length, action_repeat)
    if batch_size:
        env = VectorWrapper(env, batch_size)
    if fixed_init_state:
        # retrieve the base env
        if env_name not in _qdax_custom_envs.keys():
            base_env_name = env_name
        # wrap the env
        env = FixedInitialStateWrapper(env, base_env_name=base_env_name)  # type: ignore
    if auto_reset:
        env = AutoResetWrapper(env)
        if env_name in _qdax_custom_envs.keys():
            env = StateDescriptorResetWrapper(env)
    if eval_metrics:
        env = EvalWrapper(env)
        env = CompletedEvalWrapper(env)

    return env


def create_fn(env_name: str, **kwargs: Any) -> Callable[..., Env]:
    """Returns a function that when called, creates an Env.
    Please use namespace to avoid confusion between this function and
    brax.envs.create_fn.
    """
    return functools.partial(create, env_name, **kwargs)
