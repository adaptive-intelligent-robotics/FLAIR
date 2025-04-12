from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from qdax.core.neuroevolution.buffers.buffer import (
    QDTransition,
    Transition,
)
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.types import (
    Descriptor,
    EnvState,
    ExtraScores,
    Fitness,
    Genotype,
    Observation,
    Params,
    RNGKey,
)

import brax_environments


@partial(jax.jit, static_argnames=("play_step_fn", "episode_length"))
def generate_unroll_time(
    init_state: EnvState,
    policy_params: Params,
    random_key: RNGKey,
    episode_length: int,
    play_step_fn: Callable[
        [EnvState, Params, RNGKey, int],
        Tuple[
            EnvState,
            Params,
            RNGKey,
            Transition,
            int,
        ],
    ],
) -> Tuple[EnvState, Transition]:
    """Generates an episode according to the agent's policy, returns the final state of
    the episode and the transitions of the episode.

    Args:
        init_state: first state of the rollout.
        policy_params: params of the individual.
        random_key: random key for stochasiticity handling.
        episode_length: length of the rollout.
        play_step_fn: function describing how a step need to be taken.

    Returns:
        A new state, the experienced transition.
    """

    def _scan_play_step_fn(
        carry: Tuple[EnvState, Params, RNGKey, int], unused_arg: Any
    ) -> Tuple[Tuple[EnvState, Params, RNGKey, int], Transition]:
        env_state, policy_params, random_key, transitions, timestep = play_step_fn(
            *carry
        )
        return (env_state, policy_params, random_key, timestep), transitions

    (state, _, _, _), transitions = jax.lax.scan(
        _scan_play_step_fn,
        (init_state, policy_params, random_key, 0),  # 0 is the time step
        (),
        length=episode_length,
    )
    return state, transitions


@partial(
    jax.jit,
    static_argnames=(
        "episode_length",
        "play_step_fn",
        "behavior_descriptor_extractor",
    ),
)
def scoring_function_time_brax_envs(
    policies_params: Genotype,
    random_key: RNGKey,
    init_states: EnvState,
    episode_length: int,
    play_step_fn: Callable[
        [EnvState, Params, RNGKey, EnvState],
        Tuple[EnvState, Params, RNGKey, QDTransition],
    ],
    behavior_descriptor_extractor: Callable[[QDTransition, jnp.ndarray], Descriptor],
    fit_std: float = 0.0,
    desc_std: float = 0.0,
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    """Evaluates policies contained in policies_params in parallel in
    stochastic environments.
    """

    # Perform rollouts with each policy
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(
        subkey, jax.tree_util.tree_leaves(policies_params)[0].shape[0]
    )
    unroll_fn = partial(
        generate_unroll_time,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
    )

    _final_state, data = jax.vmap(unroll_fn)(init_states, policies_params, keys)

    # create a mask to extract data properly
    is_done = jnp.clip(jnp.cumsum(data.dones, axis=1), 0, 1)
    mask = jnp.roll(is_done, 1, axis=1)
    mask = mask.at[:, 0].set(0)

    # Scores - add offset to ensure positive fitness (through positive rewards)
    fitnesses = jnp.sum(data.rewards * (1.0 - mask), axis=1)
    descriptors = behavior_descriptor_extractor(data, mask)

    # Add noise to the fitnesses and descriptors
    random_key, f_subkey, d_subkey = jax.random.split(random_key, num=3)
    fitnesses = fitnesses + jax.random.normal(f_subkey, shape=fitnesses.shape) * fit_std
    descriptors = (
        descriptors + jax.random.normal(d_subkey, shape=descriptors.shape) * desc_std
    )

    return (
        fitnesses,
        descriptors,
        {
            "transitions": data,
        },
        random_key,
    )


@partial(
    jax.jit,
    static_argnames=(
        "episode_length",
        "play_reset_fn",
        "play_step_fn",
        "behavior_descriptor_extractor",
    ),
)
def reset_based_scoring_function_time_brax_envs(
    policies_params: Genotype,
    random_key: RNGKey,
    episode_length: int,
    play_reset_fn: Callable[[RNGKey], EnvState],
    play_step_fn: Callable[
        [EnvState, Params, RNGKey, Any],
        Tuple[EnvState, Params, RNGKey, QDTransition],
    ],
    behavior_descriptor_extractor: Callable[[QDTransition, jnp.ndarray], Descriptor],
    fit_std: float = 0.0,
    desc_std: float = 0.0,
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    """Evaluates policies contained in policies_params in parallel.
    The play_reset_fn function allows for a more general scoring_function that can be
    called with different batch-size and not only with a batch-size of the same
    dimension as init_states.

    To define purely stochastic environments, using the reset function from the
    environment, use "play_reset_fn = env.reset".

    To define purely deterministic environments, as in "scoring_function", generate
    a single init_state using "init_state = env.reset(random_key)", then use
    "play_reset_fn = lambda random_key: init_state".
    """

    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(
        subkey, jax.tree_util.tree_leaves(policies_params)[0].shape[0]
    )
    reset_fn = jax.vmap(play_reset_fn)
    init_states = reset_fn(keys)

    fitnesses, descriptors, extra_scores, random_key = scoring_function_time_brax_envs(
        policies_params=policies_params,
        random_key=random_key,
        init_states=init_states,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=behavior_descriptor_extractor,
        fit_std=fit_std,
        desc_std=desc_std,
    )

    return (fitnesses, descriptors, extra_scores, random_key)


def set_up_hexapod(
    env_name: str,
    episode_length: int,
    batch_size: int,
    random_key: RNGKey,
    damage:bool = False,
) -> Tuple:

    # Init environment
    dim_control = 24
    env = brax_environments.create(
        env_name,
        episode_length=episode_length,
    )

    # Define the fonction to infer the next action
    def simple_sine_controller(
        amplitude: jnp.ndarray, phase: jnp.ndarray, t: int
    ) -> jnp.ndarray:
        return amplitude * jnp.sin(
            (2 * t * jnp.pi / 50) + phase * jnp.pi
        )  # in degrees for brax

    def inference(params: Genotype, state: EnvState, timestep: int) -> jnp.ndarray:
        amplitudes_top = params.at[jnp.asarray([0, 1, 2, 3, 4, 5])].get()
        phases_top = params.at[jnp.asarray([6, 7, 8, 9, 10, 11])].get()
        amplitudes_bottom = params.at[jnp.asarray([12, 13, 14, 15, 16, 17])].get()
        phases_bottom = params.at[jnp.asarray([18, 19, 20, 21, 22, 23])].get()
        top_actions = simple_sine_controller(amplitudes_top, phases_top, timestep)
        bottom_actions = simple_sine_controller(
            amplitudes_bottom, phases_bottom, timestep
        )

        actions = jnp.zeros(shape=(18,))
        actions = actions.at[jnp.asarray([0, 3, 6, 9, 12, 15])].set(
            top_actions * (jnp.pi / 8) * (180 / jnp.pi)
        )
        actions = actions.at[jnp.asarray([1, 4, 7, 10, 13, 16])].set(
            bottom_actions * (jnp.pi / 4) * (180 / jnp.pi)
        )
        actions = actions.at[jnp.asarray([2, 5, 8, 11, 14, 17])].set(
            -bottom_actions * (jnp.pi / 4) * (180 / jnp.pi)
        )
        if damage:
            actions = actions.at[jnp.asarray([3, 4, 5])].set(jnp.pi)
        return actions

    inference_fn = jax.jit(inference)

    # Init policy structure
    class PolicyStructure(jnp.ndarray):
        @staticmethod
        def apply(params: Genotype, state: EnvState, timestep: int) -> jnp.ndarray:
            return inference_fn(params, state, timestep)

    # Init population of controllers
    def init_policies_fn(size: int, random_key: RNGKey) -> Tuple[jnp.ndarray, RNGKey]:
        random_key, subkey = jax.random.split(random_key)
        init_policies = jax.random.uniform(
            random_key, shape=(size, dim_control), minval=-1, maxval=1
        )
        return init_policies, random_key

    # Define the fonction to play a step with the policy in the environment
    def control_play_step_fn(
        env_state: EnvState,
        policy_params: Genotype,
        random_key: RNGKey,
        timestep: int,
        env: Any,
    ) -> Tuple[EnvState, Genotype, RNGKey, Transition, int]:
        """
        Play an environment step and return the updated state and the transition.
        """

        actions = inference_fn(policy_params, env_state, timestep)
        next_state = env.step(env_state, actions)

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            actions=actions,
            truncations=next_state.info["truncation"],
            state_desc=env_state.info["state_descriptor"],
            next_state_desc=next_state.info["state_descriptor"],
        )

        timestep += 1
        return next_state, policy_params, random_key, transition, timestep

    play_step_fn = partial(
        control_play_step_fn,
        env=env,
    )


    # Prepare the scoring function
    bd_extraction_fn = brax_environments.behavior_descriptor_extractor[env_name]

    # Define the function to stochastically reset the environment
    play_reset_fn = partial(env.reset)

    # Use stochastic scoring function
    scoring_fn = partial(
        reset_based_scoring_function_time_brax_envs,
        episode_length=episode_length,
        play_reset_fn=play_reset_fn,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )

    # Get minimum reward value to make sure qd_score are positive
    qd_offset = brax_environments.qd_offset[env_name]

    # Get min and max bd
    min_bd, max_bd = env.behavior_descriptor_limits

    # Return of neuroevolution env
    return (
        env,
        scoring_fn,
        init_policies_fn,
        PolicyStructure,
        min_bd,
        max_bd,
        qd_offset,
        random_key,
    )
