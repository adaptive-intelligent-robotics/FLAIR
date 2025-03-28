from __future__ import annotations

import flax.struct
import jax
import jax.numpy as jnp

from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.custom_types import Descriptor, Params

def get_velocity(data: QDTransition, mask: jnp.ndarray) -> Descriptor:
    """Compute velocity.

    This function suppose that state descriptor is the velocity, as it
    just computes the mean of the state descriptors given.
    """
    # reshape mask for bd extraction
    mask = jnp.expand_dims(mask, axis=-1)

    # Get behavior descriptor
    descriptors = jnp.sum(data.state_desc * (1.0 - mask), axis=1)
    descriptors = descriptors / jnp.sum(1.0 - mask, axis=1)

    return descriptors

