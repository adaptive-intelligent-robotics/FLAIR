# Written or Adapted by the Imperial College London Team for the FLAIR project, 2023
# Authors for this file: 
# Maxime Allard
# Manon Flageat
# Antoine Cully

from typing import Tuple

from jax.config import config

config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp


def compute_borders(all_descriptors: jnp.ndarray) -> jnp.ndarray:
    max_value = jnp.max(jnp.abs(all_descriptors))
    border_idx = jax.vmap(lambda x: jnp.any(jnp.isin(jnp.abs(x), max_value)))(
        all_descriptors
    )
    return border_idx


def compute_clipping(
    border_idx: jnp.ndarray, all_corrected_descriptors: jnp.ndarray
) -> Tuple[float, float]:
    """Given the current map, compute the clipping to ensure symmetrical
    control for the user.
    """

    # Symmetrical Map
    border_points = all_corrected_descriptors.at[border_idx, ...].get()
    all_radiuses = jax.vmap(jnp.linalg.norm)(border_points)
    max_radius = jnp.nanmin(all_radiuses)

    # If the minimal Radius is bigger than 0.99 we assume that we have a full square
    max_radius = jnp.where(max_radius >= 1.99, 2 * jnp.sqrt(2), max_radius)
    max_x = jnp.sqrt(2) * max_radius * 0.5
    max_y = jnp.sqrt(2) * max_radius * 0.5

    # Filter nan
    max_x = jnp.where(jnp.isnan(max_x), 0.5, max_x)
    max_y = jnp.where(jnp.isnan(max_y), 0.7, max_y)

    # Safety Clips
    max_x = jnp.clip(jnp.abs(max_x), a_min=0.3, a_max=0.5)
    max_y = jnp.clip(jnp.abs(max_y), a_min=0.3, a_max=0.7)

    return max_x, max_y


def compute_metric_map(
    all_corrected_descriptors: jnp.ndarray, uncertainties: jnp.ndarray
):
    """Compute the metrics map for visualisation."""

    # Concatenate map and uncertainties
    metric_map = jnp.concatenate([all_corrected_descriptors, uncertainties], axis=1)

    # Ravel and return as a list
    return jnp.ravel(metric_map).tolist()
