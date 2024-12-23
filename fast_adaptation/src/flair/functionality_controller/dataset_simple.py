# Written or Adapted by the Imperial College London Team for the FLAIR project, 2023
# Authors for this file:
# Maxime Allard
# Manon Flageat

from __future__ import annotations

import json
import time
from typing import Any, Dict, Tuple

import flax
import jax
import jax.numpy as jnp

OUT_OF_BOUND = 50.0


class OutOfBoundDataset(flax.struct.PyTreeNode):
    """Simple Dataset that is onoly uses as an interface for the GP.
    Replace non-valid values with OUT_OF_BOUND."""

    size: int = flax.struct.field(pytree_node=False)
    N: int = flax.struct.field(pytree_node=False)
    command_x: jnp.ndarray
    command_y: jnp.ndarray
    gp_prediction_x: jnp.ndarray
    gp_prediction_y: jnp.ndarray
    intent_x: jnp.ndarray
    intent_y: jnp.ndarray
    sensor_x: jnp.ndarray
    sensor_y: jnp.ndarray

    def to_string(self) -> Dict:
        """Return datasets as a json for analysis and plots."""
        json_file = {
            "command_x": {},
            "command_y": {},
            "gp_prediction_x": {},
            "gp_prediction_y": {},
            "intent_x": {},
            "intent_y": {},
            "sensor_x": {},
            "sensor_y": {},
        }
        json_file["command_x"] = self.command_x.tolist()
        json_file["command_y"] = self.command_y.tolist()
        json_file["gp_prediction_x"] = self.gp_prediction_x.tolist()
        json_file["gp_prediction_y"] = self.gp_prediction_y.tolist()
        json_file["intent_x"] = self.intent_x.tolist()
        json_file["intent_y"] = self.intent_y.tolist()
        json_file["sensor_x"] = self.sensor_x.tolist()
        json_file["sensor_y"] = self.sensor_y.tolist()
        json_string = json.dumps(json_file)
        return json_string

    @staticmethod
    def create(size: int) -> OutOfBoundDataset:
        return OutOfBoundDataset(
            size=size,
            N=size,
            command_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            command_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            gp_prediction_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            gp_prediction_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            intent_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            intent_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            sensor_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            sensor_y=OUT_OF_BOUND * jnp.ones((size, 1)),
        )

    @staticmethod
    def reset(dataset: OutOfBoundDataset) -> OutOfBoundDataset:
        """Reset the dataset content. Used to reset GP."""
        return OutOfBoundDataset.create(dataset.size)

    @staticmethod
    @jax.jit
    def _compute_mask(array: jnp.ndarray) -> jnp.ndarray:
        mask = jnp.all(array > -jnp.inf, axis=1)
        mask = jnp.logical_and(mask, jnp.all(array == array, axis=1))
        return mask

    @staticmethod
    @jax.jit
    def update_datasets(
        dataset: OutOfBoundDataset,
        command_x: jnp.ndarray,
        command_y: jnp.ndarray,
        gp_prediction_x: jnp.ndarray,
        gp_prediction_y: jnp.ndarray,
        intent_x: jnp.ndarray,
        intent_y: jnp.ndarray,
        sensor_x: jnp.ndarray,
        sensor_y: jnp.ndarray,
    ) -> OutOfBoundDataset:
        """Replace invalid point with OUT_OF_BOUND."""

        @jax.jit
        def _repeat_and_reshape(
            to_change: jnp.ndarray, array: jnp.ndarray
        ) -> jnp.ndarray:
            return jnp.reshape(
                jnp.repeat(
                    to_change,
                    array.shape[1],
                    total_repeat_length=array.shape[0] * array.shape[1],
                ),
                array.shape,
            )

        # Find non-valid values across all datasets
        mask = OutOfBoundDataset._compute_mask(command_x)
        mask = jnp.logical_and(mask, OutOfBoundDataset._compute_mask(command_y))
        mask = jnp.logical_and(mask, OutOfBoundDataset._compute_mask(gp_prediction_x))
        mask = jnp.logical_and(mask, OutOfBoundDataset._compute_mask(gp_prediction_y))
        mask = jnp.logical_and(mask, OutOfBoundDataset._compute_mask(sensor_x))
        mask = jnp.logical_and(mask, OutOfBoundDataset._compute_mask(sensor_y))
        mask = jnp.logical_and(mask, OutOfBoundDataset._compute_mask(intent_x))
        mask = jnp.logical_and(mask, OutOfBoundDataset._compute_mask(intent_y))

        # Re-order the arrays to put valid values first
        index_sort = jnp.argsort(jnp.logical_not(mask), axis=0)

        @jax.jit
        def _sort(array: jnp.ndarray, index: jnp.ndarray) -> jnp.ndarray:
            return jnp.take_along_axis(
                array,
                _repeat_and_reshape(index, array),
                axis=0,
            )

        new_command_x = _sort(command_x, index_sort)
        new_command_y = _sort(command_y, index_sort)
        new_gp_prediction_x = _sort(gp_prediction_x, index_sort)
        new_gp_prediction_y = _sort(gp_prediction_y, index_sort)
        new_sensor_x = _sort(sensor_x, index_sort)
        new_sensor_y = _sort(sensor_y, index_sort)
        new_intent_x = _sort(intent_x, index_sort)
        new_intent_y = _sort(intent_y, index_sort)
        mask = jnp.take_along_axis(mask, index_sort, axis=0)

        # Replace nan value with OUT_OF_BOUND value

        @jax.jit
        def _replace(
            array: jnp.ndarray, mask: jnp.ndarray, value: float
        ) -> jnp.ndarray:
            return jnp.where(
                _repeat_and_reshape(mask, array),
                array,
                value,
            )

        new_command_x = _replace(new_command_x, mask, OUT_OF_BOUND)
        new_command_y = _replace(new_command_y, mask, OUT_OF_BOUND)
        new_gp_prediction_x = _replace(new_gp_prediction_x, mask, OUT_OF_BOUND)
        new_gp_prediction_y = _replace(new_gp_prediction_y, mask, OUT_OF_BOUND)
        new_sensor_x = _replace(new_sensor_x, mask, OUT_OF_BOUND)
        new_sensor_y = _replace(new_sensor_y, mask, OUT_OF_BOUND)
        new_intent_x = _replace(new_intent_x, mask, OUT_OF_BOUND)
        new_intent_y = _replace(new_intent_y, mask, OUT_OF_BOUND)

        # Update the dataset
        return dataset.replace(
            command_x=new_command_x,
            command_y=new_command_y,
            gp_prediction_x=new_gp_prediction_x,
            gp_prediction_y=new_gp_prediction_y,
            intent_x=new_intent_x,
            intent_y=new_intent_y,
            sensor_x=new_sensor_x,
            sensor_y=new_sensor_y,
        )


class OutOfBoundStateDataset(OutOfBoundDataset):
    """Simple Dataset that is onoly uses as an interface for the GP.
    Replace non-valid values with OUT_OF_BOUND."""

    state: jnp.ndarray

    @staticmethod
    def create(size: int) -> OutOfBoundDataset:
        return OutOfBoundStateDataset(
            size=size,
            N=size,
            command_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            command_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            gp_prediction_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            gp_prediction_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            intent_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            intent_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            sensor_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            sensor_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            state=OUT_OF_BOUND * jnp.ones((size, 8)),
        )

    @staticmethod
    def reset(dataset: OutOfBoundStateDataset) -> OutOfBoundStateDataset:
        """Reset the dataset content. Used to reset GP."""
        return OutOfBoundStateDataset.create(dataset.size)

    @staticmethod
    @jax.jit
    def update_datasets(
        dataset: OutOfBoundStateDataset,
        command_x: jnp.ndarray,
        command_y: jnp.ndarray,
        gp_prediction_x: jnp.ndarray,
        gp_prediction_y: jnp.ndarray,
        intent_x: jnp.ndarray,
        intent_y: jnp.ndarray,
        sensor_x: jnp.ndarray,
        sensor_y: jnp.ndarray,
        state: jnp.ndarray,
    ) -> OutOfBoundStateDataset:
        """Replace invalid point with OUT_OF_BOUND."""

        @jax.jit
        def _repeat_and_reshape(
            to_change: jnp.ndarray, array: jnp.ndarray
        ) -> jnp.ndarray:
            return jnp.reshape(
                jnp.repeat(
                    to_change,
                    array.shape[1],
                    total_repeat_length=array.shape[0] * array.shape[1],
                ),
                array.shape,
            )

        # Find non-valid values across all datasets
        mask = OutOfBoundDataset._compute_mask(command_x)
        mask = jnp.logical_and(mask, OutOfBoundDataset._compute_mask(command_y))
        mask = jnp.logical_and(mask, OutOfBoundDataset._compute_mask(gp_prediction_x))
        mask = jnp.logical_and(mask, OutOfBoundDataset._compute_mask(gp_prediction_y))
        mask = jnp.logical_and(mask, OutOfBoundDataset._compute_mask(sensor_x))
        mask = jnp.logical_and(mask, OutOfBoundDataset._compute_mask(sensor_y))
        mask = jnp.logical_and(mask, OutOfBoundDataset._compute_mask(intent_x))
        mask = jnp.logical_and(mask, OutOfBoundDataset._compute_mask(intent_y))
        mask = jnp.logical_and(mask, OutOfBoundDataset._compute_mask(state))

        # Re-order the arrays to put valid values first
        index_sort = jnp.argsort(jnp.logical_not(mask), axis=0)

        @jax.jit
        def _sort(array: jnp.ndarray, index: jnp.ndarray) -> jnp.ndarray:
            return jnp.take_along_axis(
                array,
                _repeat_and_reshape(index, array),
                axis=0,
            )

        new_command_x = _sort(command_x, index_sort)
        new_command_y = _sort(command_y, index_sort)
        new_gp_prediction_x = _sort(gp_prediction_x, index_sort)
        new_gp_prediction_y = _sort(gp_prediction_y, index_sort)
        new_sensor_x = _sort(sensor_x, index_sort)
        new_sensor_y = _sort(sensor_y, index_sort)
        new_intent_x = _sort(intent_x, index_sort)
        new_intent_y = _sort(intent_y, index_sort)
        new_state = _sort(state, index_sort)
        mask = jnp.take_along_axis(mask, index_sort, axis=0)

        # Replace nan value with OUT_OF_BOUND value

        @jax.jit
        def _replace(
            array: jnp.ndarray, mask: jnp.ndarray, value: float
        ) -> jnp.ndarray:
            return jnp.where(
                _repeat_and_reshape(mask, array),
                array,
                value,
            )

        new_command_x = _replace(new_command_x, mask, OUT_OF_BOUND)
        new_command_y = _replace(new_command_y, mask, OUT_OF_BOUND)
        new_gp_prediction_x = _replace(new_gp_prediction_x, mask, OUT_OF_BOUND)
        new_gp_prediction_y = _replace(new_gp_prediction_y, mask, OUT_OF_BOUND)
        new_sensor_x = _replace(new_sensor_x, mask, OUT_OF_BOUND)
        new_sensor_y = _replace(new_sensor_y, mask, OUT_OF_BOUND)
        new_intent_x = _replace(new_intent_x, mask, OUT_OF_BOUND)
        new_intent_y = _replace(new_intent_y, mask, OUT_OF_BOUND)
        new_state = _replace(new_state, mask, OUT_OF_BOUND)

        # Update the dataset
        return dataset.replace(
            command_x=new_command_x,
            command_y=new_command_y,
            gp_prediction_x=new_gp_prediction_x,
            gp_prediction_y=new_gp_prediction_y,
            intent_x=new_intent_x,
            intent_y=new_intent_y,
            sensor_x=new_sensor_x,
            sensor_y=new_sensor_y,
            state=new_state,
        )
