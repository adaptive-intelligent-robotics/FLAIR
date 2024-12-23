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
from gpjax.dataset import Dataset

from functionality_controller.datapoint import DataPoints

OUT_OF_BOUND = 50.0

class FIFODataset(flax.struct.PyTreeNode):
    """FIFO Dataset with a fix size. Written to be fully jitable."""

    size: int = flax.struct.field(pytree_node=False)
    command_x: jnp.ndarray
    command_y: jnp.ndarray
    gp_prediction_x: jnp.ndarray
    gp_prediction_y: jnp.ndarray
    intent_x: jnp.ndarray
    intent_y: jnp.ndarray
    sensor_x: jnp.ndarray
    sensor_y: jnp.ndarray

    @staticmethod
    def create(size: int) -> FIFODataset:
        return FIFODataset(
            command_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            command_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            gp_prediction_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            gp_prediction_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            intent_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            intent_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            sensor_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            sensor_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            size=size,
        )

    @staticmethod
    def reset(dataset: FIFODataset) -> FIFODataset:
        """Reset the dataset content. Used to reset GP."""
        return FIFODataset.create(dataset.size)

    @staticmethod
    @jax.jit
    def _invalid_values(array: jnp.ndarray) -> jnp.ndarray:
        mask = jnp.all(array > -jnp.inf, axis=1)
        mask = jnp.logical_and(mask, jnp.all(array == array, axis=1))
        return mask

    @staticmethod
    @jax.jit
    def _apply_mask(array: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(
            jnp.reshape(
                jnp.repeat(mask, array.shape[1], total_repeat_length=array.shape[0] * array.shape[1]),
                array.shape,
            ),
            array,
            -jnp.inf,
        )

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
    @jax.jit
    def add_and_roll(array: jnp.ndarray, datapoints: jnp.ndarray) -> jnp.ndarray:
        """Function used by add to update the datasets."""

        # Find the valid datapoints (no -jnp.inf anywhere)
        non_valid_values = jnp.any(datapoints == -jnp.inf, axis=1)
        non_valid_values = jnp.reshape(
            jnp.repeat(
                non_valid_values,
                datapoints.shape[1],
                total_repeat_length=datapoints.shape[0] * datapoints.shape[1],
            ),
            datapoints.shape,
        )

        # Sort to put the valid datapoints first
        index_sort = jnp.argsort(non_valid_values, axis=0)
        non_valid_values = jnp.take_along_axis(non_valid_values, index_sort, axis=0)
        array_datapoints = jnp.take_along_axis(datapoints, index_sort, axis=0)

        # Resize to have the same size as the array
        non_valid_values = jnp.resize(non_valid_values, array.shape)
        array_datapoints = jnp.resize(array_datapoints, array.shape)

        # Filter non-existing individuals that have been created by this resize
        non_valid_values = jnp.where(
            jnp.reshape(
                jnp.repeat(
                    jnp.arange(0, array_datapoints.shape[0]), array_datapoints.shape[1]
                ),
                non_valid_values.shape,
            )
            >= datapoints.shape[0],
            True,
            non_valid_values,
        )

        # Replace the first elements of the array with the new datapoints
        new_array = jnp.where(non_valid_values, array, array_datapoints)

        # Roll the array so the older datapoints are at the top
        roll_index = array.shape[0] - jnp.sum(
            jnp.all(1 - non_valid_values, axis=1), axis=0
        )
        new_array = jnp.roll(new_array, roll_index, axis=0)
        return new_array

    @staticmethod
    @jax.jit
    def add(
        dataset: FIFODataset,
        datapoint: DataPoints,
    ) -> FIFODataset:
        """Add a datapoint to the dataset."""

        # Filter any NaN or -jnp.inf
        mask = FIFODataset._invalid_values(datapoint.array_command_x)
        mask = jnp.logical_and(mask, FIFODataset._invalid_values(datapoint.array_command_y))
        mask = jnp.logical_and(mask, FIFODataset._invalid_values(datapoint.array_gp_prediction_x))
        mask = jnp.logical_and(mask, FIFODataset._invalid_values(datapoint.array_gp_prediction_y))
        mask = jnp.logical_and(mask, FIFODataset._invalid_values(datapoint.array_sensor_x))
        mask = jnp.logical_and(mask, FIFODataset._invalid_values(datapoint.array_sensor_y))
        mask = jnp.logical_and(mask, FIFODataset._invalid_values(datapoint.array_intent_x))
        mask = jnp.logical_and(mask, FIFODataset._invalid_values(datapoint.array_intent_y))
        command_x = FIFODataset._apply_mask(datapoint.array_command_x, mask)
        command_y = FIFODataset._apply_mask(datapoint.array_command_y, mask)
        gp_prediction_x = FIFODataset._apply_mask(datapoint.array_gp_prediction_x, mask)
        gp_prediction_y = FIFODataset._apply_mask(datapoint.array_gp_prediction_y, mask)
        intent_x = FIFODataset._apply_mask(datapoint.array_intent_x, mask)
        intent_y = FIFODataset._apply_mask(datapoint.array_intent_y, mask)
        sensor_x = FIFODataset._apply_mask(datapoint.array_sensor_x, mask)
        sensor_y = FIFODataset._apply_mask(datapoint.array_sensor_y, mask)

        # Add to existing dataset
        new_command_x = FIFODataset.add_and_roll(
            dataset.command_x,
            command_x,
        )
        new_command_y = FIFODataset.add_and_roll(
            dataset.command_y,
            command_y,
        )
        new_gp_prediction_x = FIFODataset.add_and_roll(
            dataset.gp_prediction_x,
            gp_prediction_x,
        )
        new_gp_prediction_y = FIFODataset.add_and_roll(
            dataset.gp_prediction_y,
            gp_prediction_y,
        )
        new_intent_x = FIFODataset.add_and_roll(
            dataset.intent_x,
            intent_x,
        )
        new_intent_y = FIFODataset.add_and_roll(
            dataset.intent_y,
            intent_y,
        )
        new_sensor_x = FIFODataset.add_and_roll(
            dataset.sensor_x,
            sensor_x,
        )
        new_sensor_y = FIFODataset.add_and_roll(
            dataset.sensor_y,
            sensor_y,
        )

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


class StateFIFODataset(FIFODataset):
    """FIFO Dataset with a fix size. Written to be fully jitable."""

    state: jnp.ndarray

    @staticmethod
    def create(size: int) -> StateFIFODataset:
        return StateFIFODataset(
            command_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            command_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            gp_prediction_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            gp_prediction_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            intent_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            intent_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            sensor_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            sensor_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            state=OUT_OF_BOUND * jnp.ones((size, 8)),
            size=size,
        )

    @staticmethod
    def reset(dataset: StateFIFODataset) -> StateFIFODataset:
        """Reset the dataset content. Used to reset GP."""
        return StateFIFODataset.create(dataset.size)

    @staticmethod
    @jax.jit
    def add(
        dataset: StateFIFODataset,
        datapoint: DataPoints,
    ) -> StateFIFODataset:
        """Add a datapoint to the dataset."""

        # Filter any NaN or -jnp.inf
        mask = FIFODataset._invalid_values(datapoint.array_command_x)
        mask = jnp.logical_and(mask, FIFODataset._invalid_values(datapoint.array_command_y))
        mask = jnp.logical_and(mask, FIFODataset._invalid_values(datapoint.array_gp_prediction_x))
        mask = jnp.logical_and(mask, FIFODataset._invalid_values(datapoint.array_gp_prediction_y))
        mask = jnp.logical_and(mask, FIFODataset._invalid_values(datapoint.array_sensor_x))
        mask = jnp.logical_and(mask, FIFODataset._invalid_values(datapoint.array_sensor_y))
        mask = jnp.logical_and(mask, FIFODataset._invalid_values(datapoint.array_intent_x))
        mask = jnp.logical_and(mask, FIFODataset._invalid_values(datapoint.array_intent_y))
        mask = jnp.logical_and(mask, FIFODataset._invalid_values(datapoint.array_state))
        command_x = FIFODataset._apply_mask(datapoint.array_command_x, mask)
        command_y = FIFODataset._apply_mask(datapoint.array_command_y, mask)
        gp_prediction_x = FIFODataset._apply_mask(datapoint.array_gp_prediction_x, mask)
        gp_prediction_y = FIFODataset._apply_mask(datapoint.array_gp_prediction_y, mask)
        intent_x = FIFODataset._apply_mask(datapoint.array_intent_x, mask)
        intent_y = FIFODataset._apply_mask(datapoint.array_intent_y, mask)
        sensor_x = FIFODataset._apply_mask(datapoint.array_sensor_x, mask)
        sensor_y = FIFODataset._apply_mask(datapoint.array_sensor_y, mask)
        state = FIFODataset._apply_mask(datapoint.array_state, mask)

        # Add to existing dataset
        new_command_x = FIFODataset.add_and_roll(
            dataset.command_x,
            command_x,
        )
        new_command_y = FIFODataset.add_and_roll(
            dataset.command_y,
            command_y,
        )
        new_gp_prediction_x = FIFODataset.add_and_roll(
            dataset.gp_prediction_x,
            gp_prediction_x,
        )
        new_gp_prediction_y = FIFODataset.add_and_roll(
            dataset.gp_prediction_y,
            gp_prediction_y,
        )
        new_intent_x = FIFODataset.add_and_roll(
            dataset.intent_x,
            intent_x,
        )
        new_intent_y = FIFODataset.add_and_roll(
            dataset.intent_y,
            intent_y,
        )
        new_sensor_x = FIFODataset.add_and_roll(
            dataset.sensor_x,
            sensor_x,
        )
        new_sensor_y = FIFODataset.add_and_roll(
            dataset.sensor_y,
            sensor_y,
        )
        new_state = FIFODataset.add_and_roll(
            dataset.state,
            state,
        )

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
