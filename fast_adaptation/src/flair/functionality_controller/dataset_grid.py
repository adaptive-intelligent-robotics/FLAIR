# Written or Adapted by the Imperial College London Team for the FLAIR project, 2023
# Authors for this file:
# Maxime Allard
# Manon Flageat

from __future__ import annotations

import json
import math
import sys
import time
from copy import deepcopy
from functools import partial
from typing import Any, Dict, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
from functionality_controller.datapoint import DataPoints

jax.numpy.set_printoptions(threshold=sys.maxsize)

OUT_OF_BOUND = 50.0


class Grid(flax.struct.PyTreeNode):
    """Class defining a grid of FIFOs to maintain dataset as well
    as the functions to extract non-outliers datapoints from it.
    """

    # Constants
    cell_depth: int = flax.struct.field(pytree_node=False)
    num_cells: int = flax.struct.field(pytree_node=False)
    out_of_bound: int = flax.struct.field(pytree_node=False)
    input_bins: int = flax.struct.field(pytree_node=False)
    k_neighbours: int = flax.struct.field(pytree_node=False)
    novelty_threshold: float = flax.struct.field(pytree_node=False)
    input_dimension: int = flax.struct.field(pytree_node=False)
    grid_1d_projection: jnp.ndarray
    min_array: jnp.ndarray
    range_array: jnp.ndarray

    # Grid
    size: int = flax.struct.field(pytree_node=False)
    point_id_grid: jnp.ndarray
    selected_grid: jnp.ndarray
    inputs_grid: jnp.ndarray
    sensors_grid: jnp.ndarray

    @staticmethod
    def create(
        size: int,
        min_array: jnp.ndarray,
        range_array: jnp.ndarray,
        cell_depth: int,
        input_bins: int,
        k_neighbours: int,
        novelty_threshold: float,
        input_dimension: int,
        input_shape: int,
        sensor_shape: int,
    ) -> Grid:
        """
        Create the Grid..

        Args:
            size: global FIFO size.
            cell_depth: size of the FIFO within each cell.
            input_bins: number of bins per input dimension.
        """

        # Compute grid sizes
        num_cells = input_bins**input_dimension

        # Compute 1D projection
        grid_1d_projection = jnp.asarray(
            [input_bins**j for j in range(input_dimension)]
        )
        out_of_bound = num_cells * cell_depth + 1

        # Create the grid
        point_id_grid = -jnp.inf * jnp.ones(shape=(num_cells, cell_depth))
        selected_grid = jnp.zeros(shape=(num_cells, cell_depth), dtype=bool)
        inputs_grid = -jnp.inf * jnp.ones(shape=(num_cells, cell_depth, input_shape))
        sensors_grid = -jnp.inf * jnp.ones(shape=(num_cells, cell_depth, sensor_shape))

        return Grid(
            size=size,
            cell_depth=cell_depth,
            num_cells=num_cells,
            out_of_bound=out_of_bound,
            input_bins=input_bins,
            k_neighbours=k_neighbours,
            novelty_threshold=novelty_threshold,
            input_dimension=input_dimension,
            grid_1d_projection=grid_1d_projection,
            min_array=min_array,
            range_array=range_array,
            point_id_grid=point_id_grid,
            selected_grid=selected_grid,
            inputs_grid=inputs_grid,
            sensors_grid=sensors_grid,
        )

    @staticmethod
    def reset(grid: Grid) -> Grid:

        point_id_grid = -jnp.inf * jnp.ones_like(grid.point_id_grid)
        selected_grid = jnp.zeros_like(grid.selected_grid, dtype=bool)
        inputs_grid = -jnp.inf * jnp.ones_like(grid.inputs_grid)
        sensors_grid = -jnp.inf * jnp.ones_like(grid.sensors_grid)

        return grid.replace(
            point_id_grid=point_id_grid,
            selected_grid=selected_grid,
            inputs_grid=inputs_grid,
            sensors_grid=sensors_grid,
        )

    @staticmethod
    @jax.jit
    def select(grid: Grid) -> Grid:
        """
        Return a array of the same shape as the grid with value 1 for datapoints that
        should be added to the datasets for the GP, based on the novelty criteria.
        """

        @partial(jax.jit, static_argnames=("k_neighbours",))
        def novelty(
            cell: jnp.ndarray, filled: jnp.ndarray, k_neighbours: int
        ) -> jnp.ndarray:
            """Compute novelty of each of the datapoint in the cell."""

            @jax.jit
            def distance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
                return jnp.sqrt(jnp.sum(jnp.square(x - y)))

            # Compute distance matrix
            distances = jax.vmap(
                jax.vmap(partial(distance), in_axes=(None, 0)), in_axes=(0, None)
            )(cell, cell)

            # Filter distance with empty slot
            distances = jax.vmap(lambda distance: jnp.where(filled, distance, jnp.inf))(
                distances
            )

            # Find k nearest neighbours
            _, indices = jax.lax.top_k(-distances, k_neighbours)

            # Compute novelty as average distance with k neirest neighbours
            distances = jnp.where(distances == jnp.inf, jnp.nan, distances)
            novelty = jnp.nanmean(
                jnp.take_along_axis(distances, indices, axis=1), axis=1
            )
            return novelty

        # Compute novelty of each datapoints using its sensor values
        novelties = jax.vmap(partial(novelty, k_neighbours=grid.k_neighbours))(
            grid.sensors_grid, grid.point_id_grid > -jnp.inf
        )

        # Remove nan to allow comparison
        novelties = jnp.where(novelties == jnp.nan, jnp.inf, novelties)

        # Filter all novelty greater than the threshold
        selected_grid = novelties < grid.novelty_threshold

        # Remove empty individual
        selected_grid = jnp.where(grid.point_id_grid > -jnp.inf, selected_grid, False)
        return grid.replace(selected_grid=selected_grid)

    @staticmethod
    @jax.jit
    def get_indices(grid: Grid, commands: jnp.ndarray) -> jnp.ndarray:
        """Compute the indice of a command tuple in the grid."""
        cell_range = jnp.expand_dims(grid.range_array / grid.input_bins, axis=0)
        batch_of_indices = jnp.ceil((commands - grid.min_array) / cell_range)
        batch_of_indices = jnp.sum(batch_of_indices * grid.grid_1d_projection, axis=1)
        return batch_of_indices.astype(jnp.int32)

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=(
            "num_cells",
            "out_of_bound",
        ),
    )
    def indices_to_occurence(
        batch_of_indices: jnp.ndarray, num_cells: int, out_of_bound: int
    ) -> jnp.ndarray:
        """
        Sub-method for add(). Return an array similar to the batch_of_indices
        replacing each indice with its occurence number in the batch.

        Args:
            batch_of_indices: indices of new indivs

        Returns: batch_of_occurences: number of occurence for each indice
        """

        @partial(jax.jit, static_argnames=("max_indice",))
        def _cumulative_count(
            idx: int,
            indices: jnp.ndarray,
            batch_of_indices: jnp.ndarray,
            max_indice: int,
        ) -> int:
            filter_batch_of_indices = jnp.where(
                indices.ravel() <= idx, batch_of_indices, max_indice
            )
            count_indices = jnp.bincount(filter_batch_of_indices, length=max_indice)
            return count_indices.at[batch_of_indices[idx]].get() - 1  # type: ignore

        # Get occurence
        indices = jnp.arange(0, batch_of_indices.shape[0], step=1)
        cumulative_count = partial(
            _cumulative_count,
            indices=indices,
            batch_of_indices=batch_of_indices,
            max_indice=num_cells,
        )
        batch_of_occurence = jax.vmap(cumulative_count)(indices)

        # Filter out-of-bond datapoints
        batch_of_occurence = jnp.where(
            batch_of_indices < out_of_bound, batch_of_occurence, out_of_bound
        )
        return batch_of_occurence

    @staticmethod
    @jax.jit
    def add(
        grid: Grid,
        inputs: jnp.ndarray,
        sensors: jnp.ndarray,
    ) -> Grid:
        """
        Add a new datapoint to the grid, fully jitted to make this operation faster.
        """

        # Compute the cell index of the inputs
        batch_of_indices = Grid.get_indices(grid, inputs[:, 0 : grid.input_dimension])

        # Remove all datapoints with inf inputs
        batch_of_indices = jnp.where(
            inputs[:, 0] > -jnp.inf, batch_of_indices, grid.out_of_bound
        )

        # Only add cell_depth datapoints per cell, filter datapoints that pass that limit
        batch_of_occurence = Grid.indices_to_occurence(
            batch_of_indices, grid.num_cells, grid.out_of_bound
        )
        batch_of_indices = jnp.where(
            batch_of_occurence < grid.cell_depth,
            batch_of_indices,
            grid.out_of_bound,
        )
        batch_of_occurence = jnp.where(
            batch_of_occurence < grid.cell_depth,
            batch_of_occurence,
            grid.out_of_bound,
        )

        # Set up the ids of the new datapoints
        current_id = jnp.max(grid.point_id_grid)
        current_id = jnp.where(current_id > -jnp.inf, current_id, 0)
        batch_of_ids = Grid.indices_to_occurence(
            (batch_of_indices < grid.out_of_bound).astype(jnp.int32),
            grid.num_cells,
            grid.out_of_bound,
        )
        batch_of_ids = jnp.where(batch_of_indices < grid.out_of_bound, batch_of_ids, 0)
        batch_of_ids = jnp.max(batch_of_ids) - batch_of_ids
        batch_of_ids = current_id + 1 + batch_of_ids
        batch_of_ids = jnp.where(
            batch_of_indices < grid.out_of_bound, batch_of_ids, -jnp.inf
        )

        # Function to add datapoints inside cells
        @partial(
            jax.jit,
            static_argnames=(
                "out_of_bound",
                "cell_depth",
            ),
        )
        def _cell_add_and_roll(
            cell: jnp.ndarray,
            cell_indice: jnp.ndarray,
            datapoints: jnp.ndarray,
            batch_of_indices: jnp.ndarray,
            batch_of_occurence: jnp.ndarray,
            out_of_bound: int,
            cell_depth: int,
        ) -> jnp.ndarray:
            """
            Sub-function used to do addition to grid while handling the
            fifo within each cell. In short this function put within each cell
            the oldest datapoints at the top of teh cell (i.e. start), this way no
            need to store the FIFO position.
            """

            # Extract the new datapoints that belong to this cell only
            cell_batch_of_occurence = jnp.where(
                batch_of_indices == cell_indice,
                batch_of_occurence,
                out_of_bound,
            )

            # Reshape cell_batch_of_occurence to the same shape as datapoints
            cell_batch_of_occurence = jnp.reshape(
                jnp.repeat(cell_batch_of_occurence, datapoints.shape[1], axis=0),
                (-1, datapoints.shape[1]),
            )

            # Sort to put the datapoints that belong to this cell first
            index_sort = jnp.argsort(cell_batch_of_occurence, axis=0)
            cell_batch_of_occurence = jnp.take_along_axis(
                cell_batch_of_occurence, index_sort, axis=0
            )
            cell_datapoints = jnp.take_along_axis(datapoints, index_sort, axis=0)

            # Resize to have something size of one cell
            cell_batch_of_occurence = jnp.resize(cell_batch_of_occurence, cell.shape)
            cell_datapoints = jnp.resize(cell_datapoints, cell.shape)

            # Filter non-existing individuals that have been created by this resize
            cell_batch_of_occurence = jnp.where(
                jnp.reshape(
                    jnp.repeat(jnp.arange(0, cell_depth), datapoints.shape[1]),
                    cell_batch_of_occurence.shape,
                )
                >= datapoints.shape[0],
                out_of_bound,
                cell_batch_of_occurence,
            )

            # Replace the first datapoints of cell with new datapoints
            new_cell = jnp.where(
                cell_batch_of_occurence < out_of_bound,
                cell_datapoints,
                cell,
            )

            # Roll the cell so the older points are at the top
            roll_index = cell_depth - jnp.max(
                jnp.where(
                    cell_batch_of_occurence < out_of_bound,
                    cell_batch_of_occurence + 1,
                    0,
                )
            )
            new_cell = jnp.roll(
                new_cell,
                roll_index,
                axis=0,
            )
            return new_cell

        # Add all datapoints
        cell_indices = jnp.arange(0, grid.num_cells)
        cell_add_and_roll = partial(
            _cell_add_and_roll,
            batch_of_indices=batch_of_indices,
            batch_of_occurence=batch_of_occurence,
            out_of_bound=grid.out_of_bound,
            cell_depth=grid.cell_depth,
        )

        new_point_id_grid = jax.vmap(
            partial(cell_add_and_roll, datapoints=jnp.expand_dims(batch_of_ids, axis=1))
        )(grid.point_id_grid, cell_indices)

        new_inputs_grid = jax.vmap(partial(cell_add_and_roll, datapoints=inputs))(
            grid.inputs_grid, cell_indices
        )

        new_sensors_grid = jax.vmap(partial(cell_add_and_roll, datapoints=sensors))(
            grid.sensors_grid, cell_indices
        )

        # Remove datapoints that are too old according to the time condition
        # current_grid_ids = jnp.sort(jnp.ravel(new_point_id_grid))[::-1]
        # new_max_id = current_grid_ids.at[grid.size].get()
        current_id = jnp.max(new_point_id_grid)
        current_id = jnp.where(current_id > -jnp.inf, current_id, 0)
        new_max_id = current_id - grid.size
        new_point_id_grid = jnp.where(
            new_point_id_grid < new_max_id,
            -jnp.inf,
            new_point_id_grid,
        )

        # Return
        return grid.replace(
            point_id_grid=new_point_id_grid,
            inputs_grid=new_inputs_grid,
            sensors_grid=new_sensors_grid,
        )


class InputGridFIFODataset(flax.struct.PyTreeNode):
    """Dataset class wrapping the Grid ddefined previously for standard no-state GP."""

    size: int = flax.struct.field(pytree_node=False)
    cell_depth: int = flax.struct.field(pytree_node=False)
    input_bins: int = flax.struct.field(pytree_node=False)
    prop_neighbours: float = flax.struct.field(pytree_node=False)
    novelty_threshold: float = flax.struct.field(pytree_node=False)
    k_neighbours: int = flax.struct.field(pytree_node=False)
    command_x: jnp.ndarray
    command_y: jnp.ndarray
    gp_prediction_x: jnp.ndarray
    gp_prediction_y: jnp.ndarray
    intent_x: jnp.ndarray
    intent_y: jnp.ndarray
    sensor_x: jnp.ndarray
    sensor_y: jnp.ndarray
    grid: Grid

    @staticmethod
    def create(
        size: int,
        min_command: int,
        max_command: int,
        cell_depth: int,
        input_bins: int,
        prop_neighbours: float,
        novelty_threshold: int,
        datapoint_batch_size: int,
    ) -> InputGridFIFODataset:
        """
        Create the InputGridFIFODataset dataset.

        Args:
            size: global FIFO size.
            cell_depth: size of the FIFO within each cell.
            input_bins: number of bins per input dimension.
            prop_neighbours: proportion of cell size used to compute
                novelty using knn.
            novelty_threshold: threshold on novelty to be considered
                as an outlier and hidden from dataset.
        """

        # Compute novelty parameter
        k_neighbours = math.floor(prop_neighbours * cell_depth)
        novelty_threshold = (
            math.sqrt(math.pow(max_command - min_command, 2) * 2) * novelty_threshold
        )

        # Compute range for indices
        min_array = jnp.asarray([min_command, min_command])
        range_array = jnp.asarray(
            [max_command - min_command, max_command - min_command]
        )

        grid = Grid.create(
            size=deepcopy(size),
            min_array=min_array,
            range_array=range_array,
            cell_depth=deepcopy(cell_depth),
            input_bins=deepcopy(input_bins),
            k_neighbours=deepcopy(k_neighbours),
            novelty_threshold=deepcopy(novelty_threshold),
            input_dimension=2,
            input_shape=6,
            sensor_shape=2,
        )

        # Force the add and select jitting by adding a false batch
        empty_inputs = -jnp.inf * jnp.ones((datapoint_batch_size, 6))
        empty_sensors = -jnp.inf * jnp.ones((datapoint_batch_size, 2))
        grid = Grid.add(grid, empty_inputs, empty_sensors)
        grid = Grid.select(grid)

        return InputGridFIFODataset(
            size=size,
            cell_depth=cell_depth,
            input_bins=input_bins,
            prop_neighbours=prop_neighbours,
            novelty_threshold=novelty_threshold,
            k_neighbours=k_neighbours,
            command_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            command_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            gp_prediction_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            gp_prediction_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            intent_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            intent_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            sensor_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            sensor_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            grid=grid,
        )

    @staticmethod
    def reset(dataset: InputGridFIFODataset) -> InputGridFIFODataset:
        """Reset the dataset content. Used to reset GP."""
        grid = Grid.reset(dataset.grid)
        size = dataset.size
        return dataset.replace(
            command_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            command_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            gp_prediction_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            gp_prediction_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            intent_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            intent_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            sensor_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            sensor_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            grid=grid,
        )

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
                jnp.repeat(
                    mask,
                    array.shape[1],
                    total_repeat_length=array.shape[0] * array.shape[1],
                ),
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

    def grid_to_string(self) -> Dict:
        """Return grid as a json for analysis and plots."""

        filled = self.grid.point_id_grid > -jnp.inf

        # Get the command and sensors filled in the grid as points
        inputs = self.grid.inputs_grid.at[filled].get()
        sensors = self.grid.sensors_grid.at[filled].get()

        # Get the corresponding cell indices as cluster-colors
        indices = Grid.get_indices(self.grid, inputs[:, 0:2])

        # Smooth selected to [0.5, 1] to use them as points-opacity
        selected = self.grid.selected_grid.at[filled].get()
        modified_selected = jnp.where(selected, 1, 0.3)

        # Send everything
        json_file = {  # type: ignore
            "command_x": {},
            "command_x": {},
            "sensor_x": {},
            "sensor_y": {},
            "selected": {},
            "cells": {},
        }
        json_file["command_x"] = inputs[:, 0].tolist()
        json_file["command_y"] = inputs[:, 1].tolist()
        json_file["sensor_x"] = sensors[:, 0].tolist()
        json_file["sensor_y"] = sensors[:, 1].tolist()
        json_file["selected"] = modified_selected.tolist()
        json_file["cells"] = indices.tolist()
        json_string = json.dumps(json_file)
        return json_string

    @staticmethod
    @jax.jit
    def create_vectors(
        inputs: jnp.ndarray,
        sensors: jnp.ndarray,
    ) -> Tuple:
        """Create the vectors for add method from the selected vectors."""

        command_x = jnp.expand_dims(inputs[:, 0], axis=1)
        command_y = jnp.expand_dims(inputs[:, 1], axis=1)
        gp_prediction_x = jnp.expand_dims(inputs[:, 2], axis=1)
        gp_prediction_y = jnp.expand_dims(inputs[:, 3], axis=1)
        intent_x = jnp.expand_dims(inputs[:, 4], axis=1)
        intent_y = jnp.expand_dims(inputs[:, 5], axis=1)
        sensor_x = jnp.expand_dims(sensors[:, 0], axis=1)
        sensor_y = jnp.expand_dims(sensors[:, 1], axis=1)

        return (
            command_x,
            command_y,
            gp_prediction_x,
            gp_prediction_y,
            intent_x,
            intent_y,
            sensor_x,
            sensor_y,
        )

    @staticmethod
    @jax.jit
    def _add(
        grid: Grid,
        datapoint: DataPoints,
    ) -> Tuple:
        """Add new datapoints to the grid and return the new datasets."""

        # Filter any NaN or -jnp.inf
        mask = InputGridFIFODataset._invalid_values(datapoint.array_command_x)
        mask = jnp.logical_and(mask, InputGridFIFODataset._invalid_values(datapoint.array_command_y))
        mask = jnp.logical_and(mask, InputGridFIFODataset._invalid_values(datapoint.array_gp_prediction_x))
        mask = jnp.logical_and(mask, InputGridFIFODataset._invalid_values(datapoint.array_gp_prediction_y))
        mask = jnp.logical_and(mask, InputGridFIFODataset._invalid_values(datapoint.array_sensor_x))
        mask = jnp.logical_and(mask, InputGridFIFODataset._invalid_values(datapoint.array_sensor_y))
        mask = jnp.logical_and(mask, InputGridFIFODataset._invalid_values(datapoint.array_intent_x))
        mask = jnp.logical_and(mask, InputGridFIFODataset._invalid_values(datapoint.array_intent_y))
        array_command_x = InputGridFIFODataset._apply_mask(datapoint.array_command_x, mask)

        # Get the vectors
        inputs = jnp.concatenate(
            [
                array_command_x,
                datapoint.array_command_y,
                datapoint.array_gp_prediction_x,
                datapoint.array_gp_prediction_y,
                datapoint.array_intent_x,
                datapoint.array_intent_y,
            ],
            axis=1,
        )
        sensors = jnp.concatenate(
            [
                datapoint.array_sensor_x,
                datapoint.array_sensor_y,
            ],
            axis=1,
        )

        # Add to grid
        grid = Grid.add(grid, inputs, sensors)

        # Remove outliers
        grid = Grid.select(grid)

        # Extract only selected values from the grid
        indexes = jnp.where(
            grid.selected_grid, size=grid.size, fill_value=grid.out_of_bound
        )
        selected_inputs = grid.inputs_grid.at[indexes].get(
            indices_are_sorted=True,
            unique_indices=True,
            mode="fill",
            fill_value=OUT_OF_BOUND,
        )
        selected_sensors = grid.sensors_grid.at[indexes].get(
            indices_are_sorted=True,
            unique_indices=True,
            mode="fill",
            fill_value=OUT_OF_BOUND,
        )

        # Create datasets from the selected values
        (
            command_x,
            command_y,
            gp_prediction_x,
            gp_prediction_y,
            intent_x,
            intent_y,
            sensor_x,
            sensor_y,
        ) = InputGridFIFODataset.create_vectors(
            selected_inputs,
            selected_sensors,
        )

        return (
            command_x,
            command_y,
            gp_prediction_x,
            gp_prediction_y,
            intent_x,
            intent_y,
            sensor_x,
            sensor_y,
            grid,
        )

    @staticmethod
    @jax.jit
    def add(
        dataset: InputGridFIFODataset,
        datapoint: DataPoints,
    ) -> InputGridFIFODataset:
        """Add a datapoint to the dataset."""

        (
            command_x,
            command_y,
            gp_prediction_x,
            gp_prediction_y,
            intent_x,
            intent_y,
            sensor_x,
            sensor_y,
            grid,
        ) = dataset._add(
            grid=dataset.grid,
            datapoint=datapoint,
        )

        return dataset.replace(
            command_x=command_x,
            command_y=command_y,
            gp_prediction_x=gp_prediction_x,
            gp_prediction_y=gp_prediction_y,
            intent_x=intent_x,
            intent_y=intent_y,
            sensor_x=sensor_x,
            sensor_y=sensor_y,
            grid=grid,
        )


class MultiDimsStateInputGridFIFODataset(InputGridFIFODataset):
    """Dataset class wrapping the Grid defined previously, for State-based GP."""

    state: jnp.ndarray
    state_dimensions: jnp.ndarray

    @staticmethod
    def create(
        size: int,
        min_command: int,
        max_command: int,
        cell_depth: int,
        input_bins: int,
        prop_neighbours: float,
        novelty_threshold: int,
        state_dimensions: jnp.ndarray,
        state_dimensions_min: jnp.ndarray,
        state_dimensions_max: jnp.ndarray,
        datapoint_batch_size: int,
    ) -> MultiDimsStateInputGridFIFODataset:
        """
        Create the MultiDimsStateInputGridFIFODataset dataset.

        Args:
            size: global FIFO size.
            cell_depth: size of the FIFO within each cell.
            input_bins: number of bins per input dimension.
            prop_neighbours: proportion of cell size used to compute
                novelty using knn.
            novelty_threshold: threshold on novelty to be considered
                as an outlier and hidden from dataset.
            state_dimensions: dimensions of the state used as additional
                input dimensions of the dataset grid
        """

        # Compute novelty parameter
        k_neighbours = math.floor(prop_neighbours * cell_depth)
        novelty_threshold = (
            math.sqrt(math.pow(max_command - min_command, 2) * 2) * novelty_threshold
        )

        # Compute range for indices
        min_array = jnp.concatenate(
            [jnp.asarray([min_command, min_command]), state_dimensions_min], axis=0
        )
        range_array = jnp.concatenate(
            [
                jnp.asarray([max_command - min_command, max_command - min_command]),
                state_dimensions_max - state_dimensions_min,
            ],
            axis=0,
        )

        grid = Grid.create(
            size=deepcopy(size),
            min_array=min_array,
            range_array=range_array,
            cell_depth=deepcopy(cell_depth),
            input_bins=deepcopy(input_bins),
            k_neighbours=deepcopy(k_neighbours),
            novelty_threshold=deepcopy(novelty_threshold),
            input_dimension=2 + state_dimensions.shape[0],
            input_shape=14 + state_dimensions.shape[0],
            sensor_shape=2,
        )

        # Force the add and select jitting by adding a false batch
        empty_inputs = -jnp.inf * jnp.ones(
            (datapoint_batch_size, 14 + state_dimensions.shape[0])
        )
        empty_sensors = -jnp.inf * jnp.ones((datapoint_batch_size, 2))
        grid = Grid.add(grid, empty_inputs, empty_sensors)
        grid = Grid.select(grid)

        return MultiDimsStateInputGridFIFODataset(
            size=size,
            cell_depth=cell_depth,
            input_bins=input_bins,
            prop_neighbours=prop_neighbours,
            novelty_threshold=novelty_threshold,
            k_neighbours=k_neighbours,
            state_dimensions=state_dimensions,
            command_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            command_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            gp_prediction_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            gp_prediction_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            intent_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            intent_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            sensor_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            sensor_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            state=OUT_OF_BOUND * jnp.ones((size, 8)),
            grid=grid,
        )

    @staticmethod
    def reset(dataset: InputGridFIFODataset) -> InputGridFIFODataset:
        """Reset the dataset content. Used to reset GP."""
        grid = Grid.reset(dataset.grid)
        size = dataset.size
        return dataset.replace(
            command_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            command_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            gp_prediction_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            gp_prediction_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            intent_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            intent_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            sensor_x=OUT_OF_BOUND * jnp.ones((size, 1)),
            sensor_y=OUT_OF_BOUND * jnp.ones((size, 1)),
            state=OUT_OF_BOUND * jnp.ones((size, 8)),
            grid=grid,
        )

    def grid_to_string(self) -> Dict:
        """Return grid as a json for analysis and plots."""

        filled = self.grid.point_id_grid > -jnp.inf

        # Get the command and sensors filled in the grid as points
        inputs = self.grid.inputs_grid.at[filled].get()
        sensors = self.grid.sensors_grid.at[filled].get()

        # Get the corresponding cell indices as cluster-colors
        indices = Grid.get_indices(self.grid, inputs[:, 0:3])

        # Smooth selected to [0.5, 1] to use them as points-opacity
        selected = self.grid.selected_grid.at[filled].get()
        modified_selected = jnp.where(selected, 1, 0.3)

        # Send everything
        json_file = {  # type: ignore
            "command_x": {},
            "command_y": {},
            "sensor_x": {},
            "sensor_y": {},
            "selected": {},
            "cells": {},
        }
        json_file["command_x"] = inputs[:, 0].tolist()
        json_file["command_y"] = inputs[:, 1].tolist()
        json_file["sensor_x"] = sensors[:, 0].tolist()
        json_file["sensor_y"] = sensors[:, 1].tolist()
        json_file["selected"] = modified_selected.tolist()
        json_file["cells"] = indices.tolist()
        json_string = json.dumps(json_file)
        return json_string

    @staticmethod
    @partial(jax.jit, static_argnames=("offset",))
    def create_vectors(
        offset: int,
        inputs: jnp.ndarray,
        sensors: jnp.ndarray,
    ) -> Tuple:
        """Create the vectors for add method from the selected vectors."""

        command_x = jnp.expand_dims(inputs[:, 0], axis=1)
        command_y = jnp.expand_dims(inputs[:, 1], axis=1)
        gp_prediction_x = jnp.expand_dims(inputs[:, offset + 2], axis=1)
        gp_prediction_y = jnp.expand_dims(inputs[:, offset + 3], axis=1)
        intent_x = jnp.expand_dims(inputs[:, offset + 4], axis=1)
        intent_y = jnp.expand_dims(inputs[:, offset + 5], axis=1)
        sensor_x = jnp.expand_dims(sensors[:, 0], axis=1)
        sensor_y = jnp.expand_dims(sensors[:, 1], axis=1)
        state = inputs[:, offset + 6 : offset + 14]

        return (
            command_x,
            command_y,
            gp_prediction_x,
            gp_prediction_y,
            intent_x,
            intent_y,
            sensor_x,
            sensor_y,
            state,
        )

    @staticmethod
    @jax.jit
    def _add(
        grid: Grid,
        datapoint: DataPoints,
        state_dimensions: jnp.ndarray,
    ) -> Tuple:
        """Add new datapoints to the grid and return the new datasets."""

        # Filter any NaN or -jnp.inf
        mask = InputGridFIFODataset._invalid_values(datapoint.array_command_x)
        mask = jnp.logical_and(mask, InputGridFIFODataset._invalid_values(datapoint.array_command_y))
        mask = jnp.logical_and(mask, InputGridFIFODataset._invalid_values(datapoint.array_gp_prediction_x))
        mask = jnp.logical_and(mask, InputGridFIFODataset._invalid_values(datapoint.array_gp_prediction_y))
        mask = jnp.logical_and(mask, InputGridFIFODataset._invalid_values(datapoint.array_sensor_x))
        mask = jnp.logical_and(mask, InputGridFIFODataset._invalid_values(datapoint.array_sensor_y))
        mask = jnp.logical_and(mask, InputGridFIFODataset._invalid_values(datapoint.array_intent_x))
        mask = jnp.logical_and(mask, InputGridFIFODataset._invalid_values(datapoint.array_intent_y))
        mask = jnp.logical_and(mask, InputGridFIFODataset._invalid_values(datapoint.array_state))
        array_command_x = InputGridFIFODataset._apply_mask(datapoint.array_command_x, mask)

        # Get the vectors
        inputs = jnp.concatenate(
            [
                array_command_x,
                datapoint.array_command_y,
                datapoint.array_state[:, state_dimensions],
                datapoint.array_gp_prediction_x,
                datapoint.array_gp_prediction_y,
                datapoint.array_intent_x,
                datapoint.array_intent_y,
                datapoint.array_state,
            ],
            axis=1,
        )
        sensors = jnp.concatenate(
            [
                datapoint.array_sensor_x,
                datapoint.array_sensor_y,
            ],
            axis=1,
        )

        # Add to grid
        grid = Grid.add(grid, inputs, sensors)

        # Remove outliers
        grid = Grid.select(grid)

        # Extract only selected values from the grid and place fake point to -jnp.inf
        indexes = jnp.where(
            grid.selected_grid, size=grid.size, fill_value=grid.out_of_bound
        )
        selected_inputs = grid.inputs_grid.at[indexes].get(
            indices_are_sorted=True,
            unique_indices=True,
            mode="fill",
            fill_value=OUT_OF_BOUND,
        )
        selected_sensors = grid.sensors_grid.at[indexes].get(
            indices_are_sorted=True,
            unique_indices=True,
            mode="fill",
            fill_value=OUT_OF_BOUND,
        )

        # Create datasets from the selected values
        (
            command_x,
            command_y,
            gp_prediction_x,
            gp_prediction_y,
            intent_x,
            intent_y,
            sensor_x,
            sensor_y,
            state,
        ) = MultiDimsStateInputGridFIFODataset.create_vectors(
            state_dimensions.shape[0],
            selected_inputs,
            selected_sensors,
        )

        return (
            command_x,
            command_y,
            gp_prediction_x,
            gp_prediction_y,
            intent_x,
            intent_y,
            sensor_x,
            sensor_y,
            state,
            grid,
        )

    @staticmethod
    @jax.jit
    def add(
        dataset: MultiDimsStateInputGridFIFODataset,
        datapoint: DataPoints,
    ) -> MultiDimsStateInputGridFIFODataset:
        """Add a datapoint to the dataset."""

        (
            command_x,
            command_y,
            gp_prediction_x,
            gp_prediction_y,
            intent_x,
            intent_y,
            sensor_x,
            sensor_y,
            state,
            grid,
        ) = dataset._add(
            grid=dataset.grid,
            datapoint=datapoint,
            state_dimensions=dataset.state_dimensions,
        )

        return dataset.replace(
            command_x=command_x,
            command_y=command_y,
            gp_prediction_x=gp_prediction_x,
            gp_prediction_y=gp_prediction_y,
            intent_x=intent_x,
            intent_y=intent_y,
            sensor_x=sensor_x,
            sensor_y=sensor_y,
            state=state,
            grid=grid,
        )
