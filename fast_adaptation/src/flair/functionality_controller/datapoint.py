# Written or Adapted by the Imperial College London Team for the FLAIR project, 2023
# Authors for this file:
# Maxime Allard
# Manon Flageat
# Bryan Lim
# Antoine Cully

from __future__ import annotations

import flax
import jax
import jax.numpy as jnp
import numpy as np


class DataPoints(flax.struct.PyTreeNode):
    """
    This class implements the DataPoints structure used across files in
    the fucntionality controller.

    It suses exclusively static methods to avoid any ambiguity on
    which input methods apply: datapoints to modify are given as input.

    List of methods:
        create: create a new datapoint from floats.
        add: add floats to an existing datapoint.
        to_msg: transform DataPoints into a list of msg to send to Influx.
    """

    max_capacity: jnp.int16
    current_size: jnp.int16

    array_point_id: jnp.ndarray

    array_sensor_time_sec: jnp.ndarray
    array_sensor_time_nanosec: jnp.ndarray
    array_command_time_sec: jnp.ndarray
    array_command_time_nanosec: jnp.ndarray

    array_state: jnp.ndarray

    array_command_x: jnp.ndarray
    array_command_y: jnp.ndarray
    array_gp_prediction_x: jnp.ndarray
    array_gp_prediction_y: jnp.ndarray
    array_intent_x: jnp.ndarray
    array_intent_y: jnp.ndarray

    array_sensor_x: jnp.ndarray
    array_sensor_y: jnp.ndarray

    @staticmethod
    def init(max_capacity: int, state_dims: int) -> DataPoints:
        """Create a DataPoints from the float values."""

        # Init id to -1
        array_point_id = jnp.ones((max_capacity, 1)) * -1

        # Init timings to 0
        array_sensor_time_sec = jnp.zeros((max_capacity, 1))
        array_sensor_time_nanosec = jnp.zeros((max_capacity, 1))
        array_command_time_sec = jnp.zeros((max_capacity, 1))
        array_command_time_nanosec = jnp.zeros((max_capacity, 1))

        # Init everything else to -jnp.inf
        array_state = -jnp.inf * jnp.ones((max_capacity, state_dims))

        array_command_x = -jnp.inf * jnp.ones((max_capacity, 1))
        array_command_y = -jnp.inf * jnp.ones((max_capacity, 1))
        array_gp_prediction_x = -jnp.inf * jnp.ones((max_capacity, 1))
        array_gp_prediction_y = -jnp.inf * jnp.ones((max_capacity, 1))
        array_intent_x = -jnp.inf * jnp.ones((max_capacity, 1))
        array_intent_y = -jnp.inf * jnp.ones((max_capacity, 1))

        array_sensor_x = -jnp.inf * jnp.ones((max_capacity, 1))
        array_sensor_y = -jnp.inf * jnp.ones((max_capacity, 1))

        return DataPoints(
            max_capacity=max_capacity,
            current_size=0,
            array_point_id=array_point_id,
            array_sensor_time_sec=array_sensor_time_sec,
            array_sensor_time_nanosec=array_sensor_time_nanosec,
            array_command_time_sec=array_command_time_sec,
            array_command_time_nanosec=array_command_time_nanosec,
            array_state=array_state,
            array_command_x=array_command_x,
            array_command_y=array_command_y,
            array_gp_prediction_x=array_gp_prediction_x,
            array_gp_prediction_y=array_gp_prediction_y,
            array_intent_x=array_intent_x,
            array_intent_y=array_intent_y,
            array_sensor_x=array_sensor_x,
            array_sensor_y=array_sensor_y,
        )

    @staticmethod
    @jax.jit
    def add(
        datapoint: DataPoints,
        point_id: float,
        sensor_time_sec: float,
        sensor_time_nanosec: float,
        command_time_sec: float,
        command_time_nanosec: float,
        state: np.ndarray,
        command_x: float,
        command_y: float,
        gp_prediction_x: float,
        gp_prediction_y: float,
        intent_x: float,
        intent_y: float,
        sensor_x: float,
        sensor_y: float,
    ) -> DataPoints:
        """Add floats values to an existing DataPoints."""

        # New array
        current_size = datapoint.current_size
        array_point_id = datapoint.array_point_id.at[current_size].set(point_id)
        array_sensor_time_sec = datapoint.array_sensor_time_sec.at[current_size].set(
            sensor_time_sec
        )
        array_sensor_time_nanosec = datapoint.array_sensor_time_nanosec.at[
            current_size
        ].set(sensor_time_nanosec)
        array_command_time_sec = datapoint.array_command_time_sec.at[current_size].set(
            command_time_sec
        )
        array_command_time_nanosec = datapoint.array_command_time_nanosec.at[
            current_size
        ].set(command_time_nanosec)
        array_state = datapoint.array_state.at[current_size].set(state)
        array_command_x = datapoint.array_command_x.at[current_size].set(command_x)
        array_command_y = datapoint.array_command_y.at[current_size].set(command_y)
        array_gp_prediction_x = datapoint.array_gp_prediction_x.at[current_size].set(
            gp_prediction_x
        )
        array_gp_prediction_y = datapoint.array_gp_prediction_y.at[current_size].set(
            gp_prediction_y
        )
        array_intent_x = datapoint.array_intent_x.at[current_size].set(intent_x)
        array_intent_y = datapoint.array_intent_y.at[current_size].set(intent_y)
        array_sensor_x = datapoint.array_sensor_x.at[current_size].set(sensor_x)
        array_sensor_y = datapoint.array_sensor_y.at[current_size].set(sensor_y)

        # New size for next addition
        current_size = current_size + 1
        _, current_size = jnp.divmod(current_size, datapoint.max_capacity)

        return DataPoints(
            max_capacity=datapoint.max_capacity,
            current_size=current_size,
            array_point_id=array_point_id,
            array_sensor_time_sec=array_sensor_time_sec,
            array_sensor_time_nanosec=array_sensor_time_nanosec,
            array_command_time_sec=array_command_time_sec,
            array_command_time_nanosec=array_command_time_nanosec,
            array_state=array_state,
            array_command_x=array_command_x,
            array_command_y=array_command_y,
            array_gp_prediction_x=array_gp_prediction_x,
            array_gp_prediction_y=array_gp_prediction_y,
            array_intent_x=array_intent_x,
            array_intent_y=array_intent_y,
            array_sensor_x=array_sensor_x,
            array_sensor_y=array_sensor_y,
        )

    @staticmethod
    @jax.jit
    def filter(datapoint: DataPoints, datapoint_filter: jnp.ndarray) -> DataPoints:
        """Filter the datapoints."""

        array_point_id = jnp.where(datapoint_filter, datapoint.array_point_id, 1)

        array_sensor_time_sec = jnp.where(
            datapoint_filter, datapoint.array_sensor_time_sec, 0
        )
        array_sensor_time_nanosec = jnp.where(
            datapoint_filter, datapoint.array_sensor_time_nanosec, 0
        )
        array_command_time_sec = jnp.where(
            datapoint_filter, datapoint.array_command_time_sec, 0
        )
        array_command_time_nanosec = jnp.where(
            datapoint_filter, datapoint.array_command_time_nanosec, 0
        )

        array_point_id = jnp.where(datapoint_filter, datapoint.array_point_id, -jnp.inf)
        array_state = jnp.where(
            jnp.reshape(
                jnp.repeat(datapoint_filter, datapoint.array_state.shape[1], axis=0),
                datapoint.array_state.shape,
            ),
            datapoint.array_state,
            -jnp.inf,
        )
        array_command_x = jnp.where(
            datapoint_filter, datapoint.array_command_x, -jnp.inf
        )
        array_command_y = jnp.where(
            datapoint_filter, datapoint.array_command_y, -jnp.inf
        )
        array_gp_prediction_x = jnp.where(
            datapoint_filter, datapoint.array_gp_prediction_x, -jnp.inf
        )
        array_gp_prediction_y = jnp.where(
            datapoint_filter, datapoint.array_gp_prediction_y, -jnp.inf
        )
        array_intent_x = jnp.where(datapoint_filter, datapoint.array_intent_x, -jnp.inf)
        array_intent_y = jnp.where(datapoint_filter, datapoint.array_intent_y, -jnp.inf)
        array_sensor_x = jnp.where(datapoint_filter, datapoint.array_sensor_x, -jnp.inf)
        array_sensor_y = jnp.where(datapoint_filter, datapoint.array_sensor_y, -jnp.inf)

        return datapoint.replace(
            array_point_id=array_point_id,
            array_sensor_time_sec=array_sensor_time_sec,
            array_sensor_time_nanosec=array_sensor_time_nanosec,
            array_command_time_sec=array_command_time_sec,
            array_command_time_nanosec=array_command_time_nanosec,
            array_state=array_state,
            array_command_x=array_command_x,
            array_command_y=array_command_y,
            array_gp_prediction_x=array_gp_prediction_x,
            array_gp_prediction_y=array_gp_prediction_y,
            array_intent_x=array_intent_x,
            array_intent_y=array_intent_y,
            array_sensor_x=array_sensor_x,
            array_sensor_y=array_sensor_y,
        )
