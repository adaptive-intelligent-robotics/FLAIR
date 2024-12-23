# Written or Adapted by the Imperial College London Team for the FLAIR project, 2023
# Authors for this file:
# Maxime Allard
# Manon Flageat
# Bryan Lim
# Antoine Cully

import copy
import json
from typing import Any, Dict, Tuple

import numpy as np
from scipy.signal import correlate

################
# Organisation #

# This section aims to avoid indexing problems in the code
# WARNING: if you change it, make sure to update _update_buffers() accordingly

# Command buffer
COMMAND_INDEX_TIMESTAMP = 0
COMMAND_INDEX_TIME_SEC = 1
COMMAND_INDEX_TIME_NANOSEC = 2
COMMAND_INDEX_EVENT = 3
COMMAND_INDEX_INTENT_X = 4
COMMAND_INDEX_INTENT_Y = 5
COMMAND_INDEX_GP_PREDICTION_X = 6
COMMAND_INDEX_GP_PREDICTION_Y = 7
COMMAND_INDEX_COMMAND_X = 8
COMMAND_INDEX_COMMAND_Y = 9
COMMAND_INDEX_STATE_START = 10
COMMAND_INDEX_STATE_END = 19

# Sensor buffer
SENSOR_INDEX_TIMESTAMP = 0
SENSOR_INDEX_TIME_SEC = 1
SENSOR_INDEX_TIME_NANOSEC = 2
SENSOR_INDEX_SENSOR_X = 3
SENSOR_INDEX_SENSOR_Y = 4
SENSOR_INDEX_SENSOR_WX = 5
SENSOR_INDEX_SENSOR_WY = 6

##############
# Main Class #


class DataCollection:
    def __init__(
        self,
        logger: Any,
        filter_transition: bool,
        filter_varying_angle: bool,
        filter_turning_only: bool,
        buffer_size: int,
        min_delay: int,
        max_delay: int,
        selection_size: int,
        IQC_Q1: float,
        IQC_Qn: float,
        filter_transition_size: int,
        filter_transition_tolerance: float,
        filter_varying_angle_size: int,
        filter_varying_angle_tolerance: float,
        filter_turning_only_tolerance: float,
    ):
        """
        Initialise the data collection object. To understand this code, see:
        https://drive.google.com/file/d/1LskUJkZYTyaQqC9J2FpCJHSMmkqTbVUN/view?usp=sharing

        Args:
            logger
            filter_transition
            filter_varying_angle
            filter_turning_only
            buffer_size: size of the stored buffer in number of command points (T in slides)
            min_delay: used to limit the automatic delay computation to reasonable values (nanosec)
            max_delay: used to limit the automatic delay computation to reasonable values (nanosec)
            selection_size: how many point in the past to select from (M in slides)
            IQC_Q1: pourcentage considered as first quantile
            IQC_Qn: pourcentage considered as last quantile
            filter_transition_size: number of points in the past used to filter transition (L in slides)
            filter_transition_tolerance: tolerance used to filter transition (E in slides)
            filter_varying_angle_size
            filter_varying_angle_tolerance
            filter_turning_only_tolerance
        """

        self._logger = logger
        self.downsampling_N = 1
        self.filter_transition = filter_transition
        self.filter_varying_angle = filter_varying_angle
        self.filter_turning_only = filter_turning_only
        self.buffer_size_T = buffer_size
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.selection_size_M = selection_size
        self.IQC_Q1 = IQC_Q1
        self.IQC_Qn = IQC_Qn
        self.filter_transition_size_L = filter_transition_size
        self.filter_transition_tolerance_E = filter_transition_tolerance
        self.filter_varying_angle_size_L = filter_varying_angle_size
        self.filter_varying_angle_tolerance_E = filter_varying_angle_tolerance
        self.filter_turning_only_tolerance_E = filter_turning_only_tolerance

        self.reset()

    def reset(self):
        """Reset the data collection."""
        self.initialised = False
        self.point_count = 0
        self.datapoint_id = 0
        self.delay_x = self.min_delay
        self.delay_y = self.min_delay

    def _stamp_to_int(self, stamp: Any) -> int:
        """
        Transform  msg time stamp into an int.

        WARNING: please do not modify this operation if you do not test it in details,
        there is a lot of rounding problems with this operation.
        """
        if stamp is None:
            return stamp
        if isinstance(stamp, int):
            return stamp
        return int(stamp.sec) * int(1e9) + int(stamp.nanosec)

    def _int_to_stamp(self, stamp_int: Any, empty_stamp: Any) -> Any:
        """
        Transform int time stamp into msg.

        WARNING: please do not modify this operation if you do not test it in details,
        there is a lot of rounding problems with this operation.
        """
        if stamp_int is None:
            return stamp_int
        if isinstance(stamp_int, type(empty_stamp)):
            return stamp_int
        stamp = copy.deepcopy(empty_stamp)
        floor = int(np.floor(stamp_int / int(1e9)))
        stamp.sec = floor
        stamp.nanosec = int(stamp_int - floor * int(1e9))
        return stamp

    def data_collection(
        self,
        state: Any,
        command_x: float,
        command_y: float,
        gp_prediction_x: float,
        gp_prediction_y: float,
        intent_x: float,
        intent_y: float,
        sensor_time: np.array,
        sensor_x: np.array,
        sensor_y: np.array,
        sensor_wx: np.array,
        sensor_wy: np.array,
    ) -> Tuple[Tuple, Tuple, Any, int]:
        """
        Main data collection function.

        Args:
            state
            command_x
            command_y
            gp_prediction_x
            gp_prediction_y
            intent_x
            intent_y
            sensor_time
            sensor_x
            sensor_y
            sensor_wx
            sensor_wy

        Returns:
            the final datapoint to send to GP Training.
            the final datapoint to send to influx.
            a json file containing metrics to send to influx.
            an error code to send to influx.
        """

        ######################################
        # CASE 1: if nan in data, do not add #
        if (
            command_x is None
            or command_y is None
            or command_x != command_x
            or command_y != command_y
            or intent_x != intent_x
            or intent_y != intent_y
            or (sensor_x != sensor_x).any()
            or (sensor_y != sensor_y).any()
        ):
            self._logger.debug("NO DATAPOINT: Nan values.")
            return None, None, None, 1 # Error code for influx

        # Add current command and sensor to buffers
        self._update_buffers(
            state=state,
            command_x=command_x,
            command_y=command_y,
            gp_prediction_x=gp_prediction_x,
            gp_prediction_y=gp_prediction_y,
            intent_x=intent_x,
            intent_y=intent_y,
            sensor_time=sensor_time,
            sensor_x=sensor_x,
            sensor_y=sensor_y,
            sensor_wx=sensor_wx,
            sensor_wy=sensor_wy,
        )

        # Count total number of points since last added point
        self.point_count += 1

        ##########################################################
        # CASE 2: if less than downsampling_N points, do not add #
        if self.point_count < self.downsampling_N:
            # self._logger.debug("NO DATAPOINT: Downsampling command frequency.")
            return None, None, None, 2 # Error code for influx

        # Compute interval to select from
        (
            delay_metrics,
            point_sensor_x_select_start,
            point_sensor_x_select_end,
            point_sensor_y_select_start,
            point_sensor_y_select_end,
            point_command_select_start,
            point_command_select_end,
        ) = self._compute_interval()

        ##########################################################
        # CASE 3: if no data for considered interval, do not add #
        if point_sensor_x_select_start is None or point_command_select_start is None:
            return None, None, None, 3 # Error code for influx

        ###############################################
        # CASE 4: if filtering transition, do not add #
        if self.filter_transition:

            # Compute the interval to check
            only_command_bufer = self.buffer_command[:point_command_select_start, :]
            only_command_bufer = only_command_bufer[only_command_bufer[:, COMMAND_INDEX_EVENT] == 1, :]
            only_command_bufer = only_command_bufer[-self.filter_transition_size_L:, :]

            # Check that change in command is within tolerance on the interval
            command_x_change = np.max(only_command_bufer[:, COMMAND_INDEX_COMMAND_X]) \
                - np.min(only_command_bufer[:, COMMAND_INDEX_COMMAND_X])
            command_y_change = np.max(only_command_bufer[:, COMMAND_INDEX_COMMAND_Y]) \
                - np.min(only_command_bufer[:, COMMAND_INDEX_COMMAND_Y])
            if (command_x_change > self.filter_transition_tolerance_E) or \
               (command_y_change > self.filter_transition_tolerance_E):
                self._logger.debug(f"NO DATAPOINT: filtering transition.")
                return None, None, None, 4 # Error code for influx

        ##################################################
        # CASE 5: if filtering varying angle, do not add #
        if self.filter_varying_angle:

            # Compute the interval to check
            constant_sensor_x_start = (
                point_sensor_x_select_start - self.filter_varying_angle_size_L
            )
            constant_sensor_x_start = max(0, constant_sensor_x_start)  # sanity check
            constant_sensor_x_buffer = self.buffer_sensor[
                constant_sensor_x_start:point_sensor_x_select_end, :
            ]
            constant_sensor_y_start = (
                point_sensor_y_select_start - self.filter_varying_angle_size_L
            )
            constant_sensor_y_start = max(0, constant_sensor_y_start)  # sanity check
            constant_sensor_y_buffer = self.buffer_sensor[
                constant_sensor_y_start:point_sensor_y_select_end, :
            ]

            # Check that angle speeds are within tolerance on the interval
            if (
                (constant_sensor_x_buffer[:, SENSOR_INDEX_SENSOR_WX] > self.filter_varying_angle_tolerance_E).any()
                or (constant_sensor_x_buffer[:, SENSOR_INDEX_SENSOR_WY] > self.filter_varying_angle_tolerance_E).any()
                or (constant_sensor_y_buffer[:, SENSOR_INDEX_SENSOR_WX] > self.filter_varying_angle_tolerance_E).any()
                or (constant_sensor_y_buffer[:, SENSOR_INDEX_SENSOR_WY] > self.filter_varying_angle_tolerance_E).any()
            ):
                self._logger.debug(f"NO DATAPOINT: filtering varying angle.")
                return None, None, None, 5 # Error code for influx

        ###############################################
        # CASE 6: if filtering turning only, do not add #
        if self.filter_turning_only:
            current_command_x = self.buffer_command[point_command_select_start, COMMAND_INDEX_COMMAND_X]
            current_command_y = self.buffer_command[point_command_select_start, COMMAND_INDEX_COMMAND_Y]

            if (
                current_command_x < self.filter_turning_only_tolerance_E 
                and current_command_x > -self.filter_turning_only_tolerance_E 
                and current_command_y != 0.0
            ):
                self._logger.debug("NO DATAPOINT: turning only.")
                return None, None, None, 6 # Error code for influx

        ########################################
        # CASE 7: if none of the previous, add #

        # Reset counter of points since last added point
        self.point_count = 0

        # Select the final point to add for both command and sensor
        (
            final_sensor_x_point,
            final_sensor_y_point,
            final_command_point,
        ) = self._final_point(
            point_sensor_x_select_start=point_sensor_x_select_start,
            point_sensor_x_select_end=point_sensor_x_select_end,
            point_sensor_y_select_start=point_sensor_y_select_start,
            point_sensor_y_select_end=point_sensor_y_select_end,
            point_command_select_start=point_command_select_start,
            point_command_select_end=point_command_select_end,
        )

        # Create datapoints
        final_datapoint = (
            self.datapoint_id,  # point_id
            self.buffer_sensor[final_sensor_x_point, SENSOR_INDEX_TIME_SEC],  # sensor_time_sec
            self.buffer_sensor[final_sensor_x_point, SENSOR_INDEX_TIME_NANOSEC],  # sensor_time_nanosec
            self.buffer_command[final_command_point, COMMAND_INDEX_TIME_SEC],  # command_time_sec
            self.buffer_command[final_command_point, COMMAND_INDEX_TIME_NANOSEC],  # command_time_nanosec
            self.buffer_command[final_command_point, COMMAND_INDEX_GP_PREDICTION_X],  # gp_prediction_x
            self.buffer_command[final_command_point, COMMAND_INDEX_GP_PREDICTION_Y],  # gp_prediction_y
            self.buffer_command[final_command_point, COMMAND_INDEX_COMMAND_X],  # command_x
            self.buffer_command[final_command_point, COMMAND_INDEX_COMMAND_Y],  # command_y
            self.buffer_command[final_command_point, COMMAND_INDEX_INTENT_X],  # intent_x
            self.buffer_command[final_command_point, COMMAND_INDEX_INTENT_Y],  # intent_y
            self.buffer_sensor[final_sensor_x_point, SENSOR_INDEX_SENSOR_X],  # sensor_x
            self.buffer_sensor[final_sensor_y_point, SENSOR_INDEX_SENSOR_Y],  # sensor_y
            1,  # num_accumulate
            self.buffer_command[final_command_point, COMMAND_INDEX_STATE_START:COMMAND_INDEX_STATE_END],  # state
        )

        # Create corresponding msg for Influx
        final_datapoints_msg = [
            ("Datapoint_ID", final_datapoint[0]),
            ("GP_time_sensor_sec", int(final_datapoint[1])),
            ("GP_time_sensor_nanosec", int(final_datapoint[2])),
            ("GP_time_command_sec", int(final_datapoint[3])),
            ("GP_time_command_nanosec", int(final_datapoint[4])),
            ("FC_sampled_command_x", final_datapoint[5]),
            ("FC_sampled_command_y", final_datapoint[6]),
            ("FC_executed_x", final_datapoint[7]),
            ("FC_executed_y", final_datapoint[8]),
            ("Human_command_x", final_datapoint[9]),
            ("Human_command_y", final_datapoint[10]),
            ("Sensor_x", final_datapoint[11]),
            ("Sensor_y", final_datapoint[12]),
            ("FC_command_error_x", float(final_datapoint[11] - final_datapoint[7])),
            ("FC_command_error_y", float(final_datapoint[12] - final_datapoint[8])),
            ("FC_intent_error_x", float(final_datapoint[11] - final_datapoint[9])),
            ("FC_intent_error_y", float(final_datapoint[12] - final_datapoint[10])),
        ] + [("State_{}".format(i), e) for i, e in enumerate(final_datapoint[14])]

        # Increase counters
        self.datapoint_id += 1

        # Create metrics for Influx
        try:
            metrics = self._build_metrics(
                delay_metrics=delay_metrics,
                final_sensor_x_point=final_sensor_x_point,
                final_sensor_y_point=final_sensor_y_point,
                final_command_point=final_command_point,
                point_command_select_start=point_command_select_start,
                point_command_select_end=point_command_select_end,
                point_sensor_x_select_start=point_sensor_x_select_start,
                point_sensor_x_select_end=point_sensor_x_select_end,
                point_sensor_y_select_start=point_sensor_y_select_start,
                point_sensor_y_select_end=point_sensor_y_select_end,
            )
        except BaseException as e:
            metrics = None
            self._logger.debug("Influx metrics computation failed.")
            self._logger.debug(e)

        # Return final data points, msg and metrics for Influx
        return final_datapoint, final_datapoints_msg, metrics, 0 # Error code for influx

        ##############################################
        # Point temporal accumulation - kept in case #

        # Point accumulation parameters
        # Input: (command_x, command_y, msg.vx, msg.vy, msg.wz, msg.tx, msg.ty, msg.yaw, msg.roll, msg.pitch)
        # input_same_threshold = (0.01, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05)

        # Compare previous and current input
        # current_input = self.buffer_command[final_command_point, COMMAND_INDEX_COMMAND_X:]
        # if self.accumulate_input and self.previous_input is not None:

        #    # Input similarity
        #    same_inputs = True
        #    for i in range(len(current_input)):
        #        same_inputs = same_inputs and (
        #            abs(self.previous_input[i] - current_input[i]) < input_same_threshold[i-2]
        #        )
        #
        #    # If the input are the same, accumulate
        #    if same_inputs:
        #        self.num_accumulate += 1

        #        # Average accumulated sensor readings
        #        if self.mean_sensor_x is None or self.mean_sensor_y is None:
        #            self.mean_sensor_x = self.buffer_sensor[final_sensor_x_point, SENSOR_INDEX_SENSOR_X]
        #            self.mean_sensor_y = self.buffer_sensor[final_sensor_y_point, SENSOR_INDEX_SENSOR_Y]
        #        else:
        #            self.mean_sensor_x = (
        #                self.mean_sensor_x +
        #                (self.buffer_sensor[final_sensor_x_point, SENSOR_INDEX_SENSOR_X] - self.mean_sensor_x)
        #                / self.num_accumulate
        #            )
        #            self.mean_sensor_y = (
        #                self.mean_sensor_y +
        #                (self.buffer_sensor[final_sensor_y_point, SENSOR_INDEX_SENSOR_Y] - self.mean_sensor_y)
        #                / self.num_accumulate
        #            )

        #        # Reset counter
        #        self.point_count = 0

        #        # Return
        #        self._logger.debug(f"NO DATAPOINT: Accumulating. Total accumulated points: {self.num_accumulate}")
        #        return None, None, None

        ## If no accumulation, take actual value of sensor reading
        # if self.mean_sensor_x is None or self.mean_sensor_y is None:
        #    mean_sensor_x = self.buffer_sensor[final_sensor_x_point, SENSOR_INDEX_SENSOR_X]
        #    mean_sensor_y = self.buffer_sensor[final_sensor_y_point, SENSOR_INDEX_SENSOR_Y]
        ## If accumulation, use accumulated value
        # else:
        #    mean_sensor_x = self.mean_sensor_x
        #    mean_sensor_y = self.mean_sensor_y

        # Reset attributes
        # self.num_accumulate = 1
        # self.previous_input = current_input
        # self.mean_sensor_x = None
        # self.mean_sensor_y = None

    def _build_metrics(
        self,
        delay_metrics: Dict,
        final_sensor_x_point: int,
        final_sensor_y_point: int,
        final_command_point: int,
        point_command_select_start: int,
        point_command_select_end: int,
        point_sensor_x_select_start: int,
        point_sensor_x_select_end: int,
        point_sensor_y_select_start: int,
        point_sensor_y_select_end: int,
    ) -> Any:
        """
        Create the json file of metrics for Influx to debug DataCollection.
        """
        json_file = {
            "Command_time": self.buffer_command[:, COMMAND_INDEX_TIMESTAMP].tolist(),
            "Sensor_time": self.buffer_sensor[:, SENSOR_INDEX_TIMESTAMP].tolist(),
            "Sensor_x_delayed_time": (
                self.buffer_sensor[:, SENSOR_INDEX_TIMESTAMP]
                - float(delay_metrics["delay_x"])
            ).tolist(),
            "Sensor_y_delayed_time": (
                self.buffer_sensor[:, SENSOR_INDEX_TIMESTAMP]
                - float(delay_metrics["delay_y"])
            ).tolist(),
            "Delay_x": float(delay_metrics["delay_x"]),
            "Delay_y": float(delay_metrics["delay_y"]),
            "Cross_correlation_x": delay_metrics["cross_correlation_x"].tolist(),
            "Cross_correlation_y": delay_metrics["cross_correlation_y"].tolist(),
            "Command_x": self.buffer_command[:, COMMAND_INDEX_COMMAND_X].tolist(),
            "Command_y": self.buffer_command[:, COMMAND_INDEX_COMMAND_Y].tolist(),
            "Sensor_x": self.buffer_sensor[:, SENSOR_INDEX_SENSOR_X].tolist(),
            "Sensor_y": self.buffer_sensor[:, SENSOR_INDEX_SENSOR_Y].tolist(),
            "Command_add": float(final_command_point),
            "Sensor_x_add": float(final_sensor_x_point),
            "Sensor_y_add": float(final_sensor_y_point),
            "Command_select_start": float(point_command_select_start),
            "Command_select_end": float(point_command_select_end),
            "Sensor_x_select_start": float(point_sensor_x_select_start),
            "Sensor_x_select_end": float(point_sensor_x_select_end),
            "Sensor_y_select_start": float(point_sensor_y_select_start),
            "Sensor_y_select_end": float(point_sensor_y_select_end),
        }
        return json.dumps(json_file)

    def _update_buffers(
        self,
        state: Any,
        command_x: float,
        command_y: float,
        gp_prediction_x: float,
        gp_prediction_y: float,
        intent_x: float,
        intent_y: float,
        sensor_time: np.array,
        sensor_x: np.array,
        sensor_y: np.array,
        sensor_wx: np.array,
        sensor_wy: np.array,
    ) -> None:
        """
        Function to update the content of all the buffers.
        """

        # Convert sensor_time
        timestamp = np.vectorize(self._stamp_to_int)(sensor_time)
        sensor_time_sec = np.vectorize(lambda x: x.sec)(sensor_time)
        sensor_time_nanosec = np.vectorize(lambda x: x.nanosec)(sensor_time)

        # Create arrays for buffer
        command_x = np.full_like(sensor_x, command_x)
        command_y = np.full_like(sensor_x, command_y)
        gp_prediction_x = np.full_like(sensor_x, gp_prediction_x)
        gp_prediction_y = np.full_like(sensor_x, gp_prediction_y)
        intent_x = np.full_like(sensor_x, intent_x)
        intent_y = np.full_like(sensor_x, intent_y)
        event = np.zeros_like(sensor_x)
        event[0] = 1

        # New entry for each array
        command = np.concatenate(
            [
                timestamp,
                sensor_time_sec,
                sensor_time_nanosec,
                event,
                intent_x,
                intent_y,
                gp_prediction_x,
                gp_prediction_y,
                command_x,
                command_y,
                state,
            ],
            axis=1,
        )
        sensor = np.concatenate(
            [
                timestamp,
                sensor_time_sec,
                sensor_time_nanosec,
                sensor_x,
                sensor_y,
                sensor_wx,
                sensor_wy,
            ],
            axis=1,
        )

        # If buffers are not initialised
        if not self.initialised:
            self.buffer_command = command
            self.buffer_sensor = sensor
            self.initialised = True

        # If buffers are already initialised
        else:
            self.buffer_command = np.append(self.buffer_command, command, axis=0)
            self.buffer_sensor = np.append(self.buffer_sensor, sensor, axis=0)

            # Remove from buffer everything older than self.buffer_size_T points
            buffer_filter = (
                np.sum(self.buffer_command[:, COMMAND_INDEX_EVENT])
                - np.cumsum(self.buffer_command[:, COMMAND_INDEX_EVENT])
            ) < self.buffer_size_T
            # self._logger.debug(f"Filter for self.buffer_size_T: {buffer_filter}")
            self.buffer_command = self.buffer_command[buffer_filter]
            self.buffer_sensor = self.buffer_sensor[buffer_filter]

        assert self.buffer_command.shape[0] == self.buffer_sensor.shape[0]

        return

    def _compute_min_max_delay(self, index_command: int, index_sensor: int) -> float:
        """
        UNUSED - KEPT IN CASE

        Compute the delay between self.buffer_command and self.buffer_sensor using
        the min-max method.
        Args:
            index_command: index in self.buffer_command of the command to use, allow
                to choose if computing delay_x or delay_y.
            index_sensor: index in self.buffer_sensor of the sensor to use, allow
                to choose if computing delay_x or delay_y.
        Returns:
            the delay value
        """
        num_point_delay_J = 3

        # Find the num_point_delay_J min and max of sensor and command
        max_command = self.buffer_command[:, index_command].argsort()[
            -num_point_delay_J:
        ][::-1]
        min_command = self.buffer_command[:, index_command].argsort()[
            :num_point_delay_J
        ]
        max_sensor = self.buffer_sensor[:, index_sensor].argsort()[-num_point_delay_J:][
            ::-1
        ]
        min_sensor = self.buffer_sensor[:, index_sensor].argsort()[:num_point_delay_J]

        # Compute all the corresponding delays
        delays = np.array(
            [
                [
                    self.buffer_sensor[max_sensor[i], SENSOR_INDEX_TIMESTAMP]
                    - self.buffer_command[max_command[i], COMMAND_INDEX_TIMESTAMP],
                    self.buffer_sensor[min_sensor[i], SENSOR_INDEX_TIMESTAMP]
                    - self.buffer_command[min_command[i], COMMAND_INDEX_TIMESTAMP],
                ]
                for i in range(num_point_delay_J)
            ]
        ).flatten()

        if len(delays) == 0:
            return self.min_delay

        # Filtering negative delays
        # delays = delays[delays > 0]

        # Clipping delays
        delays = np.clip(delays, self.min_delay, self.max_delay)

        # Take the median of the list as the delay
        return np.median(delays)

    def _compute_cross_correlation_delay(
        self,
        index_command: int,
        index_sensor: int,
        point_command_select_start: int,
        point_command_select_end: int,
    ) -> Tuple[float, int, int, np.ndarray]:
        """
        Compute the delay between self.buffer_command and self.buffer_sensor using
        the cross-correlation method.
        Args:
            index_command: index in self.buffer_command of the command to use, allow
                to choose if computing delay_x or delay_y.
            index_sensor: index in self.buffer_sensor of the sensor to use, allow
                to choose if computing delay_x or delay_y.
            point_command_select_start
            point_command_select_end
        Returns:
            the delay value
            point_sensor_select_start
            point_sensor_select_end
            cross_correlation: the cross-correlation used to compute the matching
        """

        # Compute cross correlation
        cross_correlation = correlate(
            self.buffer_command[:, index_command], self.buffer_sensor[:, index_sensor]
        )

        # Use argmax of correlation to compute delay
        index_delay = np.argmax(cross_correlation) - self.buffer_command.shape[0]

        # Find corresponding points
        try:
            point_sensor_select_start = point_command_select_start - index_delay
            point_sensor_select_end = point_command_select_end - index_delay
            point_sensor_select_start = min(
                max(point_sensor_select_start, 0), self.buffer_sensor.shape[0] - 1
            )
            point_sensor_select_end = min(
                max(point_sensor_select_end, 0), self.buffer_sensor.shape[0] - 1
            )
        except BaseException as e:
            point_sensor_select_start = 0
            point_sensor_select_end = 0
            self._logger.debug(f"Cross-correlation delay computation failed: {e}")

        # Compute corresponding delay
        delay = (
            self.buffer_sensor[point_sensor_select_start, SENSOR_INDEX_TIMESTAMP]
            - self.buffer_command[point_command_select_start, COMMAND_INDEX_TIMESTAMP]
        )

        return (
            delay,
            point_sensor_select_start,
            point_sensor_select_end,
            cross_correlation,
        )

    def _compute_default_interval(
        self,
        delay: float,
        point_command_select_start: int,
        point_command_select_end: int,
    ) -> Tuple[float, int, int]:
        """
        Compute the intervals of points to consider for selection for a given delay.

        Args:
            point_command_select_start: the interval start in self.buffer_command.
            point_command_select_end: the interval end in self.buffer_command.

        Returns:
            delay: the delay value.
            point_sensor_select_start: the interval start in self.buffer_sensor.
            point_sensor_select_end: the interval end in self.buffer_sensor.
        """

        try:
            # Range of possible point values
            min_point = 0
            max_point = self.buffer_sensor.shape[0] - 1

            # Some constants for computation
            delayed_sensor = self.buffer_sensor[:, SENSOR_INDEX_TIMESTAMP] - delay
            time_command_select_start = self.buffer_command[point_command_select_start, COMMAND_INDEX_TIMESTAMP]
            time_command_select_end = self.buffer_command[point_command_select_end, COMMAND_INDEX_TIMESTAMP]

            # All sensor point after the start of the command interval
            buffer_after_start = delayed_sensor >= time_command_select_start

            # Start point is the first one of them
            point_sensor_select_start = np.flatnonzero(buffer_after_start)[0]
            point_sensor_select_start = min(max(point_sensor_select_start, min_point), max_point)

            # All sensor point before the end of the command interval
            buffer_before_end = delayed_sensor <= time_command_select_end

            # End point is the last one of them
            point_sensor_select_end = np.flatnonzero(buffer_before_end)[-1]
            point_sensor_select_end = min(max(point_sensor_select_end, min_point), max_point)

        except BaseException as e:
            # Just some security just in case something goes really wrong 
            self._logger.debug(f"Constant delay computation failed: {e}")
            point_sensor_select_start = 0
            point_sensor_select_end = 0

        return delay, point_sensor_select_start, point_sensor_select_end

    def _compute_interval(self) -> Tuple[Dict, int, int, int, int, int, int]:
        """
        Compute the current delay between self.buffer_command and self.buffer_sensor, and the
        corresponding intervals of points to consider for selection.

        Args:
            None

        Returns:
            delay_metrics: some metrics used by delay computation to send to Inlfux.
            point_sensor_x_select_start: the interval start in self.buffer_sensor.
            point_sensor_x_select_end: the interval end in self.buffer_sensor.
            point_sensor_y_select_start: the interval start in self.buffer_sensor.
            point_sensor_y_select_end: the interval end in self.buffer_sensor.
            point_command_select_start: the interval start in self.buffer_command.
            point_command_select_end: the interval end in self.buffer_command.
        """

        # If not enough datapoints in buffers yet, return None
        point_command_select = np.sum(
            self.buffer_command[:, COMMAND_INDEX_EVENT]
        ) - np.cumsum(self.buffer_command[:, COMMAND_INDEX_EVENT])
        if np.all(point_command_select < (self.selection_size_M + 1)):
            self._logger.debug(f"NO DATAPOINT: not enough datapoints yet.")
            return {}, None, None, None, None, None, None

        # Select the command points for which we choose a datapoint
        point_command_select = np.nonzero(
            point_command_select == self.selection_size_M
        )[0]
        point_command_select_start = point_command_select[0]
        point_command_select_end = point_command_select[-1]

        # If the interval for command points is empty, return None
        buffer_command_select = self.buffer_command[point_command_select_start:point_command_select_end, :]
        if buffer_command_select.shape[0] == 0:
            self._logger.debug(f"NO DATAPOINT: Empty command interval.")
            return {}, None, None, None, None, None, None

        # Create dict for delay_metrics
        delay_metrics = {
            "cross_correlation_x": np.array([0]),
            "cross_correlation_y": np.array([0]),
        }

        # First, delay X
        # If the command stay constant all along, use default delay value
        if all(
            self.buffer_command[:, COMMAND_INDEX_COMMAND_X]
            == self.buffer_command[0, COMMAND_INDEX_COMMAND_X]
        ):
            (
                delay_x,
                point_sensor_x_select_start,
                point_sensor_x_select_end,
            ) = self._compute_default_interval(
                self.min_delay,
                point_command_select_start,
                point_command_select_end,
            )
            delay_type = "constant"

        # Else, use cross-correlation
        else:
            (
                delay_x,
                point_sensor_x_select_start,
                point_sensor_x_select_end,
                cross_correlation_x,
            ) = self._compute_cross_correlation_delay(
                COMMAND_INDEX_COMMAND_X,
                SENSOR_INDEX_SENSOR_X,
                point_command_select_start,
                point_command_select_end,
            )
            delay_metrics["cross_correlation_x"] = cross_correlation_x
            delay_type = "cross"

        # If delay too big or too small, probably something wrong, reset it
        if delay_x < self.min_delay:
            # self._logger.debug(f"Setting too small delay_x {delay_x} < {self.min_delay} to self.min_delay.")
            (
                delay_x,
                point_sensor_x_select_start,
                point_sensor_x_select_end,
            ) = self._compute_default_interval(
                self.min_delay,
                point_command_select_start,
                point_command_select_end,
            )
            delay_type = "constant min_delay"
        elif delay_x > self.max_delay:
            # self._logger.debug(f"Setting too big delay_x {delay_x} > {self.max_delay} to self.max_delay.")
            (
                delay_x,
                point_sensor_x_select_start,
                point_sensor_x_select_end,
            ) = self._compute_default_interval(
                self.max_delay,
                point_command_select_start,
                point_command_select_end,
            )
            delay_type = "constant max_delay"

        # If the interval for sensor points is empty, return None
        buffer_sensor_select_x = self.buffer_sensor[point_sensor_x_select_start:point_sensor_x_select_end, :]
        if buffer_sensor_select_x.shape[0] == 0:
            self._logger.debug(f"NO DATAPOINT: Empty sensor x interval - {delay_type} delay.")
            return {}, None, None, None, None, None, None

        # Second, delay Y
        # If the command stay constant all along, use default delay value
        if all(
            self.buffer_command[:, COMMAND_INDEX_COMMAND_Y]
            == self.buffer_command[0, COMMAND_INDEX_COMMAND_Y]
        ):
            (
                delay_y,
                point_sensor_y_select_start,
                point_sensor_y_select_end,
            ) = self._compute_default_interval(
                self.min_delay,
                point_command_select_start,
                point_command_select_end,
            )
            delay_type = "constant"

        # Else, use cross-correlation
        else:
            (
                delay_y,
                point_sensor_y_select_start,
                point_sensor_y_select_end,
                cross_correlation_y,
            ) = self._compute_cross_correlation_delay(
                COMMAND_INDEX_COMMAND_Y,
                SENSOR_INDEX_SENSOR_Y,
                point_command_select_start,
                point_command_select_end,
            )
            delay_metrics["cross_correlation_y"] = cross_correlation_y
            delay_type = "cross"

        # If delay too big or too small, probably something wrong, reset it
        if delay_y < self.min_delay:
            # self._logger.debug(f"Setting too small delay_y {delay_y} < {self.min_delay} to self.min_delay.")
            (
                delay_y,
                point_sensor_y_select_start,
                point_sensor_y_select_end,
            ) = self._compute_default_interval(
                self.min_delay,
                point_command_select_start,
                point_command_select_end,
            )
            delay_type = "constant min_delay"
        elif delay_y > self.max_delay:
            # self._logger.debug(f"Setting too big delay_y {delay_y} > {self.max_delay} to self.max_delay.")
            (
                delay_y,
                point_sensor_y_select_start,
                point_sensor_y_select_end,
            ) = self._compute_default_interval(
                self.max_delay,
                point_command_select_start,
                point_command_select_end,
            )
            delay_type = "constant max_delay"

        # If the interval for sensor points is empty, return None
        buffer_sensor_select_y = self.buffer_sensor[point_sensor_y_select_start:point_sensor_y_select_end, :]
        if buffer_sensor_select_y.shape[0] == 0:
            self._logger.debug(f"NO DATAPOINT: Empty sensor y interval - {delay_type} delay.")
            return {}, None, None, None, None, None, None

        # Update delay value
        delay_metrics["delay_x"] = delay_x
        delay_metrics["delay_y"] = delay_y
        self.delay_x = delay_x
        self.delay_y = delay_y

        return (
            delay_metrics,
            point_sensor_x_select_start,
            point_sensor_x_select_end,
            point_sensor_y_select_start,
            point_sensor_y_select_end,
            point_command_select_start,
            point_command_select_end,
        )

    def _moving_median_final_point(self, sub_buffer_sensor: np.ndarray) -> int:
        """
        UNUSED - KEPT IN CASE

        Select the final point in a sub-buffer as the point with smallest distance to others.

        Args:
            sub_buffer_sensor: sub-buffer to select from.
        Returns:
            final_point: the final point in sub_buffer_sensor.
        """

        distances = np.abs(np.subtract.outer(sub_buffer_sensor, sub_buffer_sensor))
        distances = np.sum(distances, axis=1) / np.count_nonzero(distances, axis=1)
        return np.argmin(distances)

    def _IQC_final_point(
        self, sub_buffer_sensor: np.ndarray, sub_buffer_command: np.ndarray
    ) -> int:
        """
        Select the final point in a sub-buffer as Inter-Quantile Closest (IQC).

        Args:
            sub_buffer_sensor: sub-buffer to select from.
            sub_buffer_command: in case necessary for computation.

        Returns:
            final_point: the final point in sub_buffer_sensor.
        """

        # Get the values corresponding to quantile self.IQC_Q1 and self.IQC_Qn
        filtered_buffer_sensor = sub_buffer_sensor.copy()
        q1_sensor = np.quantile(filtered_buffer_sensor, self.IQC_Q1)
        q3_sensor = np.quantile(filtered_buffer_sensor, self.IQC_Qn)

        # Filter the values by setting to a really high value
        filtered_buffer_sensor = np.where(
            filtered_buffer_sensor < q1_sensor, 1000, filtered_buffer_sensor
        )
        filtered_buffer_sensor = np.where(
            filtered_buffer_sensor > q3_sensor, 1000, filtered_buffer_sensor
        )

        # Find in the filtered bugger the index of the closest point
        distances = np.abs(filtered_buffer_sensor - sub_buffer_command)
        return np.argmin(distances)

    def _final_point(
        self,
        point_sensor_x_select_start: int,
        point_sensor_x_select_end: int,
        point_sensor_y_select_start: int,
        point_sensor_y_select_end: int,
        point_command_select_start: int,
        point_command_select_end: int,
    ) -> Tuple[int, int, int]:
        """
        Select the final points to add.

        Args:
            point_sensor_x_select_start: the interval start in self.buffer_sensor.
            point_sensor_x_select_end: the interval end in self.buffer_sensor.
            point_sensor_y_select_start: the interval start in self.buffer_sensor.
            point_sensor_y_select_end: the interval end in self.buffer_sensor.
            point_command_select_start: the interval start in self.buffer_command.
            point_command_select_end: the interval end in self.buffer_command.

        Returns:
            final_sensor_x_point: the final point in self.buffer_sensor.
            final_sensor_y_point: the final point in self.buffer_sensor.
            final_command_point: the final point in self.buffer_command.
        """
        final_sensor_x_point = 0
        final_sensor_y_point = 0
        final_command_point = 0

        # Extract sub-buffer to select from
        sub_buffer_sensor_x = self.buffer_sensor[
            point_sensor_x_select_start:point_sensor_x_select_end, SENSOR_INDEX_SENSOR_X
        ]
        sub_buffer_sensor_y = self.buffer_sensor[
            point_sensor_y_select_start:point_sensor_y_select_end, SENSOR_INDEX_SENSOR_Y
        ]

        # For command copy same value as many times as necessary
        sub_buffer_command_x = np.repeat(
            self.buffer_command[
                point_command_select_start : point_command_select_start + 1,
                COMMAND_INDEX_COMMAND_X,
            ],
            sub_buffer_sensor_x.shape[0],
            axis=0,
        )
        sub_buffer_command_y = np.repeat(
            self.buffer_command[
                point_command_select_start : point_command_select_start + 1,
                COMMAND_INDEX_COMMAND_Y,
            ],
            sub_buffer_sensor_y.shape[0],
            axis=0,
        )

        # Select
        final_point_x = self._IQC_final_point(
            sub_buffer_sensor=sub_buffer_sensor_x,
            sub_buffer_command=sub_buffer_command_x,
        )
        final_point_y = self._IQC_final_point(
            sub_buffer_sensor=sub_buffer_sensor_y,
            sub_buffer_command=sub_buffer_command_y,
        )

        # Transform final_point in selection buffers into final point in main buffer
        final_sensor_x_point = final_point_x + point_sensor_x_select_start
        final_sensor_y_point = final_point_y + point_sensor_y_select_start
        final_command_point = point_command_select_start

        return final_sensor_x_point, final_sensor_y_point, final_command_point

        # KEPT IN CASE

        # # If using moving-median-based selection
        # if self.point_selection == MOVING_MEDIAN:

        #     # Extract sub-buffer to select from
        #     sub_buffer_sensor_x = self.buffer_sensor[
        #         point_sensor_x_select_start:point_sensor_x_select_end, SENSOR_INDEX_SENSOR_X,
        #     ]
        #     sub_buffer_sensor_y = self.buffer_sensor[
        #         point_sensor_y_select_start:point_sensor_y_select_end, SENSOR_INDEX_SENSOR_Y,
        #     ]

        #     # Select
        #     final_point_x = self._moving_median_final_point(
        #         sub_buffer_sensor=sub_buffer_sensor_x
        #     )
        #     final_point_y = self._moving_median_final_point(
        #         sub_buffer_sensor=sub_buffer_sensor_y
        #     )

        #     # Transform final_point in selection buffers into final point in main buffer
        #     final_sensor_x_point = final_point_x + point_sensor_x_select_start
        #     final_sensor_y_point = final_point_y + point_sensor_y_select_start
        #     final_command_point = point_command_select_start

        # # If using IQC-based selection
        # elif self.point_selection == IQC_POINT:

        #     # Extract sub-buffer to select from
        #     sub_buffer_sensor_x = self.buffer_sensor[
        #         point_sensor_x_select_start:point_sensor_x_select_end, SENSOR_INDEX_SENSOR_X,
        #     ]
        #     sub_buffer_sensor_y = self.buffer_sensor[
        #         point_sensor_y_select_start:point_sensor_y_select_end, SENSOR_INDEX_SENSOR_Y,
        #     ]

        #     # For command copy same value as many times as necessary
        #     sub_buffer_command_x = np.repeat(
        #         self.buffer_command[
        #             point_command_select_start : point_command_select_start + 1,
        #             COMMAND_INDEX_COMMAND_X,
        #         ],
        #         sub_buffer_sensor_x.shape[0],
        #         axis=0,
        #     )
        #     sub_buffer_command_y = np.repeat(
        #         self.buffer_command[
        #             point_command_select_start : point_command_select_start + 1,
        #             COMMAND_INDEX_COMMAND_Y,
        #         ],
        #         sub_buffer_sensor_y.shape[0],
        #         axis=0,
        #     )

        #     # Select
        #     final_point_x = self._IQC_final_point(
        #         sub_buffer_sensor=sub_buffer_sensor_x,
        #         sub_buffer_command=sub_buffer_command_x,
        #     )
        #     final_point_y = self._IQC_final_point(
        #         sub_buffer_sensor=sub_buffer_sensor_y,
        #         sub_buffer_command=sub_buffer_command_y,
        #     )

        #     # Transform final_point in selection buffers into final point in main buffer
        #     final_sensor_x_point = final_point_x + point_sensor_x_select_start
        #     final_sensor_y_point = final_point_y + point_sensor_y_select_start
        #     final_command_point = point_command_select_start

        # # If using Extended-IQC-based selection
        # elif self.point_selection == EXTENDED_RANGE_IQC_POINT:

        #     # For sensor, take slightly more datapoint
        #     extended_start_x = point_sensor_x_select_start - EXTENDED_IQC_LENGTH
        #     extended_start_x = min(max(extended_start_x, 0), len(self.buffer_sensor))
        #     extended_end_x = point_sensor_x_select_end + EXTENDED_IQC_LENGTH
        #     extended_end_x = min(max(extended_end_x, 0), len(self.buffer_sensor))
        #     sub_buffer_sensor_x = self.buffer_sensor[extended_start_x:extended_end_x, SENSOR_INDEX_SENSOR_X]

        #     extended_start_y = point_sensor_y_select_start - EXTENDED_IQC_LENGTH
        #     extended_start_y = min(max(extended_start_y, 0), len(self.buffer_sensor))
        #     extended_end_y = point_sensor_y_select_end + EXTENDED_IQC_LENGTH
        #     extended_end_y = min(max(extended_end_y, 0), len(self.buffer_sensor))
        #     sub_buffer_sensor_y = self.buffer_sensor[extended_start_y:extended_end_y, SENSOR_INDEX_SENSOR_Y]

        #     # For command copy same value as many times as necessary
        #     sub_buffer_command_x = np.repeat(
        #         self.buffer_command[
        #             point_command_select_start : point_command_select_start + 1,
        #             COMMAND_INDEX_COMMAND_X,
        #         ],
        #         sub_buffer_sensor_x.shape[0],
        #         axis=0,
        #     )
        #     sub_buffer_command_y = np.repeat(
        #         self.buffer_command[
        #             point_command_select_start : point_command_select_start + 1,
        #             COMMAND_INDEX_COMMAND_Y,
        #         ],
        #         sub_buffer_sensor_y.shape[0],
        #         axis=0,
        #     )

        #     # Select
        #     final_point_x = self._IQC_final_point(
        #         sub_buffer_sensor=sub_buffer_sensor_x,
        #         sub_buffer_command=sub_buffer_command_x,
        #     )
        #     final_point_y = self._IQC_final_point(
        #         sub_buffer_sensor=sub_buffer_sensor_y,
        #         sub_buffer_command=sub_buffer_command_y,
        #     )

        #     # Transform final_point in selection buffers into final point in main buffer
        #     final_sensor_x_point = (
        #         final_point_x + point_sensor_x_select_start - EXTENDED_IQC_LENGTH
        #     )
        #     final_sensor_y_point = (
        #         final_point_y + point_sensor_y_select_start - EXTENDED_IQC_LENGTH
        #     )
        #     final_command_point = point_command_select_start

        # # If unknown selection
        # else:
        #     self._logger.error(
        #         f"Error: undefined point selection {self.point_selection}."
        #     )
        #     assert 0

        # return final_sensor_x_point, final_sensor_y_point, final_command_point


######################
# Data Quality Class #

# UNUSED - KEPT IN CASE


class DataQuality:
    def __init__(self, number_datapoints=100):
        self._number_datapoints = number_datapoints
        self.reset()

    def reset(self):
        self.datapoints_input = None
        self.datapoints_output = None
        self.quality = 0.0

    def add_datapoint(self, final_datapoint: Tuple) -> float:
        """Add latest datapoints and return the current data quality."""

        new_input = np.concatenate(
            (
                final_datapoint[9],  # command_x
                final_datapoint[10],  # command_y
            ),
            axis=1,
        )
        new_output = np.concatenate(
            (
                final_datapoint[11],  # sensor_x
                final_datapoint[12],  # sensor_y
            ),
            axis=1,
        )
        if self.datapoints_input is None:
            self.datapoints_input = new_input
            self.datapoints_output = new_output
        else:
            self.datapoints_input = np.concatenate(
                (self.datapoints_input, new_input), axis=0
            )
            self.datapoints_input = self.datapoints_input[-self._number_datapoints :]
            self.datapoints_output = np.concatenate(
                (self.datapoints_output, new_output), axis=0
            )
            self.datapoints_output = self.datapoints_output[-self._number_datapoints :]

        # Compute the mse error as the quality
        R, _, _, _ = np.linalg.lstsq(
            self.datapoints_input, self.datapoints_output, rcond=None
        )
        self.quality = (
            (self.datapoints_output - (self.datapoints_input @ R)) ** 2
        ).mean()

        # Return the quality
        return self.quality
