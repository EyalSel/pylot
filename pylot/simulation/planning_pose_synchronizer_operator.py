import erdos
import time

from collections import deque
from copy import deepcopy
from pylot.utils import time_epoch_ms
from pylot.planning.messages import WaypointsMessage
from pylot.planning.utils import remove_completed_waypoints

# The runtimes of the EfficientDet family of models in milliseconds.
# Taken from https://arxiv.org/abs/1911.09070
EFFICIENTDET_RUNTIMES = {
    "efficientdet-d0": 16,
    "efficientdet-d1": 20,
    "efficientdet-d2": 24,
    "efficientdet-d3": 42,
    "efficientdet-d4": 74,
    "efficientdet-d5": 141,
    "efficientdet-d6": 190,
    "efficientdet-d7": 262
}


class PlanningPoseSynchronizerOperator(erdos.Operator):
    """ Synchronizes and relays waypoints and pose messages to the control
    module to enable pseudo-synchronization of the Carla simulation with a
    high-frequency control.

    This operator receives waypoints (and the corresponding localization
    messages) along with a high-frequency control stream. It applies the
    waypoints after the given runtime (deduced from the time of the receipt of
    the localization and the waypoints message), and sends both control and
    waypoint messages applicable at the given time to the downstream control
    operator.

    Args:
        waypoints_read_stream (:py:class:`erdos.ReadStream`): Stream on which
            the waypoint updates are received.
        pose_read_stream (:py:class:`erdos.ReadStream`): Stream on which the
            high-frequency pose messages are received. These messages trigger
            a control update from the pipeline.
        localization_read_stream (:py:class:`erdos.ReadStream`): Stream on
            which the localization messages that would trigger a future
            waypoint update are received.
        notify_stream1 (:py:class:`erdos.ReadStream`): Stream on which the
            notifications from the first sensor are received.
        notify_stream2 (:py:class:`erdos.ReadStream`): Stream on which the
            notifications from the second sensor are received.
        detector_runtime_stream (:py:class:`erdos.ReadStream`): Stream on which
            a RuntimeMessage from the object detector is received.
        waypoints_write_stream (:py:class:`erdos.WriteStream`): Stream on which
            the waypoints matched to the given pose message are sent to the
            downstream control operator.
        pose_write_stream (:py:class:`erdos.WriteStream`): Stream that relays
            the pose messages from the CarlaOperator to the control module.
    """

    def __init__(self, waypoints_read_stream, pose_read_stream,
                 localization_pose_stream, notify_stream1, notify_stream2,
                 detector_runtime_stream, waypoints_write_stream,
                 pose_write_stream, release_sensor_stream,
                 pipeline_finish_notify_stream, flags):
        # Register callbacks on both the waypoints and the pose stream.
        waypoints_read_stream.add_callback(self.on_waypoints_update)
        pose_read_stream.add_callback(self.on_pose_update)
        localization_pose_stream.add_callback(self.on_localization_update)
        erdos.add_watermark_callback([notify_stream1, notify_stream2],
                                     [release_sensor_stream],
                                     self.on_sensor_ready)
        if detector_runtime_stream:
            detector_runtime_stream.add_callback(self.on_runtime_update)

        # Register watermark callback on pose and the joined stream.
        erdos.add_watermark_callback(
            [pose_read_stream], [waypoints_write_stream, pose_write_stream],
            self.on_pose_watermark)

        # Save the write streams.
        self._waypoints_write_stream = waypoints_write_stream
        self._pipeline_finish_notify_stream = pipeline_finish_notify_stream

        # Initialize a logger.
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            self.config.name + '-csv', self.config.csv_log_file_name)

        # Data used by the operator.
        self._pose_map = dict()
        self._runtime_map = dict()
        self._waypoints = deque()
        self._first_waypoint = True
        self._waypoint_num = 0
        self._last_highest_applicable_time = None

    @staticmethod
    def connect(waypoints_read_stream, pose_read_stream,
                localization_pose_stream, notify_stream1, notify_stream2,
                detector_runtime_stream):
        waypoints_write_stream = erdos.WriteStream()
        pose_write_stream = erdos.WriteStream()
        release_sensor_stream = erdos.WriteStream()
        pipeline_finish_notify_stream = erdos.WriteStream()
        return [
            waypoints_write_stream,
            pose_write_stream,
            release_sensor_stream,
            pipeline_finish_notify_stream,
        ]

    def adjust_processing_time(self, processing_time, detection_runtime):
        """ Adjust the processing time of the message (if required). The
        runtimes are expected to be in milliseconds.

        Currently, only fixes the runtime if EfficientDet is used.

        Args:
            processing_time (int): The runtime of the pipeline.
            detection_time (int): The runtime of the detection module.

        Returns:
            An int representing the modified runtime of the pipeline.
        """
        # Use the first model if multiple models are passed.
        model_name = None
        if isinstance(self._flags.obstacle_detection_model_names, list):
            model_name = self._flags.obstacle_detection_model_names[0]
        else:
            model_name = self._flags.obstacle_detection_model_names

        if model_name.startswith("efficientdet"):
            paper_runtime = EFFICIENTDET_RUNTIMES[model_name]
            processing_time -= (detection_runtime - paper_runtime)

        # TODO (Sukrit) :: Apply a standard deviation to these results from
        # our experiments.
        return processing_time

    @erdos.profile_method()
    def on_waypoints_update(self, msg):
        """ Invoked upon receipt of a waypoints message from the pipeline.

        This method retrieves the pose message for the timestamp, calculates
        the runtime of the pipeline, logs it and saves the waypoints for the
        future.

        Args:
            msg (:py:class:`pylot.planning.messages.WaypointsMessage`): The
                waypoints message received for the given timestamp.
        """
        waypoint_recv_time = time.time()
        self._logger.debug("@{}: received waypoints update.".format(
            msg.timestamp))

        watermark = erdos.WatermarkMessage(msg.timestamp)
        if self._waypoint_num < 10:
            self._logger.debug(
                "@{}: received first waypoint. "
                "Skipping because the simulator might not be in sync.".format(
                    msg.timestamp))
            self._first_waypoint = False
            self._waypoint_num += 1
            # Send a message on the notify stream to ask CARLA to send a new
            # sensor stream.
            self._pipeline_finish_notify_stream.send(watermark)
            return

        # Retrieve the game time.
        game_time = msg.timestamp.coordinates[0]

        # Retrieve the pose message for this timestamp.
        (pose_msg, pose_recv_time) = self._pose_map[game_time]

        # Calculate and log the processing time for this waypoints message.
        processing_time = int((waypoint_recv_time - pose_recv_time) * 1000)
        if game_time in self._runtime_map:
            initial_processing_time = processing_time
            processing_time = self.adjust_processing_time(
                processing_time, self._runtime_map[game_time])
            self._logger.debug("@[{}]: updated processing time from {} to {} "
                               "because of a detection runtime of {}".format(
                                   game_time, initial_processing_time,
                                   processing_time,
                                   self._runtime_map[game_time]))
        else:
            self._logger.debug(
                "@[{}]: did not find a runtime. skipping adjust.")
        self._csv_logger.info('{},{},{},{:.4f}'.format(time_epoch_ms(),
                                                       game_time,
                                                       'end-to-end-runtime',
                                                       processing_time))

        # Apply the waypoints at the timestamp + processing time.
        applicable_time = game_time + processing_time
        if (self._last_highest_applicable_time is None
                or self._last_highest_applicable_time < applicable_time):
            self._last_highest_applicable_time = applicable_time
            self._waypoints.append((applicable_time, msg))
            self._logger.debug(
                "@{}: waypoints will be applicable at {}".format(
                    msg.timestamp, applicable_time))
        else:
            # We add the applicable time to the time between localization
            # readings, and put these waypoints at that location.
            sensor_frequency = self._flags.carla_fps
            if self._flags.carla_localization_frequency != -1:
                sensor_frequency = self._flags.carla_localization_frequency
            applicable_time = self._last_highest_applicable_time + int(
                1000 / sensor_frequency)
            self._last_highest_applicable_time = applicable_time
            self._waypoints.append((applicable_time, msg))
            self._logger.debug(
                "@{}: the waypoints were adjusted by the localization "
                "frequency and will be applicable at {}".format(
                    msg.timestamp, applicable_time))

        # Delete the pose from the map.
        self._pose_map.pop(game_time, None)

        # Send a message on the notify stream to ask CARLA to send a new
        # sensor stream.
        self._pipeline_finish_notify_stream.send(watermark)

    def on_pose_update(self, msg):
        """ Invoked when we receive a pose message from the simulation.

        Save the pose message to be used on the receipt of the watermark.

        Args:
            msg (:py:class:`erdos.Message`): A message containing the pose
                of the ego-vehicle.
        """
        self._logger.debug("@{}: received pose message.".format(msg.timestamp))

        # Retrieve the game time.
        game_time = msg.timestamp.coordinates[0]

        # Save the pose message along with the time at which it was received.
        if game_time in self._pose_map:
            self._pose_map[game_time][0] = msg
        else:
            self._pose_map[game_time] = [msg, None]

    def on_localization_update(self, msg):
        """ Invoked upon receipt of a localization message that will lead
        to a future waypoint update.

        This callback logs the arrival time of the message in order to compute
        the runtime of the pipeline.

        Args:
            msg (:py:class:`erdos.Message`): A message containing the pose
                of the ego-vehicle.
        """
        self._logger.debug("@{}: received localization message.".format(
            msg.timestamp))

        # Retrieve the game time.
        game_time = msg.timestamp.coordinates[0]

        # Save the pose message along with the time at which it was received.
        self._pose_map[game_time] = [msg, time.time()]

    def on_runtime_update(self, msg):
        """ Invoked upon receipt of a runtime message from the detector which
        will be used to adjust the application of runtime in the simulator.

        This callback logs the runtime for the given timestamp.

        Args:
            msg (:py:class:`erdos.Message`): A message containing the runtime
                of the detector.
        """
        self._logger.debug("@[{}]: received runtime message.".format(
            msg.timestamp))

        # Retrieve the game time.
        game_time = msg.timestamp.coordinates[0]

        # Save the runtime from the message.
        self._runtime_map[game_time] = msg.data

    def on_pose_watermark(self, timestamp, waypoint_stream, pose_stream):
        """ Invoked upon receipt of the watermark on the pose stream.

        This callback matches the waypoints to the given timestamp and releases
        both the waypoints and the pose message to the control operator.

        Args:
            timestamp (:py:class:`erdos.Timestamp`): The timestamp of the
                watermark.
            waypoint_stream (:py:class:`erdos.WriteStream`): The stream to send
                the waypoints out on.
            pose_stream (:py:class:`erdos.WriteStream`): The stream to send
                the pose out on.
        """
        self._logger.info("@{}: received pose watermark.".format(timestamp))

        # Retrieve the game time.
        game_time = timestamp.coordinates[0]

        # Retrieve the pose message for the given timestamp.
        pose_msg, pose_ingress_time = self._pose_map[game_time]

        # Match the waypoints to the given timestamp.
        waypoint_index, waypoints = -1, None
        for i, (time, _waypoints) in enumerate(self._waypoints):
            if time <= game_time:
                waypoint_index, waypoints = i, _waypoints
            else:
                break
        self._logger.debug("@{} waypoint index is {}".format(
            timestamp, waypoint_index))

        if waypoints is None:
            # If we haven't received a single waypoint, send an empty message.
            self._waypoints_write_stream.send(
                WaypointsMessage(timestamp, deque([]), deque([])))
        else:
            # Send the trimmed waypoints on the write stream.
            trimmed_waypoints, trimmed_target_speeds = remove_completed_waypoints(
                deepcopy(waypoints.waypoints),
                deepcopy(waypoints.target_speeds),
                pose_msg.data.transform.location)
            waypoints_msg = WaypointsMessage(timestamp, trimmed_waypoints,
                                             trimmed_target_speeds)
            self._waypoints_write_stream.send(waypoints_msg)

            # Trim the saved waypoints.
            for i in range(waypoint_index):
                self._logger.debug("@{}: Pruning {}".format(
                    timestamp, self._waypoints.popleft()))

        # Send the pose and the watermark messages.
        watermark = erdos.WatermarkMessage(timestamp)
        pose_stream.send(pose_msg)
        pose_stream.send(watermark)
        waypoint_stream.send(watermark)
        # Clean up the pose from the dict.
        self._pose_map.pop(game_time, None)
        self._runtime_map.pop(game_time, None)

    @erdos.profile_method()
    def on_sensor_ready(self, timestamp, release_sensor_stream):
        """ Invoked upon receipt of a notification of the sensors being
        ready for the given timestamp.

        Releases a watermark on the release_sensor_stream to notify all the
        sensors to release their data for the given timestamp.

        Args:
            timestamp (:py:class:`erdos.Timestamp`): The timestamp of the
                watermark.
            sensor_ready_stream (:py:class:`erdos.WriteStream`): The stream
                on which to write the notification.
        """
        self._logger.debug("@{}: the sensors are all ready.".format(timestamp))
        release_sensor_stream.send(erdos.WatermarkMessage(timestamp))

        # Retrieve the game time.
        game_time = timestamp.coordinates[0]

        # Also rewrite the receive time for the pose update because the sensor
        # callbacks might take too long.
        if game_time in self._pose_map:
            self._pose_map[game_time][1] = time.time()
        else:
            self._pose_map[game_time] = [None, time.time()]
