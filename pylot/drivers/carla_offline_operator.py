from pathlib import Path
import cv2
import numpy as np
import pickle
import json

from pylot.perception.camera_frame import CameraFrame
from pylot.perception.messages import FrameMessage, ObstaclesMessage
from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.detection.utils import BoundingBox2D

import erdos


class OfflineCarlaSensorV1(erdos.Operator):
    DATASET_PATH = Path("/data/sukritk/faster-rcnn-driving/training_data/"
                        "town01_start1/TrainingDataSet/ClearNoon")

    def __init__(self, release_sensor_stream, camera_stream,
                 notify_reading_stream, perfect_obstacles_stream,
                 camera_setup, flags):
        self.image_w = camera_setup.width
        self.image_h = camera_setup.height
        assert camera_setup.camera_type == "sensor.camera.rgb", \
            camera_setup.camera_type

        # extract file paths from directory
        self.index = 0
        json_files = self.DATASET_PATH.glob("*.json")
        png_files = self.DATASET_PATH.glob("*.png")

        def get_file_number(path):
            # get fn, drop extension, split by '-' and get last value
            return int(path.stem.split("-")[-1])

        # verify indices of json and png files match
        assert sorted([get_file_number(p) for p in json_files]) == \
               sorted([get_file_number(p) for p in png_files]), \
               "fn numbers of json & png files in directory {} don't match".\
               format(self.DATASET_PATH)

        self.zipped_paths = zip(sorted(json_files, key=get_file_number),
                                sorted(png_files, key=get_file_number))
        self.zipped_paths = list(self.zipped_paths)

    def get_messages(self, index):
        if self.index > len(self.zipped_paths):
            raise StopIteration()
        json_path, png_path = self.zipped_paths[self.index]
        self.index += 1

        timestamp = erdos.Timestamp(coordinates=[self.index])  # ???

        # Get obstacles from json
        def get_obst(row):
            # bbox_coords format: [[x_min,y_min],[x_max,y_max]]
            class_label, bbox_coords = row
            arg_order = np.array(bbox_coords).T.reshape(-1)  # xmn,xmx,ymn,ymx
            return Obstacle(BoundingBox2D(*arg_order), 1.0, class_label)

        ground_obstacles = [get_obst(r) for r in json.load(json_path)]
        obst_message = ObstaclesMessage(timestamp, ground_obstacles)

        # get image from png and resize
        img = cv2.imread(str(png_path)).astype(np.uint8)  # BGR
        resized_img = cv2.resize(img, dsize=(self.image_w, self.image_h),
                                 interpolation=cv2.INTER_CUBIC)
        msg = FrameMessage(timestamp, CameraFrame(resized_img, "BGR"))
        pickled_camera_msg = pickle.dumps(msg,
                                          protocol=pickle.HIGHEST_PROTOCOL)
        return pickled_camera_msg, obst_message

    @erdos.profile_method()
    def release_data(self, timestamp):
        if timestamp.is_top:
            # The operator can always send data ASAP.
            self._release_data = True
        else:
            self._logger.debug("@{}: {} releasing sensor data".format(
                timestamp, self.config.name))
            watermark_msg = erdos.WatermarkMessage(timestamp)
            self._camera_stream.send_pickled(timestamp,
                                             self._pickled_messages[timestamp])
            # Note: The operator is set not to automatically propagate
            # watermark messages received on input streams. Thus, we can
            # issue watermarks only after the Carla callback is invoked.
            self._camera_stream.send(watermark_msg)
            with self._pickle_lock:
                del self._pickled_messages[timestamp]

    @staticmethod
    def connect(release_sensor_stream):
        camera_stream = erdos.WriteStream()
        notify_reading_stream = erdos.WriteStream()
        ground_obstacles_stream = erdos.WriteStream()
        return [camera_stream, notify_reading_stream, ground_obstacles_stream]
