from pathlib import Path
import cv2
import numpy as np
import pickle
import json
import time

from pylot.perception.camera_frame import CameraFrame
from pylot.perception.messages import FrameMessage, ObstaclesMessage
from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.detection.utils import BoundingBox2D

import erdos


class OfflineCarlaSensorV1(erdos.Operator):
    DATASET_PATH = Path("/home/erdos/offline_carla_data/town01_start30/"
                        "TrainingDataSet/ClearNoon")

    def __init__(self, camera_stream, ground_obstacles_stream,
                 time_to_decision_stream, camera_setup, flags):
        self._camera_stream = camera_stream
        self._ground_obstacles_stream = ground_obstacles_stream
        self._time_to_decision_stream = time_to_decision_stream
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)

        self.image_w = camera_setup.width
        self.image_h = camera_setup.height
        assert camera_setup.camera_type == "sensor.camera.rgb", \
            camera_setup.camera_type

        self.zipped_paths = self.get_json_png_pairs(self.DATASET_PATH)

        self._logger.debug("Found {} json, png pairs in {}".format(
            len(self.zipped_paths), self.DATASET_PATH
        ))

    def get_json_png_pairs(self, data_path):
        # extract file paths from directory
        json_files = list(data_path.glob("*.json"))
        png_files = list(data_path.glob("*.png"))

        def get_file_number(path):
            # get fn, drop extension, split by '-' and get last value
            return int(path.stem.split("-")[-1])

        # verify some properties of the json and png file numbers
        json_numbers = sorted([get_file_number(p) for p in json_files])
        png_numbers = sorted([get_file_number(p) for p in png_files])
        # all unique
        assert np.array_equal(np.unique(json_numbers), json_numbers), \
            "json numbers not unique in directory {}".format(data_path)
        assert np.array_equal(np.unique(png_numbers), png_numbers), \
            "png numbers not unique in directory {}".format(data_path)
        only_in_json = list(set(json_numbers) - set(png_numbers))
        only_in_png = list(set(png_numbers) - set(json_numbers))
        assert len(only_in_json) == 0, \
               "nonzero empty set of fn numbers only in json files in \
                   directory {}: {}".format(data_path, only_in_json)
        assert len(only_in_png) == 0, \
               "nonzero empty set of fn numbers only in png files in \
                   directory {}: {}".format(data_path, only_in_png)

        zipped_paths = zip(sorted(json_files, key=get_file_number),
                           sorted(png_files, key=get_file_number))
        zipped_paths = list(zipped_paths)
        assert len(zipped_paths) > 0, len(zipped_paths)
        return zipped_paths

    @erdos.profile_method()
    def get_messages(self, index, timestamp):
        self._logger.debug("@{}: {} releasing sensor index {}".format(
                timestamp, self.config.name, index))
        json_path, png_path = self.zipped_paths[index]

        # Get obstacles from json
        def get_obst(row):
            # bbox_coords format: [[x_min,y_min],[x_max,y_max]]
            class_label, bbox_coords = row
            arg_order = np.array(bbox_coords).T.reshape(-1)  # xmn,xmx,ymn,ymx
            return Obstacle(BoundingBox2D(*arg_order), 1.0, class_label)

        def read_json(path):
            with open(path, 'r') as f:
                return json.load(f)

        ground_obstacles = [get_obst(r) for r in read_json(json_path)]
        obst_message = ObstaclesMessage(timestamp, ground_obstacles)

        # get image from png and resize
        img = cv2.imread(str(png_path)).astype(np.uint8)  # BGR
        resized_img = cv2.resize(img, dsize=(self.image_w, self.image_h),
                                 interpolation=cv2.INTER_CUBIC)
        msg = FrameMessage(timestamp, CameraFrame(resized_img, "BGR"))
        pickled_camera_msg = pickle.dumps(msg,
                                          protocol=pickle.HIGHEST_PROTOCOL)
        return pickled_camera_msg, obst_message

    @staticmethod
    def connect():
        camera_stream = erdos.WriteStream()
        ground_obstacles_stream = erdos.WriteStream()
        time_to_decision_stream = erdos.WriteStream()
        return [camera_stream, ground_obstacles_stream,
                time_to_decision_stream]

    def run(self):
        for i in range(len(self.zipped_paths)):
            time.sleep(0.1)
            timestamp = erdos.Timestamp(coordinates=[i])  # ???
            result = self.get_messages(i, timestamp)
            print(result)
            pickled_camera_msg, obst_message = result
            ttd_msg = erdos.Message(timestamp, 10000)  # 10s time to decision
            self._camera_stream.send_pickled(timestamp, pickled_camera_msg)
            self._ground_obstacles_stream.send(obst_message)
            self._time_to_decision_stream.send(ttd_msg)
        while True:
            time.sleep(10)
