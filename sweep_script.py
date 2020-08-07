from itertools import product
from pathlib import Path

MODELS = [
    "dependencies/models/obstacle_detection/faster-rcnn/frozen_inference_graph.pb",
    "dependencies/models/obstacle_detection/efficientdet/efficientdet-d0/efficientdet-d0_frozen.pb",
    "dependencies/models/obstacle_detection/efficientdet/efficientdet-d1/efficientdet-d1_frozen.pb",
    "dependencies/models/obstacle_detection/efficientdet/efficientdet-d2/efficientdet-d2_frozen.pb",
    "dependencies/models/obstacle_detection/efficientdet/efficientdet-d3/efficientdet-d3_frozen.pb",
    "dependencies/models/obstacle_detection/efficientdet/efficientdet-d4/efficientdet-d4_frozen.pb",
    "dependencies/models/obstacle_detection/efficientdet/efficientdet-d5/efficientdet-d5_frozen.pb",
    "dependencies/models/obstacle_detection/efficientdet/efficientdet-d6/efficientdet-d6_frozen.pb",
    "dependencies/models/obstacle_detection/ssdlite-mobilenet-v2/frozen_inference_graph.pb"
]

town_runs = [
    "/data/ges/faster-rcnn-driving/training_data/town01_start30/TrainingDataSet/ClearNoon/",
    "/data/ges/faster-rcnn-driving/training_data/town02_start1/TrainingDataSet/ClearNoon/",
    "/data/ges/faster-rcnn-driving/training_data/town02_start80/TestDataSet/ClearNoon/",
]


def run_config():
    pass


def get_numbers():
    pass


def clean_up():
    pass


if __name__ == "__main__":
    for model_path, run in product(MODELS, town_runs):
        model_name = Path(model_path).parent.stem
        dataset_name = "{}-{}".format(Path(run).parent.parent.stem,
                                      Path(run).parent.stem)
        run_name = "{}-{}".format(model_name, dataset_name)

        profile_fn_flag = "--profile_file_name=sweep_files/{}.json".format(run_name)
        csv_log_fn_flag = "--csv_log_file_name=sweep_files/{}.csv".format(run_name)
        log_fn_flag = "--log_file_name=sweep_files/{}.log".format(run_name)

        model_path_flag = "--obstacle_detection_model_paths={}".format(
            model_path
        )
        model_name_flag = "--obstacle_detection_model_names={}".format(
            model_name
        )
        dataset_path_flag = "--offline_carla_dataset_path={}".format(
            run
        )

        flagfile_flag = "--flagfile=configs/detection.conf"
        print("python run_offline_carla.py {} {} {} {} {} {} {}".format(
            flagfile_flag,
            model_path_flag, model_name_flag, dataset_path_flag,
            profile_fn_flag, log_fn_flag,  csv_log_fn_flag
        ))
