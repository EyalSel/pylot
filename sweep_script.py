from itertools import product
from pathlib import Path
import argparse
from detection_models import model_dict

MODELS = list(model_dict.values())

RUNS = \
    filter(lambda p: p.is_dir() and p.stem == "ClearNoon" and any(p.iterdir()),
           Path("/data/ges/faster-rcnn-driving/training_data").rglob("*"))
RUNS = list(RUNS)


def run_config():
    pass


def clean_up():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--save_frames', action='store_true',
                        help='saves frames. Execution takes much longer, '
                             'so disregard latency values')
    args = parser.parse_args()

    for model, run in product(MODELS, RUNS):
        dataset_name = "{}-{}".format(Path(run).parent.parent.stem,
                                      Path(run).parent.stem)
        run_name = "{}-{}".format(model.name, dataset_name)

        profile_fn_flag = \
            "--profile_file_name=sweep_files/{}.json".format(run_name)
        csv_log_fn_flag = \
            "--csv_log_file_name=sweep_files/{}.csv".format(run_name)
        log_fn_flag = "--log_file_name=sweep_files/{}.log".format(run_name)
        result_file_flags = "{} {} {}".format(profile_fn_flag, csv_log_fn_flag,
                                              log_fn_flag)

        model_path_flag = "--obstacle_detection_model_paths={}".format(
            model.path
        )
        model_name_flag = "--obstacle_detection_model_names={}".format(
            model.name
        )
        label_map_flag = "--path_coco_labels={}".format(model.label_map)
        model_flags = "{} {} {}".format(model_path_flag, model_name_flag,
                                        label_map_flag)

        dataset_path_flag = "--offline_carla_dataset_path={}".format(
            run
        )

        if args.save_frames:
            Path("sweep_files/{}".format(run_name)).mkdir(exist_ok=True)
            detector_output_flag = "--log_detector_output=True"
            data_path_flag = "--data_path=sweep_files/{}".format(run_name)
            save_frames_flags = \
                "{} {}".format(detector_output_flag, data_path_flag)
        else:
            save_frames_flags = ""

        flagfile_flag = "--flagfile=configs/detection.conf"
        print("python run_offline_carla.py {} {} {} {} {}".format(
            flagfile_flag,
            model_flags,
            dataset_path_flag,
            result_file_flags,
            save_frames_flags
        ))
