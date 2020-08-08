import signal
import os
import erdos
import pylot
import pylot.flags
import pylot.operator_creator
from absl import app, flags
from pylot.drivers.carla_offline_operator import OfflineCarlaSensorV1

flags.DEFINE_string('offline_carla_dataset_path', None, 'path to offline carla dataset')
# flag below shouldn't be set. It's just used to pass information to the
# OfflineCarlaSensorV1 operator so it can send a sigint and terminate the run
# once it finished sending everything. Shitty design that violates abstraction
# barriers, but I couldn't think up of a better way to do this.
flags.DEFINE_integer('parent_pid', os.getpid(), "pid of parent process, don\'t set this")

FLAGS = flags.FLAGS

# copied from pylot.py
CENTER_CAMERA_LOCATION = pylot.utils.Location(1.3, 0.0, 1.8)


def driver():
    # create camera operator
    from pylot.drivers.sensor_setup import RGBCameraSetup
    # dummy transform
    transform = pylot.utils.Transform(CENTER_CAMERA_LOCATION,
                                      pylot.utils.Rotation(pitch=-15))
    name = "offline-carla-camera"
    offline_camera_setup = RGBCameraSetup(name, FLAGS.camera_image_width,
                                          FLAGS.camera_image_height, transform,
                                          fov=90)
    op_config = erdos.OperatorConfig(name=offline_camera_setup.get_name() +
                                     '_operator',
                                     flow_watermarks=False,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    center_camera_stream, perfect_obstacles_stream, time_to_decision_stream = \
        erdos.connect(OfflineCarlaSensorV1, op_config, [],
                      offline_camera_setup, FLAGS)

    # create detection operator
    obstacles_stream = None
    if any('efficientdet' in model
            for model in FLAGS.obstacle_detection_model_names):
        obstacles_streams = pylot.operator_creator.\
                add_efficientdet_obstacle_detection(
                    center_camera_stream, time_to_decision_stream)
        obstacle_stream = obstacles_streams[0]
    else:
        obstacles_streams = pylot.operator_creator.add_obstacle_detection(
            center_camera_stream, time_to_decision_stream)
        obstacle_stream = obstacles_streams[0]

    if FLAGS.evaluate_obstacle_detection:
        pylot.operator_creator.add_detection_evaluation(
            obstacle_stream, perfect_obstacles_stream)

    # create tracker operator

    if FLAGS.obstacle_tracking:
        obstacles_wo_history_tracking_stream = \
            pylot.operator_creator.add_obstacle_tracking(
                obstacles_stream,
                center_camera_stream,
                time_to_decision_stream)

    if FLAGS.evaluate_obstacle_tracking:
        pylot.operator_creator.add_tracking_evaluation(
            obstacles_wo_history_tracking_stream, perfect_obstacles_stream)

    node_handle = erdos.run_async()
    return node_handle


def shutdown(sig, frame):
    raise KeyboardInterrupt


def shutdown_pylot(node_handle):
    node_handle.shutdown()


def main(args):
    node_handle = None
    try:
        node_handle = driver()
        signal.signal(signal.SIGINT, shutdown)
        node_handle.join()
    except KeyboardInterrupt:
        shutdown_pylot(node_handle)
    except Exception:
        shutdown_pylot(node_handle)
        raise


if __name__ == "__main__":
    app.run(main)
