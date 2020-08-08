from pathlib import Path

pylot_label_map = "dependencies/models/pylot.names"
coco_label_map = "dependencies/models/coco.names"

base_path = Path("dependencies/models/obstacle_detection")


class DetectionModel:
    def __init__(self, name, path, label_map):
        self.name = name
        self.path = path
        self.label_map = label_map


def edet_info_from_n(n):
    edet_path = base_path/"efficientdet"
    name = "efficientdet-d{}".format(n)
    return name, DetectionModel(name,
                                edet_path/name/"{}_frozen.pb".format(name),
                                coco_label_map)


edet_models = {k: v for k, v in (edet_info_from_n(i) for i in range(7))}
pylot_models = {
    n: DetectionModel(n, base_path/"{}/frozen_inference_graph.pb".format(n),
                      pylot_label_map) for n in ["faster-rcnn",
                                                 "ssdlite-mobilenet-v2"]
}

model_dict = {**pylot_models, **edet_models}
