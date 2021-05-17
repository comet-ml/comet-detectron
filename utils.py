import os
import cv2
import json

import numpy as np
from detectron2.structures import BoxMode


def format_predictions(outputs, annotations):
    """Format Dectectron Predictions so that they can be visualized using
    Comet Panels

    Args:
        outputs (list): List of formatted dicts
    """
    data = []
    prediction = outputs["instances"].to("cpu")

    predicted_boxes = prediction.pred_boxes.tensor.numpy().tolist()
    predicted_scores = prediction.scores.numpy().tolist()
    predicted_classes = prediction.pred_classes.numpy().tolist()

    for annotation in annotations:
        bbox = annotation["bbox"]

        # Convert from numpy.int64 to int
        x, y, x2, y2 = map(lambda x: x.item(), bbox)

        label = annotation["category_id"]
        data.append(
            {
                "label": f"ground_truth-{label}",
                "score": 100,
                "box": {"x": x, "y": y, "x2": x2, "y2": y2},
            }
        )

    for predicted_box, predicted_score, predicted_class in zip(
        predicted_boxes, predicted_scores, predicted_classes
    ):
        x, y, x2, y2 = predicted_box
        data.append(
            {
                "label": predicted_class,
                "box": {"x": x, "y": y, "x2": x2, "y2": y2},
                "score": predicted_score * 100,
            }
        )

    return data


def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts
