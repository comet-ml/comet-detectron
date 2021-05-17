import comet_ml

comet_ml.init()

import os
import random
import cv2

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor, hooks

from comet_trainer import CometDefaultTrainer
from utils import format_predictions, get_balloon_dicts


def setup():
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.DATASETS.TRAIN = ("balloon_train",)
    cfg.DATASETS.TEST = ("balloon_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        128  # faster, and good enough for this toy dataset (default: 512)
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


def log_predictions(predictor, dataset_dicts, experiment):
    """Log Model Predictions to Comet for analysis.

    Args:
        predictor (DefaultPredictor): Predictor Object for Detectron Model
        dataset_dicts (dict): Dataset Dictionary contaning samples of data and annotations
        experiment (comet_ml.Experiment): Comet Experiment Object
    """
    predictions_data = {}
    for d in random.sample(dataset_dicts, 3):
        file_name = str(d["file_name"])

        im = cv2.imread(file_name)
        annotations = d["annotations"]

        outputs = predictor(
            im
        )  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        formatted_data = format_predictions(outputs, annotations)
        predictions_data[file_name] = formatted_data
        experiment.log_image(file_name, name=file_name)

    experiment.log_asset_data(predictions_data, name="predictions-data.json")


def main():
    experiment = comet_ml.Experiment()
    cfg = setup()

    for d in ["train", "val"]:
        DatasetCatalog.register(
            "balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d)
        )
        MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
    balloon_metadata = MetadataCatalog.get("balloon_train")

    # Wrap the Detectron Default Trainer
    trainer = CometDefaultTrainer(cfg, experiment)
    trainer.resume_or_load(resume=False)

    # Register Hook to compute metrics using an Evaluator Object
    trainer.register_hooks(
        [hooks.EvalHook(10, lambda: trainer.evaluate_metrics(cfg, trainer.model))]
    )

    # Register Hook to compute eval loss
    trainer.register_hooks(
        [hooks.EvalHook(10, lambda: trainer.evaluate_loss(cfg, trainer.model))]
    )
    trainer.train()

    # Evaluate Model Predictions
    cfg.MODEL.WEIGHTS = os.path.join(
        cfg.OUTPUT_DIR, "model_final.pth"
    )  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    log_predictions(predictor, get_balloon_dicts("balloon/val"), experiment)


if __name__ == "__main__":
    main()
