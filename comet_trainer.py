import os
import time
import datetime
import logging
import torch
import numpy as np

from typing import Dict
from collections import OrderedDict

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.utils import comm
from detectron2.utils.events import get_event_storage
from fvcore.common.config import CfgNode


def log_config(cfg, experiment):
    """Traverse the Detectron Config graph and log the parameters

    Args:
        cfg (CfgNode): Detectron Config Node
        experiment (comet_ml.Experiment): Comet ML Experiment object
    """

    def log_node(node, prefix):
        if not isinstance(node, CfgNode):
            if isinstance(node, dict):
                experiment.log_parameters(node, prefix=prefix)

            else:
                experiment.log_parameter(name=prefix, value=node)
            return

        node_dict = dict(node)
        for k, v in node_dict.items():
            _prefix = f"{prefix}-{k}" if prefix else k
            log_node(v, _prefix)

    log_node(cfg, "")


class CometDefaultTrainer(DefaultTrainer):
    def __init__(self, cfg, experiment):
        """
        Args:
            cfg (CfgNode): Detectron Config Node
            experiment (comet_ml.Experiment): Comet Experiment object
        """
        super().__init__(cfg)
        self.experiment = experiment
        log_config(cfg, self.experiment)

        self._trainer._write_metrics = self._write_metrics

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    def evaluate_metrics(self, cfg, model):
        """Compute Additional Metrics using the COCO Evaluator

        Args:
            cfg (CfgNode): Detectron Config Object
            model (torch.nn.Module): Model Object
        """
        evaluators = [
            self.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "evaluation")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = self.test(cfg, model, evaluators)
        res = OrderedDict({k: v for k, v in res.items()})

        # Log Computed Metrics to Comet
        for k, v in res.items():
            self.experiment.log_metrics(v, prefix=f"eval-{k}")

        return res

    def evaluate_loss(self, cfg, model):
        """Compute and log the validation loss to Comet

        Args:
            cfg (CfgNode): Detectron Config Object
            model (torch.nn.Module): Detectron Model

        Returns:
            dict: Empty Dict to satisfy Detectron Eval Hook API requirements
        """
        eval_loader = build_detection_test_loader(
            cfg, cfg.DATASETS.TEST[0], DatasetMapper(cfg, True)
        )

        # Copying inference_on_dataset from evaluator.py
        total = len(eval_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []

        if comm.is_main_process():
            storage = get_event_storage()

            for idx, inputs in enumerate(eval_loader):
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_compute_time = 0
                start_compute_time = time.perf_counter()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time
                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                seconds_per_img = total_compute_time / iters_after_start
                if idx >= num_warmup * 2 or seconds_per_img > 5:
                    total_seconds_per_img = (
                        time.perf_counter() - start_time
                    ) / iters_after_start
                    eta = datetime.timedelta(
                        seconds=int(total_seconds_per_img * (total - idx - 1))
                    )
                    log_every_n_seconds(
                        logging.INFO,
                        "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                            idx + 1, total, seconds_per_img, str(eta)
                        ),
                        n=5,
                    )
                loss_batch = self._get_loss(model, inputs)
                losses.append(loss_batch)
            mean_loss = np.mean(losses)

            # Log to Comet
            self.experiment.log_metric("eval_loss", mean_loss)

            storage.put_scalar("eval_loss", mean_loss)
            comm.synchronize()

        # Returns empty dict to satisfy Dectron Eval Hook requirement
        return {}

    def _get_loss(self, model, data):
        # How loss is calculated on train_loop
        metrics_dict = model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def _write_metrics(
        self,
        loss_dict: Dict[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ):
        """Patch for existing Default Trainer _write_metrics method so that
        metrics can also be logged to Comet

        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={self.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            self.experiment.log_metrics(metrics_dict, prefix=prefix)
            self.experiment.log_metric(
                "{}total_loss".format(prefix),
                total_losses_reduced,
            )
            storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)
