import torch
import numpy as np

from typing import Dict

from detectron2.engine import DefaultTrainer
from detectron2.utils import comm
from detectron2.utils.events import get_event_storage
from fvcore.common.config import CfgNode


def log_config(cfg, experiment):
    """Traverse the Detectron Config graph and log the parameters

    Args:
        cfg (CfgNode): Detectron Config Node
        experiment (comet_ml.Experiment): [description]
    """

    def log_node(node, prefix):
        if not isinstance(node, CfgNode):
            if isinstance(node, dict):
                experiment.log_parameters(node, prefix=prefix)

            else:
                experiment.log_parameter(prefix, node)
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

    def _write_metrics(
        self,
        loss_dict: Dict[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ):
        """
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
                total_losses_reduced, "{}total_loss".format(prefix)
            )
            storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)
