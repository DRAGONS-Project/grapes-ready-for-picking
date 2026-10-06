import csv
import logging
from pathlib import Path

import wandb

from .base import BaseLogger

log = logging.getLogger(__name__)


class WandBLogger(BaseLogger):
    """W&B logger with namespaced key dispatch.

    Log dict conventions:
        "metrics/<name>"   → wandb.log scalar
        "metadata/<name>"  → appended row to <name>.csv in the run dir
    """

    def __init__(
        self,
        config,
        dir,
        group,
        name,
        log_metrics=True,
        exclude_metrics=None,
        log_metadata=True,
        exclude_metadata=None,
        **kwargs,
    ):
        super().__init__()
        wandb.init(config=config, dir=dir, group=group, name=name, **kwargs)
        self.log_metrics = log_metrics
        self.exclude_metrics = exclude_metrics or []
        self.log_metadata = log_metadata
        self.exclude_metadata = exclude_metadata or []

    def _log_metrics(self, log_dict):
        metrics_dict = {
            k: v
            for k, v in log_dict.items()
            if k.startswith("metrics/") and not any(k in e for e in self.exclude_metrics)
        }
        metrics_dict = {k.replace("metrics/", "", 1): v for k, v in metrics_dict.items()}
        if metrics_dict:
            wandb.log(metrics_dict)

    def _log_metadata(self, log_dict):
        metadata_dict = {
            k: v
            for k, v in log_dict.items()
            if k.startswith("metadata/")
            and not any(k in e for e in self.exclude_metadata)
        }
        for key, value in metadata_dict.items():
            filepath = (Path(wandb.run.dir) / key).with_suffix(".csv")
            filepath.parent.mkdir(parents=True, exist_ok=True)

            import torch

            if isinstance(value, torch.Tensor):
                value = value.numpy(force=True)

            with open(filepath, "a", newline="") as f:
                writer = csv.writer(f)
                if hasattr(value, "ndim") and value.ndim > 0:
                    writer.writerows(value)
                else:
                    writer.writerow([value])

    def log(self, log_dict: dict):
        if self.log_metrics:
            self._log_metrics(log_dict)
        if self.log_metadata:
            self._log_metadata(log_dict)
