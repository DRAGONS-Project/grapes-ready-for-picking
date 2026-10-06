import logging
import os
from pathlib import Path

import hydra

log = logging.getLogger(__name__)


def preprocess_config(config):
    """Sets config.exp.log_dir to the Hydra output dir and symlinks it into CWD/outputs/.

    This makes it easy to find experiment logs relative to your working directory
    regardless of where Hydra writes them (e.g. a shared storage path).
    """
    log_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    date_subdir = log_dir.relative_to(log_dir.parents[1])
    log_cwd = Path.cwd() / "logs" / date_subdir

    if log_cwd != log_dir and not log_cwd.exists():
        log_cwd.parent.mkdir(exist_ok=True, parents=True)
        try:
            log_cwd.symlink_to(log_dir, target_is_directory=True)
        except FileExistsError:
            log.info("Attempting to symlink to existing directory.")

    # Log active env-var variants so they appear in the run's stdout
    model_var = os.getenv("MODEL", "default")
    dataset_var = os.getenv("DATASET", "default")
    log.info(f"MODEL={model_var}")
    log.info(f"DATASET={dataset_var}")

    config.exp.log_dir = str(log_dir)
