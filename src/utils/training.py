import logging
import random
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)


def setup_device(config) -> torch.device:
    """Return a torch.device based on config.exp.device.

    If config.exp.device is None or empty, auto-selects cuda > mps > cpu.
    """
    device_str = getattr(config.exp, "device", None)
    if not device_str:
        if torch.cuda.is_available():
            device_str = "cuda"
        elif torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    return torch.device(device_str)


def set_seed(seed: int):
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, optimizer, epoch: int, path: str | Path):
    """Save model + optimizer state to a .pt checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    log.info(f"Checkpoint saved: {path}")


def load_checkpoint(path: str | Path, model, optimizer=None) -> int:
    """Load model (and optionally optimizer) state from a checkpoint.

    Returns:
        epoch: the epoch at which the checkpoint was saved.
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    log.info(f"Checkpoint loaded: {path} (epoch {epoch})")
    return epoch
