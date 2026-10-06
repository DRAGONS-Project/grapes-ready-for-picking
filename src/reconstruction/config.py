"""Shared configuration loading for the reconstruction pipeline."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "reconstruction.yaml"


def load_config(path: Path | None = None) -> DictConfig:
    return OmegaConf.load(path or DEFAULT_CONFIG_PATH)
