"""Configuration helpers for repeatable experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    batch_size: int = 128
    epochs: int = 8
    learning_rate: float = 0.001
    num_workers: int = 0
    data_dir: str = "data"
    artifact_dir: str = "artifacts"
    device: str = "auto"
    # Fraction of the training split held out for validation. Neither the baseline
    # nor the repair trains on it; the test split is only used for final reporting.
    val_fraction: float = 0.1


def load_train_config(path: str | Path) -> TrainConfig:
    """Load and validate the training section of a YAML configuration file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    values = raw.get("training", {})
    unknown = set(values) - set(TrainConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown training options: {sorted(unknown)}")
    return TrainConfig(**values)
