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
    weight_mode: str = "scratch"
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
    config = TrainConfig(**values)
    if config.weight_mode not in {"scratch", "pretrained"}:
        raise ValueError("training.weight_mode must be 'scratch' or 'pretrained'.")
    return config


# Fine-tuning a pretrained backbone with a from-scratch learning rate wipes out the
# features it arrived with, and the 224×224 inputs these models see are far larger
# than the 32×32 images the configured batch sizes were chosen for.
PRETRAINED_LR_SCALE = 0.1
PRETRAINED_MAX_BATCH = 64


def effective_learning_rate(learning_rate: float, weight_mode: str) -> float:
    return learning_rate * PRETRAINED_LR_SCALE if weight_mode == "pretrained" else learning_rate


def effective_batch_size(batch_size: int, weight_mode: str, explicit: bool = False) -> int:
    """Cap the batch for pretrained models unless the caller asked for a size."""
    if weight_mode == "pretrained" and not explicit:
        return min(batch_size, PRETRAINED_MAX_BATCH)
    return batch_size


def check_weight_mode(model, expected: str, source) -> None:
    """Stop when a checkpoint's weight mode differs from the one the run asked for."""
    actual = getattr(model, "weight_mode", "scratch")
    if actual != expected:
        raise RuntimeError(
            f"{source} holds a '{actual}' model but this run uses weight mode "
            f"'{expected}'. Pass the matching --weight-mode."
        )
