"""Training, evaluation, checkpointing, and reproducibility utilities."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from .config import TrainConfig
from .model import FashionCNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * targets.size(0)
        total_examples += targets.size(0)
    return total_loss / total_examples


@torch.inference_mode()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    predictions: list[int] = []
    targets_all: list[int] = []
    confidence_sum = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        logits = model(inputs)
        probabilities = logits.softmax(dim=1)
        confidence, predicted = probabilities.max(dim=1)
        predictions.extend(predicted.cpu().tolist())
        targets_all.extend(targets.tolist())
        confidence_sum += confidence.sum().item()
    return {
        "accuracy": float(accuracy_score(targets_all, predictions)),
        "macro_f1": float(f1_score(targets_all, predictions, average="macro")),
        "mean_confidence": confidence_sum / len(targets_all),
    }


def save_checkpoint(
    path: Path,
    model: FashionCNN,
    config: TrainConfig,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_class": "FashionCNN",
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[FashionCNN, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = FashionCNN().to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def write_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
