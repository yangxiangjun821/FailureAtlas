"""Training, evaluation, checkpointing, and reproducibility utilities."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from .config import TrainConfig
from .model import (
    ConvNeXtTinyCIFAR,
    EfficientNetB0CIFAR,
    FashionCNN,
    LeNet,
    MobileNetV3CIFAR,
    ResNet18CIFAR,
    ViTCIFAR,
    build_model,
)


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


@torch.inference_mode()
def measure_inference_speed(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> dict[str, float]:
    """Measure synchronized end-to-end model latency on real test batches."""
    model.eval()
    batches = iter(loader)
    try:
        warmup, _ = next(batches)
    except StopIteration:
        return {"inference_ms_per_sample": 0.0, "throughput_samples_per_second": 0.0}
    model(warmup.to(device, non_blocking=True))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_samples = 0
    started = time.perf_counter()
    for index, (inputs, _) in enumerate(batches):
        if index >= max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        model(inputs)
        total_samples += inputs.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if not total_samples or elapsed <= 0:
        return {"inference_ms_per_sample": 0.0, "throughput_samples_per_second": 0.0}
    return {
        "inference_ms_per_sample": elapsed * 1000 / total_samples,
        "throughput_samples_per_second": total_samples / elapsed,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: TrainConfig,
    metrics: dict[str, float],
    model_id: str | None = None,
    dataset_id: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_class": type(model).__name__,
            "model_id": model_id,
            "dataset_id": dataset_id,
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "metrics": metrics,
        },
        path,
    )


MODEL_CLASSES: dict[str, type[nn.Module]] = {
    "FashionCNN": FashionCNN,
    "LeNet": LeNet,
    "ResNet18CIFAR": ResNet18CIFAR,
    "MobileNetV3CIFAR": MobileNetV3CIFAR,
    "EfficientNetB0CIFAR": EfficientNetB0CIFAR,
    "ConvNeXtTinyCIFAR": ConvNeXtTinyCIFAR,
    "ViTCIFAR": ViTCIFAR,
}


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, dict]:
    # Checkpoints hold only tensors and plain containers, so the restricted
    # unpickler is enough and untrusted files cannot execute code.
    payload = torch.load(path, map_location=device, weights_only=True)
    model_id = payload.get("model_id")
    model_class = payload.get("model_class")
    if model_id:
        model = build_model(model_id).to(device)
    elif model_class in MODEL_CLASSES:
        model = MODEL_CLASSES[model_class]().to(device)
    else:
        raise ValueError(
            f"Unknown model_class '{model_class}' in checkpoint. "
            f"Supported: {', '.join(MODEL_CLASSES)}"
        )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def write_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
