"""Training, evaluation, checkpointing, and reproducibility utilities."""

from __future__ import annotations

import contextlib
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
    ViTB16CIFAR,
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
    "ViTB16CIFAR": ViTB16CIFAR,
}


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, dict]:
    # Checkpoints hold only tensors and plain containers, so the restricted
    # unpickler is enough and untrusted files cannot execute code.
    payload = torch.load(path, map_location=device, weights_only=True)
    model_id = payload.get("model_id")
    model_class = payload.get("model_class")
    if model_id:
        weight_mode = payload.get("config", {}).get("weight_mode", "scratch")
        model = build_model(
            model_id,
            weight_mode=weight_mode,
            initialize_weights=False,
        ).to(device)
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


def halve_until_fits(requested: int, fits, minimum: int = 4) -> int:
    """Largest batch from ``requested``, halving, for which ``fits(batch)`` holds."""
    batch = max(int(requested), 1)
    while batch > minimum and not fits(batch):
        batch //= 2
    return max(batch, min(minimum, int(requested)))


@contextlib.contextmanager
def _undo_side_effects(model: nn.Module, device: torch.device):
    """Run a probe without changing the model's statistics or the random streams.

    A training-mode forward pass would otherwise overwrite BatchNorm running statistics
    (ruining pretrained ones) and shift the dropout/shuffling random numbers.
    """
    norms = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    saved = [(m.momentum, m.num_batches_tracked.clone() if m.num_batches_tracked is not None else None) for m in norms]
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device)
    for module in norms:
        module.momentum = 0.0
    try:
        yield
    finally:
        for module, (momentum, tracked) in zip(norms, saved):
            module.momentum = momentum
            if tracked is not None:
                module.num_batches_tracked.copy_(tracked)
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state, device)


def fit_batch_size(
    model: nn.Module,
    requested: int,
    device: torch.device,
    *,
    training: bool,
    sample_shape: tuple[int, ...] = (3, 32, 32),
    headroom: float = 0.8,
) -> int:
    """Shrink a batch size until a measured step fits in free GPU memory.

    Cards that run out of dedicated memory do not always raise an error: on Windows the
    driver spills into shared system memory and training becomes an order of magnitude
    slower. So the probe compares the measured peak with the free memory instead of
    waiting for an out-of-memory error. On CPU the requested size is returned unchanged.
    """
    if device.type != "cuda":
        return requested
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    # Adam keeps two extra tensors per parameter; they are not present during the probe.
    optimizer_bytes = 2 * parameter_bytes if training else 0

    def fits(batch: int) -> bool:
        torch.cuda.empty_cache()
        free_bytes, _ = torch.cuda.mem_get_info(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        try:
            with _undo_side_effects(model, device):
                inputs = torch.randn(batch, *sample_shape, device=device)
                if training:
                    model.train()
                    model(inputs).sum().backward()
                    model.zero_grad(set_to_none=True)
                else:
                    model.eval()
                    with torch.inference_mode():
                        model(inputs)
            peak = torch.cuda.max_memory_allocated(device) - baseline
        except torch.cuda.OutOfMemoryError:
            return False
        finally:
            torch.cuda.empty_cache()
        return peak + optimizer_bytes <= free_bytes * headroom

    return halve_until_fits(requested, fits)
