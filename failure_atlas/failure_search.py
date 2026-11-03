"""Evaluate controlled shifts and locate model failure boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader

from .data import FASHION_MNIST_CLASSES, NORMALIZE_MEAN, NORMALIZE_STD
from .perturbations import PERTURBATIONS


@dataclass(frozen=True)
class FailureLevels:
    mild: float = 0.03
    obvious: float = 0.10
    severe: float = 0.20


def normalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(NORMALIZE_MEAN, device=images.device).view(1, -1, 1, 1)
    std = torch.tensor(NORMALIZE_STD, device=images.device).view(1, -1, 1, 1)
    return (images - mean) / std


def expected_calibration_error(
    confidences: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    bins: int = 15,
) -> float:
    correctness = predictions.eq(targets)
    score = torch.zeros((), dtype=torch.float64)
    boundaries = torch.linspace(0, 1, bins + 1)
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        selected = (confidences > lower) & (confidences <= upper)
        if selected.any():
            accuracy = correctness[selected].float().mean()
            confidence = confidences[selected].mean()
            score += selected.float().mean().double() * abs(accuracy - confidence).double()
    return float(score)


@torch.inference_mode()
def evaluate_perturbation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    perturbation: str,
    severity: float,
    seed: int,
) -> dict:
    if perturbation not in PERTURBATIONS:
        raise ValueError(f"Unknown perturbation: {perturbation}")
    model.eval()
    transform = PERTURBATIONS[perturbation]
    generator = torch.Generator(device=device).manual_seed(seed)
    predictions_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    confidences_all: list[torch.Tensor] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        shifted = transform(images, severity, generator)
        logits = model(normalize(shifted))
        probabilities = logits.softmax(dim=1)
        confidences, predictions = probabilities.max(dim=1)
        predictions_all.append(predictions.cpu())
        targets_all.append(targets.cpu())
        confidences_all.append(confidences.cpu())

    predictions = torch.cat(predictions_all)
    targets = torch.cat(targets_all)
    confidences = torch.cat(confidences_all)
    correct = predictions.eq(targets)
    matrix = confusion_matrix(targets.numpy(), predictions.numpy(), labels=list(range(10)))
    per_class = {
        name: float(matrix[index, index] / max(matrix[index].sum(), 1))
        for index, name in enumerate(FASHION_MNIST_CLASSES)
    }
    return {
        "accuracy": float(correct.float().mean()),
        "macro_f1": float(f1_score(targets.numpy(), predictions.numpy(), average="macro")),
        "mean_confidence": float(confidences.mean()),
        "ece": expected_calibration_error(confidences, predictions, targets),
        "high_confidence_errors": int(((~correct) & (confidences >= 0.90)).sum()),
        "per_class_accuracy": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def classify_drop(drop: float, levels: FailureLevels) -> str:
    if drop >= levels.severe:
        return "severe"
    if drop >= levels.obvious:
        return "obvious"
    if drop >= levels.mild:
        return "mild"
    return "stable"


def find_boundaries(
    results: list[dict], baseline_accuracy: float, levels: FailureLevels
) -> dict[str, float | None]:
    boundaries: dict[str, float | None] = {
        "mild": None,
        "obvious": None,
        "severe": None,
    }
    thresholds = {
        "mild": levels.mild,
        "obvious": levels.obvious,
        "severe": levels.severe,
    }
    for name, required_drop in thresholds.items():
        for result in results:
            if baseline_accuracy - result["accuracy"] >= required_drop:
                boundaries[name] = result["severity"]
                break
    return boundaries
