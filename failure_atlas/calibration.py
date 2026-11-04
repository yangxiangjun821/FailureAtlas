"""Calibrate perturbation severities to a common pixel-space distortion scale.

Severities of different perturbations (degrees, noise sigma, brightness factor) are
not comparable. Here each perturbation is measured on a fixed sample of unlabeled
images and its severity is expressed as the RMSE between the clean and the perturbed
image, in [0, 1] pixel units. Dividing by a per-dataset reference RMSE gives a
"strength" that is 1.0 where the least destructive perturbation (in RMSE terms)
reaches its own limit, so all five are fully observed on [0, 1] and can be compared.

RMSE is a distortion measure, not a measure of semantic damage: a one-pixel shift can
have a large RMSE and a heavy blur a small one. Calibration makes the comparison
fair in pixel terms; it does not make the perturbations perceptually equivalent.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .perturbations import PERTURBATIONS

CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "configs" / "calibration"

# Strongest physical setting probed for each perturbation, and the clean setting.
BOUNDS: dict[str, tuple[float, float]] = {
    "rotation": (0.0, 60.0),
    "gaussian_noise": (0.0, 0.8),
    "blur": (0.0, 6.0),
    "occlusion": (0.0, 0.6),
    "brightness": (1.0, 0.2),
}

# Blur treats any sigma below this as this value, so smaller severities add no distortion.
SWEEP_FLOOR = {"blur": 0.1}

# Fractions of the reference RMSE at which the scan grid is placed. Dense at the low
# end, where models often start to fail.
STRENGTH_LEVELS = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

SWEEP_POINTS = 80


def rmse_between(clean: torch.Tensor, perturbed: torch.Tensor) -> float:
    return float((perturbed - clean).pow(2).mean().sqrt())


def _sweep_severities(name: str) -> list[float]:
    """Severities from clean to the bound: geometric near clean, then linear."""
    clean, bound = BOUNDS[name]
    span = bound - clean
    floor = SWEEP_FLOOR.get(name, 0.0)
    fractions = np.concatenate(
        [np.geomspace(1e-3, 0.1, SWEEP_POINTS // 4, endpoint=False),
         np.linspace(0.1, 1.0, SWEEP_POINTS - SWEEP_POINTS // 4)]
    )
    values = [clean + span * f for f in fractions]
    if floor:
        values = [value for value in values if abs(value - clean) >= floor]
        values.insert(0, clean + np.sign(span) * floor)
    return [float(clean), *[float(v) for v in values]]


def measure_table(
    images: torch.Tensor,
    name: str,
    seed: int = 42,
    transform: Callable | None = None,
) -> list[tuple[float, float]]:
    """(severity, RMSE) pairs, ordered by increasing distortion, strictly increasing."""
    transform = transform or PERTURBATIONS[name]
    table: list[tuple[float, float]] = []
    for severity in _sweep_severities(name):
        generator = torch.Generator().manual_seed(seed)
        perturbed = transform(images, severity, generator)
        table.append((severity, rmse_between(images, perturbed)))
    # Keep only points where the distortion strictly grows. This drops the plateau after
    # it saturates (occlusion covering all the ink) and any decline past a peak, so the
    # table stays invertible.
    kept = [table[0]]
    for severity, value in table[1:]:
        if value > kept[-1][1] + 1e-6:
            kept.append((severity, value))
    if len(kept) < 3:
        raise ValueError(f"Perturbation '{name}' produced no measurable distortion.")
    return kept


@dataclass(frozen=True)
class Calibration:
    dataset_id: str
    sample_count: int
    seed: int
    reference_rmse: float
    tables: dict[str, tuple[tuple[float, float], ...]]
    metric: str = "rmse"

    def _pairs(self, name: str) -> tuple[tuple[float, float], ...]:
        try:
            return self.tables[name]
        except KeyError as error:
            raise KeyError(
                f"No calibration for '{name}' on '{self.dataset_id}'. "
                f"Calibrated: {', '.join(self.tables)}"
            ) from error

    def max_rmse(self, name: str) -> float:
        return self._pairs(name)[-1][1]

    def rmse(self, name: str, severity: float) -> float:
        """Interpolated distortion for a severity (clamped to the measured range)."""
        pairs = sorted(self._pairs(name))
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        return float(np.interp(severity, xs, ys))

    def strength(self, name: str, severity: float) -> float:
        """Distortion in units of the reference RMSE; 1.0 is the common limit."""
        return self.rmse(name, severity) / self.reference_rmse

    def severity_at(self, name: str, rmse: float) -> float:
        pairs = self._pairs(name)
        return float(np.interp(rmse, [p[1] for p in pairs], [p[0] for p in pairs]))

    def grid(self, name: str, extension: Sequence[float] = ()) -> list[float]:
        """Scan grid: calibrated levels up to strength 1, then the given extension.

        Extension severities that are already inside the calibrated range, or that go
        beyond what the perturbation can reach, are ignored; the rest keep the scan
        reaching the strong settings where models typically hit their severe boundary.
        """
        clean = BOUNDS[name][0]
        direction = np.sign(BOUNDS[name][1] - clean)
        points = [clean]
        for level in STRENGTH_LEVELS[1:]:
            severity = self.severity_at(name, level * self.reference_rmse)
            points.append(float(f"{severity:.4g}"))
        limit = points[-1]
        points += [float(v) for v in extension if (v - limit) * direction > 1e-9]
        deduped: list[float] = []
        for value in points:
            if not deduped or (value - deduped[-1]) * direction > 1e-9:
                deduped.append(value)
        return deduped

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "metric": self.metric,
            "sample_count": self.sample_count,
            "seed": self.seed,
            "reference_rmse": self.reference_rmse,
            "tables": {
                name: [[severity, value] for severity, value in pairs]
                for name, pairs in self.tables.items()
            },
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Calibration":
        return cls(
            dataset_id=raw["dataset_id"],
            sample_count=int(raw["sample_count"]),
            seed=int(raw["seed"]),
            reference_rmse=float(raw["reference_rmse"]),
            tables={
                name: tuple((float(s), float(v)) for s, v in pairs)
                for name, pairs in raw["tables"].items()
            },
            metric=raw.get("metric", "rmse"),
        )


def calibrate(
    dataset_id: str, images: torch.Tensor, names: Sequence[str] | None = None, seed: int = 42
) -> Calibration:
    """Measure every perturbation and set the reference to the weakest one's limit."""
    names = tuple(names or BOUNDS)
    tables = {name: tuple(measure_table(images, name, seed)) for name in names}
    reference = min(pairs[-1][1] for pairs in tables.values())
    return Calibration(
        dataset_id=dataset_id,
        sample_count=len(images),
        seed=seed,
        reference_rmse=float(f"{reference:.6g}"),
        tables=tables,
    )


def calibration_path(dataset_id: str, root: Path = CALIBRATION_DIR) -> Path:
    return root / f"{dataset_id}.json"


def save_calibration(calibration: Calibration, root: Path = CALIBRATION_DIR) -> Path:
    path = calibration_path(calibration.dataset_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration.to_dict(), indent=2), encoding="utf-8")
    return path


def load_calibration(dataset_id: str, root: Path = CALIBRATION_DIR) -> Calibration:
    path = calibration_path(dataset_id, root)
    if not path.exists():
        raise FileNotFoundError(
            f"No pixel-distortion calibration for '{dataset_id}'. Create it with: "
            f"python -m scripts.calibrate_perturbations --dataset {dataset_id}"
        )
    return Calibration.from_dict(json.loads(path.read_text(encoding="utf-8")))
