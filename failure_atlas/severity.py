"""Resolve the severity grid each dataset is scanned with."""

from __future__ import annotations

from collections.abc import Mapping

from .calibration import BOUNDS, Calibration


def validate_grid(name: str, values: list[float]) -> None:
    """A grid starts at the clean setting and moves strictly toward stronger settings."""
    clean, bound = BOUNDS[name]
    if len(values) < 2:
        raise ValueError(f"Severity grid for '{name}' needs at least two points.")
    if values[0] != clean:
        raise ValueError(f"Severity grid for '{name}' must start at {clean:g}, got {values[0]:g}.")
    direction = 1 if bound > clean else -1
    if any((later - earlier) * direction <= 0 for earlier, later in zip(values, values[1:])):
        raise ValueError(f"Severity grid for '{name}' must be strictly monotonic.")


def severities_for(search: Mapping, calibration: Calibration) -> dict[str, list[float]]:
    """Per-perturbation scan grids for the calibrated dataset.

    Levels up to strength 1 come from the calibration, so every perturbation is sampled
    at the same pixel distortions. The configured severities add the stronger settings
    beyond that range, where severe failures are usually found.
    """
    grids: dict[str, list[float]] = {}
    for name, extension in search["severities"].items():
        grids[name] = calibration.grid(name, extension)
        validate_grid(name, grids[name])
    return grids
