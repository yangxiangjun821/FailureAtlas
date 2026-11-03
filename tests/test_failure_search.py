import torch

from failure_atlas.failure_search import FailureLevels, classify_drop, find_boundaries
from failure_atlas.perturbations import PERTURBATIONS


def test_perturbations_keep_shape_and_range() -> None:
    images = torch.full((2, 1, 28, 28), 0.5)
    severities = {
        "rotation": 15.0,
        "gaussian_noise": 0.2,
        "blur": 1.0,
        "occlusion": 0.25,
        "brightness": 0.5,
    }
    for name, transform in PERTURBATIONS.items():
        result = transform(images, severities[name], torch.Generator().manual_seed(42))
        assert result.shape == images.shape
        assert 0.0 <= result.min() <= result.max() <= 1.0


def test_failure_level_classification() -> None:
    levels = FailureLevels()
    assert classify_drop(0.02, levels) == "stable"
    assert classify_drop(0.03, levels) == "mild"
    assert classify_drop(0.10, levels) == "obvious"
    assert classify_drop(0.20, levels) == "severe"


def test_boundary_detection() -> None:
    results = [
        {"severity": 0.0, "accuracy": 0.90},
        {"severity": 0.2, "accuracy": 0.86},
        {"severity": 0.4, "accuracy": 0.79},
        {"severity": 0.8, "accuracy": 0.68},
    ]
    boundaries = find_boundaries(results, 0.90, FailureLevels())
    assert boundaries == {"mild": 0.2, "obvious": 0.4, "severe": 0.8}
