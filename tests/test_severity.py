from pathlib import Path

import pytest
import yaml

from failure_atlas.calibration import BOUNDS, STRENGTH_LEVELS, load_calibration
from failure_atlas.catalog import PERTURBATION_REGISTRY
from failure_atlas.severity import severities_for, validate_grid

SEARCH = yaml.safe_load(Path("configs/failure_search.yaml").read_text(encoding="utf-8"))["search"]
DATASETS = ("fashion_mnist", "mnist", "cifar10")


def test_configured_extension_matches_the_registry() -> None:
    for name, values in SEARCH["severities"].items():
        assert list(PERTURBATION_REGISTRY.get(name).severity_levels) == values
        assert values[0] == BOUNDS[name][0]


@pytest.mark.parametrize("dataset_id", DATASETS)
def test_committed_calibration_covers_every_perturbation(dataset_id: str) -> None:
    calibration = load_calibration(dataset_id)
    assert set(calibration.tables) == set(SEARCH["severities"])
    assert calibration.reference_rmse == pytest.approx(
        min(calibration.max_rmse(name) for name in calibration.tables), rel=1e-4
    )


@pytest.mark.parametrize("dataset_id", DATASETS)
def test_grids_are_valid_and_start_clean(dataset_id: str) -> None:
    for name, values in severities_for(SEARCH, load_calibration(dataset_id)).items():
        validate_grid(name, values)
        assert values[0] == BOUNDS[name][0]


@pytest.mark.parametrize("dataset_id", DATASETS)
def test_every_perturbation_is_sampled_at_the_same_pixel_distortions(dataset_id: str) -> None:
    calibration = load_calibration(dataset_id)
    grids = severities_for(SEARCH, calibration)
    for name, values in grids.items():
        for level, severity in zip(STRENGTH_LEVELS, values):
            # Rounded severities and the blur floor allow a small deviation.
            assert calibration.strength(name, severity) == pytest.approx(level, abs=0.03), (
                dataset_id,
                name,
                level,
            )


def test_calibration_depends_on_the_dataset() -> None:
    fashion = load_calibration("fashion_mnist")
    cifar = load_calibration("cifar10")
    assert fashion.reference_rmse != pytest.approx(cifar.reference_rmse, rel=0.05)
    assert fashion.severity_at("gaussian_noise", 0.1) != pytest.approx(
        cifar.severity_at("gaussian_noise", 0.1), rel=0.05
    )


def test_grid_that_does_not_start_clean_is_rejected() -> None:
    with pytest.raises(ValueError, match="must start at"):
        validate_grid("blur", [0.3, 1.0, 2.0])


def test_non_monotonic_grid_is_rejected() -> None:
    with pytest.raises(ValueError, match="monotonic"):
        validate_grid("blur", [0, 1.0, 0.6, 3.0])
    with pytest.raises(ValueError, match="monotonic"):
        validate_grid("brightness", [1.0, 0.5, 0.7])
