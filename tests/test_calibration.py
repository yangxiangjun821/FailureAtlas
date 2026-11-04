import pytest
import torch

from failure_atlas.calibration import (
    Calibration,
    calibrate,
    load_calibration,
    measure_table,
    rmse_between,
    save_calibration,
)


@pytest.fixture(scope="module")
def images() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.rand(64, 1, 28, 28, generator=generator)


def test_rmse_is_zero_for_identical_images_and_exact_for_a_shift(images) -> None:
    assert rmse_between(images, images) == 0.0
    assert rmse_between(torch.zeros(4), torch.full((4,), 0.5)) == pytest.approx(0.5)


def test_measure_table_is_strictly_increasing_in_distortion(images) -> None:
    for name in ("rotation", "gaussian_noise", "blur", "occlusion", "brightness"):
        table = measure_table(images, name)
        values = [value for _, value in table]
        assert values[0] == 0.0
        assert all(later > earlier for earlier, later in zip(values, values[1:])), name


def test_reference_is_the_weakest_perturbations_limit(images) -> None:
    calibration = calibrate("synthetic", images)
    limits = {name: calibration.max_rmse(name) for name in calibration.tables}
    assert calibration.reference_rmse == pytest.approx(min(limits.values()), rel=1e-4)
    # Every perturbation can reach the reference, so strength 1 is reachable by all.
    assert all(limit >= calibration.reference_rmse * 0.9999 for limit in limits.values())


def test_severity_and_rmse_round_trip(images) -> None:
    calibration = calibrate("synthetic", images)
    for name in calibration.tables:
        target = calibration.reference_rmse * 0.4
        severity = calibration.severity_at(name, target)
        assert calibration.rmse(name, severity) == pytest.approx(target, rel=1e-3)


def test_grid_reaches_strength_one_then_follows_the_extension(images) -> None:
    calibration = calibrate("synthetic", images)
    grid = calibration.grid("rotation", extension=[5, 20, 60])
    assert grid[0] == 0
    assert all(later > earlier for earlier, later in zip(grid, grid[1:]))
    assert grid[-1] == 60
    limit = calibration.severity_at("rotation", calibration.reference_rmse)
    assert any(value == pytest.approx(limit, rel=1e-3) for value in grid)
    # Extension points already inside the calibrated range are not repeated.
    assert len(grid) == len(set(grid))


def test_downward_scale_grid_is_monotone_downward(images) -> None:
    calibration = calibrate("synthetic", images)
    grid = calibration.grid("brightness", extension=[0.5, 0.3, 0.2])
    assert grid[0] == 1.0
    assert all(later < earlier for earlier, later in zip(grid, grid[1:]))


def test_calibration_round_trips_through_json(images, tmp_path) -> None:
    calibration = calibrate("synthetic", images)
    path = save_calibration(calibration, tmp_path)
    assert path.exists()
    assert load_calibration("synthetic", tmp_path) == Calibration.from_dict(
        calibration.to_dict()
    )


def test_missing_calibration_explains_how_to_create_it(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="calibrate_perturbations"):
        load_calibration("nope", tmp_path)
