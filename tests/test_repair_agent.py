import pytest
import torch

from failure_atlas.repair_agent import TargetedAugment, create_repair_plan


def test_agent_selects_most_vulnerable_experiment() -> None:
    analysis = {
        "baseline": {"accuracy": 0.9},
        "experiments": {
            "rotation": {
                "boundaries": {"obvious": 10, "severe": 20},
                "results": [
                    {"severity": 0.0, "accuracy": 0.9},
                    {"severity": 1.0, "accuracy": 0.5},
                ],
            },
            "brightness": {
                "boundaries": {"obvious": None, "severe": None},
                "results": [
                    {"severity": 0.0, "accuracy": 0.9},
                    {"severity": 1.0, "accuracy": 0.85},
                ],
            },
        },
    }
    plan = create_repair_plan(analysis, top_k=1)
    assert plan["actions"][0]["perturbation"] == "rotation"


def test_targeted_augment_returns_normalized_tensor() -> None:
    plan = {
        "actions": [
            {
                "perturbation": "gaussian_noise",
                "probability": 1.0,
                "max_severity": 0.1,
            }
        ]
    }
    image = torch.full((1, 28, 28), 128, dtype=torch.uint8)
    result = TargetedAugment(plan)(image)
    assert result.shape == (1, 28, 28)
    assert result.dtype == torch.float32


def test_augmentation_strength_follows_observed_boundary() -> None:
    def experiment(severe):
        return {
            "boundaries": {"obvious": None, "severe": severe},
            "results": [
                    {"severity": 0.0, "accuracy": 0.9},
                    {"severity": 1.0, "accuracy": 0.5},
                ],
        }

    analysis = {
        "baseline": {"accuracy": 0.9},
        "experiments": {
            "rotation": experiment(20.0),
            "gaussian_noise": experiment(0.8),
            "brightness": experiment(0.5),
            "blur": experiment(None),
        },
    }
    plan = create_repair_plan(analysis, top_k=4)
    actions = {item["perturbation"]: item for item in plan["actions"]}
    assert actions["rotation"]["max_severity"] == 20.0
    assert actions["rotation"]["severity_source"] == "severe_boundary"
    assert actions["gaussian_noise"]["max_severity"] == 0.4
    assert actions["gaussian_noise"]["severity_source"] == "severe_boundary_clamped"
    assert actions["brightness"]["max_severity"] == 0.5
    assert actions["blur"]["max_severity"] == 1.0
    assert actions["blur"]["severity_source"] == "default"


def _experiment(points: list[tuple[float, float]]) -> dict:
    return {
        "boundaries": {"obvious": None, "severe": None},
        "results": [{"severity": s, "accuracy": a} for s, a in points],
    }


def test_score_does_not_change_when_the_grid_is_refined() -> None:
    from failure_atlas.repair_agent import score_vulnerability

    def drop_curve(severity: float) -> float:
        return 0.9 - 0.6 * severity

    coarse = _experiment([(s, drop_curve(s)) for s in (0.0, 0.5, 1.0)])
    dense_low = _experiment(
        [(s, drop_curve(s)) for s in (0.0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0)]
    )
    assert score_vulnerability(coarse, 0.9) == pytest.approx(
        score_vulnerability(dense_low, 0.9)
    )


def test_score_handles_descending_severity_scales() -> None:
    from failure_atlas.repair_agent import score_vulnerability

    brightness = _experiment([(1.0, 0.9), (0.6, 0.8), (0.2, 0.6)])
    assert score_vulnerability(brightness, 0.9) == pytest.approx(
        (0.4 * (0.0 + 0.1) / 2 + 0.4 * (0.1 + 0.3) / 2) / 0.8
    )


def test_degenerate_curves_score_zero() -> None:
    from failure_atlas.repair_agent import score_vulnerability

    assert score_vulnerability(_experiment([(0.0, 0.9)]), 0.9) == 0.0
    assert score_vulnerability(_experiment([(0.5, 0.9), (0.5, 0.4)]), 0.9) == 0.0


def _calibrated(points: list[tuple[float, float, float]]) -> dict:
    """(severity, strength, accuracy) triples."""
    return {
        "boundaries": {"obvious": None, "severe": None},
        "results": [{"severity": s, "strength": u, "accuracy": a} for s, u, a in points],
    }


def test_calibrated_scores_compare_perturbations_at_equal_distortion() -> None:
    from failure_atlas.repair_agent import score_vulnerability

    # Same accuracy curve over strength, completely different physical units.
    rotation = _calibrated([(0, 0.0, 0.9), (7.0, 0.5, 0.6), (14.0, 1.0, 0.3)])
    noise = _calibrated([(0.0, 0.0, 0.9), (0.17, 0.5, 0.6), (0.34, 1.0, 0.3)])
    brightness = _calibrated([(1.0, 0.0, 0.9), (0.7, 0.5, 0.6), (0.43, 1.0, 0.3)])
    scores = [score_vulnerability(e, 0.9) for e in (rotation, noise, brightness)]
    assert scores == pytest.approx([scores[0]] * 3)


def test_settings_beyond_the_common_limit_do_not_change_the_base_score() -> None:
    from failure_atlas.repair_agent import score_vulnerability

    within = _calibrated([(0, 0.0, 0.9), (10, 1.0, 0.6)])
    extended = _calibrated([(0, 0.0, 0.9), (10, 1.0, 0.6), (30, 2.0, 0.1), (60, 4.0, 0.1)])
    assert score_vulnerability(extended, 0.9) == pytest.approx(score_vulnerability(within, 0.9))


def test_curve_cut_at_the_limit_interpolates_the_drop() -> None:
    from failure_atlas.repair_agent import score_vulnerability

    # Drop rises linearly from 0 to 0.7 over strength 0 to 2. Only 0 to 1 counts: the
    # drop at strength 1 is 0.35 and the mean over the interval is 0.175.
    curve = _calibrated([(0, 0.0, 0.9), (5, 2.0, 0.2)])
    assert score_vulnerability(curve, 0.9) == pytest.approx(0.175)
