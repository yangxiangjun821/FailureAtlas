import torch

from failure_atlas.repair_agent import TargetedAugment, create_repair_plan


def test_agent_selects_most_vulnerable_experiment() -> None:
    analysis = {
        "baseline": {"accuracy": 0.9},
        "experiments": {
            "rotation": {
                "boundaries": {"obvious": 10, "severe": 20},
                "results": [{"accuracy": 0.9}, {"accuracy": 0.5}],
            },
            "brightness": {
                "boundaries": {"obvious": None, "severe": None},
                "results": [{"accuracy": 0.9}, {"accuracy": 0.85}],
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
