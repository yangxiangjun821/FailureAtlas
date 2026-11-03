"""Rule-based agent that converts failure evidence into a targeted repair policy."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torchvision.transforms import functional as vision_functional

from .data import NORMALIZE_MEAN, NORMALIZE_STD
from .perturbations import occlusion


@dataclass(frozen=True)
class RepairAction:
    perturbation: str
    vulnerability_score: float
    probability: float
    max_severity: float
    rationale: str


ACTION_SETTINGS = {
    "rotation": (0.45, 15.0, "Improve orientation invariance before the severe boundary."),
    "gaussian_noise": (0.45, 0.20, "Expose the model to noise up to the observed severe boundary."),
    "blur": (0.35, 1.00, "Preserve features under the observed blur failure boundary."),
    "occlusion": (0.35, 0.30, "Reduce dependence on the center of the object."),
    "brightness": (0.40, 0.60, "Improve robustness to low-exposure inputs."),
}


def score_vulnerability(experiment: dict, baseline_accuracy: float) -> float:
    """Score sustained degradation rather than relying on one extreme endpoint."""
    shifted = experiment["results"][1:]
    if not shifted:
        return 0.0
    drops = [max(0.0, baseline_accuracy - item["accuracy"]) for item in shifted]
    mean_drop = sum(drops) / len(drops)
    severe_bonus = 0.05 if experiment["boundaries"]["severe"] is not None else 0.0
    obvious_bonus = 0.025 if experiment["boundaries"]["obvious"] is not None else 0.0
    return mean_drop + severe_bonus + obvious_bonus


def create_repair_plan(analysis: dict, top_k: int = 3) -> dict:
    baseline_accuracy = analysis["baseline"]["accuracy"]
    ranked = sorted(
        (
            (name, score_vulnerability(experiment, baseline_accuracy))
            for name, experiment in analysis["experiments"].items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    actions = []
    for name, score in ranked[:top_k]:
        probability, max_severity, rationale = ACTION_SETTINGS[name]
        actions.append(
            RepairAction(name, score, probability, max_severity, rationale).__dict__
        )
    return {
        "strategy": "targeted_augmentation_finetune",
        "selected_count": len(actions),
        "actions": actions,
        "ranking": [
            {"perturbation": name, "vulnerability_score": score}
            for name, score in ranked
        ],
    }


class TargetedAugment:
    """Apply only the transformations selected by the repair agent."""

    def __init__(self, plan: dict) -> None:
        self.actions = plan["actions"]

    def __call__(self, image) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            tensor = image.float()
            if image.dtype == torch.uint8:
                tensor = tensor / 255.0
        else:
            tensor = vision_functional.to_tensor(image)
        for action in self.actions:
            if random.random() > action["probability"]:
                continue
            name = action["perturbation"]
            maximum = action["max_severity"]
            if name == "rotation":
                angle = random.uniform(-maximum, maximum)
                tensor = vision_functional.rotate(tensor, angle, fill=0.0)
            elif name == "gaussian_noise":
                standard_deviation = random.uniform(0.02, maximum)
                tensor = (tensor + torch.randn_like(tensor) * standard_deviation).clamp(0, 1)
            elif name == "blur":
                sigma = random.uniform(0.1, maximum)
                tensor = vision_functional.gaussian_blur(tensor, 5, sigma=sigma)
            elif name == "occlusion":
                fraction = random.uniform(0.03, maximum)
                tensor = occlusion(tensor.unsqueeze(0), fraction).squeeze(0)
            elif name == "brightness":
                factor = random.uniform(maximum, 1.0)
                tensor = vision_functional.adjust_brightness(tensor, factor).clamp(0, 1)
        return vision_functional.normalize(tensor, NORMALIZE_MEAN, NORMALIZE_STD)
