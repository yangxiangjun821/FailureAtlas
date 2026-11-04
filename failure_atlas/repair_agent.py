"""Rule-based agent that converts failure evidence into a targeted repair policy."""

from __future__ import annotations

import random
from collections.abc import Sequence
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
    severity_source: str = "default"


# (probability, default severity, safety limit, rationale). The default is used
# when the scan never reached the severe boundary. Otherwise augmentation is
# trained up to the observed severe boundary, clamped to the safety limit so a
# very late boundary cannot destroy clean accuracy. For brightness the severity
# is a lower bound on the exposure factor, so the limit is a floor.
ACTION_SETTINGS = {
    "rotation": (0.45, 15.0, 30.0, "Improve orientation invariance before the severe boundary."),
    "gaussian_noise": (0.45, 0.20, 0.40, "Expose the model to noise up to the severe boundary."),
    "blur": (0.35, 1.00, 2.00, "Preserve features under the observed blur failure boundary."),
    "occlusion": (0.35, 0.30, 0.40, "Reduce dependence on the center of the object."),
    "brightness": (0.40, 0.60, 0.30, "Improve robustness to low-exposure inputs."),
}


def target_severity(name: str, experiment: dict) -> tuple[float, str]:
    """Derive the augmentation strength from the observed severe boundary."""
    _, default, limit, _ = ACTION_SETTINGS[name]
    boundary = experiment["boundaries"].get("severe")
    if boundary is None:
        return default, "default"
    clamped = max(boundary, limit) if name == "brightness" else min(boundary, limit)
    source = "severe_boundary" if clamped == boundary else "severe_boundary_clamped"
    return float(clamped), source


def _truncate(axis: list[float], drops: list[float], limit: float) -> tuple[list[float], list[float]]:
    """Cut the curve at ``limit`` on the axis, interpolating the drop at the cut."""
    points_x, points_y = [axis[0]], [drops[0]]
    for i in range(1, len(axis)):
        if axis[i] <= limit:
            points_x.append(axis[i])
            points_y.append(drops[i])
            continue
        if axis[i - 1] < limit:
            fraction = (limit - axis[i - 1]) / (axis[i] - axis[i - 1])
            points_x.append(limit)
            points_y.append(drops[i - 1] + fraction * (drops[i] - drops[i - 1]))
        break
    return points_x, points_y


def score_vulnerability(experiment: dict, baseline_accuracy: float) -> float:
    """Score sustained degradation rather than relying on one extreme endpoint.

    The base term is the mean accuracy drop over the scanned range: the trapezoidal
    area under the drop curve divided by the range, so it does not change when the
    grid is made denser in one region.

    When results carry a calibrated ``strength`` (pixel-space RMSE in units of the
    dataset's reference RMSE) the curve is integrated over strength 0 to 1 only. That
    range is reachable by every perturbation, so scores of different perturbations
    compare at equal pixel distortion. Settings beyond it still count through the
    boundary bonuses. Without ``strength`` the raw severity axis is used.
    """
    results = experiment["results"]
    if len(results) < 2:
        return 0.0
    drops = [max(0.0, baseline_accuracy - item["accuracy"]) for item in results]
    if all("strength" in item for item in results):
        axis, drops = _truncate([item["strength"] for item in results], drops, 1.0)
    else:
        axis = [item["severity"] for item in results]
    span = abs(axis[-1] - axis[0])
    if span == 0:
        return 0.0
    area = sum(
        abs(axis[i + 1] - axis[i]) * (drops[i] + drops[i + 1]) / 2
        for i in range(len(axis) - 1)
    )
    mean_drop = area / span
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
        probability, _, _, rationale = ACTION_SETTINGS[name]
        max_severity, source = target_severity(name, analysis["experiments"][name])
        actions.append(
            RepairAction(
                name,
                score,
                probability,
                max_severity,
                f"{rationale} Target severity {max_severity:g} ({source.replace('_', ' ')}).",
                source,
            ).__dict__
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

    def __init__(
        self,
        plan: dict,
        mean: Sequence[float] = NORMALIZE_MEAN,
        std: Sequence[float] = NORMALIZE_STD,
    ) -> None:
        self.actions = plan["actions"]
        self.mean = tuple(mean)
        self.std = tuple(std)

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
        return vision_functional.normalize(tensor, self.mean, self.std)
