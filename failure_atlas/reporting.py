"""Generate machine-readable and human-readable FailureAtlas reports."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


DISPLAY_NAMES = {
    "rotation": "Rotation",
    "gaussian_noise": "Gaussian noise",
    "blur": "Gaussian blur",
    "occlusion": "Center occlusion",
    "brightness": "Brightness reduction",
}

REPAIR_HINTS = {
    "rotation": "Add random-rotation augmentation and check orientation-sensitive classes.",
    "gaussian_noise": "Add noise augmentation or a light denoising preprocessor.",
    "blur": "Add blur augmentation and multi-scale feature training.",
    "occlusion": "Use random erasing/Cutout and inspect reliance on central pixels.",
    "brightness": "Add color-jitter exposure augmentation and input normalization checks.",
}


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_accuracy_curves(path: Path, experiments: dict, baseline: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    for axis, (name, experiment) in zip(axes.flat, experiments.items(), strict=False):
        x_values = [entry["severity"] for entry in experiment["results"]]
        y_values = [entry["accuracy"] * 100 for entry in experiment["results"]]
        axis.plot(x_values, y_values, marker="o", linewidth=2)
        axis.axhline((baseline - 0.03) * 100, color="#eab308", linestyle="--", label="mild")
        axis.axhline((baseline - 0.10) * 100, color="#f97316", linestyle="--", label="obvious")
        axis.axhline((baseline - 0.20) * 100, color="#dc2626", linestyle="--", label="severe")
        axis.set_title(DISPLAY_NAMES[name])
        axis.set_xlabel("severity")
        axis.set_ylabel("accuracy (%)")
        axis.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower right")
    figure.suptitle("FailureAtlas — model accuracy under controlled shifts", fontsize=15)
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_markdown_report(path: Path, payload: dict) -> None:
    baseline = payload["baseline"]
    lines = [
        "# FailureAtlas Phase 1 Report",
        "",
        "## Baseline",
        "",
        f"- Accuracy: {baseline['accuracy']:.2%}",
        f"- Macro-F1: {baseline['macro_f1']:.2%}",
        f"- ECE: {baseline['ece']:.4f}",
        f"- High-confidence errors: {baseline['high_confidence_errors']}",
        "",
        "## Discovered failure boundaries",
        "",
        "| Perturbation | Mild (-3pp) | Obvious (-10pp) | Severe (-20pp) |",
        "|---|---:|---:|---:|",
    ]
    for name, experiment in payload["experiments"].items():
        boundary = experiment["boundaries"]
        values = [boundary[level] for level in ("mild", "obvious", "severe")]
        rendered = ["not reached" if value is None else f"{value:g}" for value in values]
        lines.append(
            f"| {DISPLAY_NAMES[name]} | {rendered[0]} | {rendered[1]} | {rendered[2]} |"
        )

    ranked = sorted(
        payload["experiments"].items(),
        key=lambda pair: pair[1]["results"][-1]["accuracy"],
    )
    weakest_name, weakest = ranked[0]
    weakest_result = weakest["results"][-1]
    lines.extend(
        [
            "",
            "## Agent diagnosis",
            "",
            f"The largest tested degradation came from **{DISPLAY_NAMES[weakest_name]}**: "
            f"accuracy fell to {weakest_result['accuracy']:.2%} at severity "
            f"{weakest_result['severity']:g}.",
            "",
            "Recommended targeted repairs:",
            "",
        ]
    )
    for name, _ in ranked:
        lines.append(f"- **{DISPLAY_NAMES[name]}:** {REPAIR_HINTS[name]}")
    lines.extend(["", "![Accuracy curves](failure_curves.png)", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
