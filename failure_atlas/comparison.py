"""Compare baseline and repaired robustness experiments."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from .reporting import DISPLAY_NAMES


def save_comparison_plot(path: Path, baseline: dict, repaired: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    for axis, name in zip(axes.flat, baseline["experiments"], strict=False):
        original = baseline["experiments"][name]["results"]
        improved = repaired["experiments"][name]["results"]
        x_values = [item["severity"] for item in original]
        axis.plot(
            x_values,
            [item["accuracy"] * 100 for item in original],
            marker="o",
            label="baseline",
        )
        axis.plot(
            x_values,
            [item["accuracy"] * 100 for item in improved],
            marker="o",
            label="repaired",
        )
        axis.set_title(DISPLAY_NAMES[name])
        axis.set_xlabel("severity")
        axis.set_ylabel("accuracy (%)")
        axis.grid(alpha=0.25)
        axis.legend()
    axes.flat[-1].axis("off")
    figure.suptitle("FailureAtlas — baseline vs repaired model", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def boundary_delta(before: float | None, after: float | None) -> str:
    if before is None and after is None:
        return "not reached → not reached"
    if before is None:
        return f"not reached → {after:g}"
    if after is None:
        return f"{before:g} → not reached (improved)"
    difference = after - before
    return f"{before:g} → {after:g} ({difference:+g})"


def write_comparison_report(path: Path, comparison: dict) -> None:
    baseline = comparison["baseline"]
    repaired = comparison["repaired"]
    clean_delta = repaired["baseline"]["accuracy"] - baseline["baseline"]["accuracy"]
    lines = [
        "# FailureAtlas Automated Repair Report",
        "",
        "## Agent repair plan",
        "",
    ]
    for action in comparison["repair_plan"]["actions"]:
        lines.append(
            f"- **{DISPLAY_NAMES[action['perturbation']]}** "
            f"(score {action['vulnerability_score']:.3f}): {action['rationale']}"
        )
    lines.extend(
        [
            "",
            "## Clean-data quality guardrail",
            "",
            f"- Baseline accuracy: {baseline['baseline']['accuracy']:.2%}",
            f"- Repaired accuracy: {repaired['baseline']['accuracy']:.2%}",
            f"- Change: {clean_delta:+.2%}",
            "",
            "## Severe-boundary movement",
            "",
            "| Perturbation | Before → after | Max-severity accuracy change |",
            "|---|---:|---:|",
        ]
    )
    for name in baseline["experiments"]:
        before_experiment = baseline["experiments"][name]
        after_experiment = repaired["experiments"][name]
        movement = boundary_delta(
            before_experiment["boundaries"]["severe"],
            after_experiment["boundaries"]["severe"],
        )
        end_delta = (
            after_experiment["results"][-1]["accuracy"]
            - before_experiment["results"][-1]["accuracy"]
        )
        lines.append(f"| {DISPLAY_NAMES[name]} | {movement} | {end_delta:+.2%} |")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Repair augmentations use the same perturbation families as the scan, so "
            "this measures robustness to known shifts, not unseen ones.",
            "- Results use one fixed seed on the test split, without confidence intervals.",
            "- The repaired checkpoint was selected on a validation split held out of both "
            "baseline training and repair, not on the test split.",
            "",
            "![Repair comparison](repair_comparison.png)",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def save_comparison_json(path: Path, comparison: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
