"""Re-scan the repaired model and produce an evidence-based comparison report."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from failure_atlas.comparison import (
    save_comparison_json,
    save_comparison_plot,
    write_comparison_report,
)
from failure_atlas.failure_search import FailureLevels, classify_drop, evaluate_perturbation, find_boundaries
from failure_atlas.training import load_checkpoint, resolve_device


def main() -> None:
    search = yaml.safe_load(Path("configs/failure_search.yaml").read_text(encoding="utf-8"))["search"]
    repair = yaml.safe_load(Path("configs/repair.yaml").read_text(encoding="utf-8"))["repair"]
    baseline_analysis = json.loads(Path(repair["analysis"]).read_text(encoding="utf-8"))
    repair_plan = json.loads(Path(repair["repair_plan"]).read_text(encoding="utf-8"))
    device = resolve_device(repair["device"])
    model, checkpoint = load_checkpoint(repair["repaired_checkpoint"], device)
    dataset = datasets.FashionMNIST(
        repair["data_dir"], train=False, transform=transforms.ToTensor(), download=False
    )
    loader = DataLoader(
        dataset,
        batch_size=search["batch_size"],
        shuffle=False,
        num_workers=search["num_workers"],
        pin_memory=device.type == "cuda",
    )
    levels = FailureLevels(**search["failure_drops"])
    repaired_baseline = evaluate_perturbation(model, loader, device, "rotation", 0, repair["seed"])
    experiments = {}
    print(f"Comparing repaired model on {device}...")
    for name, severities in search["severities"].items():
        results = []
        for severity in severities:
            metrics = evaluate_perturbation(
                model, loader, device, name, float(severity), repair["seed"]
            )
            drop = repaired_baseline["accuracy"] - metrics["accuracy"]
            results.append(
                {
                    "severity": float(severity),
                    "accuracy_drop": drop,
                    "failure_level": classify_drop(drop, levels),
                    **metrics,
                }
            )
        experiments[name] = {
            "boundaries": find_boundaries(results, repaired_baseline["accuracy"], levels),
            "results": results,
        }
        last = results[-1]
        print(f"  {name}: max-severity accuracy={last['accuracy']:.4f}")

    repaired_analysis = {
        "model": checkpoint["model_class"],
        "dataset": "Fashion-MNIST test split",
        "sample_count": len(dataset),
        "failure_drops": search["failure_drops"],
        "baseline": repaired_baseline,
        "experiments": experiments,
    }
    comparison = {
        "repair_plan": repair_plan,
        "baseline": baseline_analysis,
        "repaired": repaired_analysis,
    }
    output = Path(search["output_dir"])
    save_comparison_json(output / "repair_comparison.json", comparison)
    save_comparison_plot(output / "repair_comparison.png", baseline_analysis, repaired_analysis)
    write_comparison_report(output / "repair_report.md", comparison)
    print(f"Comparison report written to: {output / 'repair_report.md'}")


if __name__ == "__main__":
    main()
