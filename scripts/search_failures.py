"""Run the first complete FailureAtlas boundary-discovery experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from failure_atlas.failure_search import (
    FailureLevels,
    classify_drop,
    evaluate_perturbation,
    find_boundaries,
)
from failure_atlas.reporting import plot_accuracy_curves, save_json, write_markdown_report
from failure_atlas.training import load_checkpoint, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/failure_search.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["search"]
    set_seed(raw["seed"])
    device = resolve_device(raw["device"])
    model, checkpoint = load_checkpoint(raw["checkpoint"], device)
    dataset = datasets.FashionMNIST(
        raw["data_dir"], train=False, transform=transforms.ToTensor(), download=False
    )
    loader = DataLoader(
        dataset,
        batch_size=raw["batch_size"],
        shuffle=False,
        num_workers=raw["num_workers"],
        pin_memory=device.type == "cuda",
    )
    levels = FailureLevels(**raw["failure_drops"])

    print(f"Failure search on {device} with {len(dataset)} test images")
    baseline = evaluate_perturbation(model, loader, device, "rotation", 0.0, raw["seed"])
    experiments: dict[str, dict] = {}
    for name, severities in raw["severities"].items():
        print(f"Scanning {name}...")
        results = []
        for severity in severities:
            metrics = evaluate_perturbation(
                model, loader, device, name, float(severity), raw["seed"]
            )
            drop = baseline["accuracy"] - metrics["accuracy"]
            entry = {
                "severity": float(severity),
                "accuracy_drop": drop,
                "failure_level": classify_drop(drop, levels),
                **metrics,
            }
            results.append(entry)
            print(
                f"  severity={severity:g} accuracy={metrics['accuracy']:.4f} "
                f"drop={drop:.4f} level={entry['failure_level']}"
            )
        experiments[name] = {
            "boundaries": find_boundaries(results, baseline["accuracy"], levels),
            "results": results,
        }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": checkpoint["model_class"],
        "dataset": "Fashion-MNIST test split",
        "sample_count": len(dataset),
        "failure_drops": raw["failure_drops"],
        "baseline": baseline,
        "experiments": experiments,
    }
    output_dir = Path(raw["output_dir"])
    save_json(output_dir / "failure_analysis.json", payload)
    plot_accuracy_curves(output_dir / "failure_curves.png", experiments, baseline["accuracy"])
    write_markdown_report(output_dir / "failure_report.md", payload)
    print(f"Report written to: {output_dir}")


if __name__ == "__main__":
    main()
