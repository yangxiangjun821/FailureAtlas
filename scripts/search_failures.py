"""Run the first complete FailureAtlas boundary-discovery experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import yaml

from failure_atlas.adapters import get_dataset_adapter, get_model_adapter
from failure_atlas.calibration import load_calibration
from failure_atlas.catalog import validate_selection
from failure_atlas.failure_search import (
    FailureLevels,
    classify_drop,
    evaluate_perturbation,
    find_boundaries,
)
from failure_atlas.paths import experiment_paths
from failure_atlas.severity import severities_for
from failure_atlas.reporting import plot_accuracy_curves, save_json, write_markdown_report
from failure_atlas.training import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/failure_search.yaml")
    parser.add_argument("--model", default="fashion_cnn")
    parser.add_argument("--dataset", default="fashion_mnist")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--perturbation", action="append", dest="perturbations")
    parser.add_argument("--severity-scale", type=float, default=1.0)
    return parser.parse_args()


def scale_severity(name: str, severity: float, scale: float) -> float:
    if name == "brightness":
        return max(0.0, min(1.0, 1.0 - (1.0 - severity) * scale))
    return severity * scale


def main() -> None:
    args = parse_args()
    selected = tuple(args.perturbations) if args.perturbations else None
    validate_selection(args.model, args.dataset, selected)
    model_adapter = get_model_adapter(args.model)
    dataset_adapter = get_dataset_adapter(args.dataset)
    calibration = load_calibration(args.dataset)
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["search"]
    if args.seed is not None:
        raw["seed"] = args.seed
    if args.batch_size is not None:
        raw["batch_size"] = args.batch_size
    if not 0.25 <= args.severity_scale <= 2.0:
        raise ValueError("severity-scale must be between 0.25 and 2.0")
    set_seed(raw["seed"])
    device = resolve_device(raw["device"])
    paths = experiment_paths(args.model, args.dataset)
    model = model_adapter.load_checkpoint(paths.baseline_checkpoint, device)
    loader = dataset_adapter.raw_test_loader(
        raw["data_dir"], raw["batch_size"], raw["num_workers"]
    )
    mean, std = dataset_adapter.normalization
    class_names = dataset_adapter.class_names
    levels = FailureLevels(**raw["failure_drops"])

    def scan(name: str, severity: float) -> dict:
        return evaluate_perturbation(
            model, loader, device, name, severity, raw["seed"], class_names, mean, std
        )

    sample_count = len(loader.dataset)
    print(f"Failure search on {device} with {sample_count} test images")
    baseline = scan("rotation", 0.0)
    experiments: dict[str, dict] = {}
    grids = severities_for(raw, calibration)
    for name, severities in grids.items():
        if selected and name not in selected:
            continue
        print(f"Scanning {name}...")
        results = []
        for severity in severities:
            severity = scale_severity(name, float(severity), args.severity_scale)
            metrics = scan(name, severity)
            drop = baseline["accuracy"] - metrics["accuracy"]
            entry = {
                "severity": float(severity),
                "strength": calibration.strength(name, severity),
                "rmse": calibration.rmse(name, severity),
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
        "model": type(model).__name__,
        "model_id": args.model,
        "dataset": f"{dataset_adapter.spec.display_name} test split",
        "dataset_id": args.dataset,
        "sample_count": sample_count,
        "failure_drops": raw["failure_drops"],
        "calibration": {
            "metric": calibration.metric,
            "reference_rmse": calibration.reference_rmse,
            "sample_count": calibration.sample_count,
        },
        "baseline": baseline,
        "experiments": experiments,
    }
    save_json(paths.failure_analysis, payload)
    plot_accuracy_curves(paths.failure_curves, experiments, baseline["accuracy"])
    write_markdown_report(paths.failure_report, payload)
    print(f"Report written to: {paths.report_dir}")


if __name__ == "__main__":
    main()
