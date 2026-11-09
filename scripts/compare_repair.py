"""Re-scan the repaired model and produce an evidence-based comparison report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from failure_atlas.adapters import get_dataset_adapter, get_model_adapter
from failure_atlas.calibration import load_calibration
from failure_atlas.catalog import resolve_weight_mode, validate_selection
from failure_atlas.config import check_weight_mode, effective_batch_size
from failure_atlas.comparison import (
    save_comparison_json,
    save_comparison_plot,
    write_comparison_report,
)
from failure_atlas.failure_search import FailureLevels, classify_drop, evaluate_perturbation, find_boundaries
from failure_atlas.paths import experiment_paths
from failure_atlas.severity import severities_for
from failure_atlas.training import fit_batch_size, resolve_device
from scripts.search_failures import scale_severity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="fashion_cnn")
    parser.add_argument("--dataset", default="fashion_mnist")
    parser.add_argument("--weight-mode", choices=("scratch", "pretrained"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--perturbation", action="append", dest="perturbations")
    parser.add_argument("--severity-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = tuple(args.perturbations) if args.perturbations else None
    model_spec, _, _ = validate_selection(args.model, args.dataset, selected)
    model_adapter = get_model_adapter(args.model)
    dataset_adapter = get_dataset_adapter(args.dataset)
    calibration = load_calibration(args.dataset)
    search = yaml.safe_load(Path("configs/failure_search.yaml").read_text(encoding="utf-8"))["search"]
    repair = yaml.safe_load(Path("configs/repair.yaml").read_text(encoding="utf-8"))["repair"]
    if args.seed is not None:
        repair["seed"] = args.seed
    weight_mode = resolve_weight_mode(args.model, args.weight_mode)
    search["batch_size"] = effective_batch_size(
        args.batch_size or search["batch_size"], weight_mode, explicit=args.batch_size is not None
    )
    if not 0.25 <= args.severity_scale <= 2.0:
        raise ValueError("severity-scale must be between 0.25 and 2.0")
    paths = experiment_paths(args.model, args.dataset, weight_mode)
    baseline_analysis = json.loads(paths.failure_analysis.read_text(encoding="utf-8"))
    repair_plan = json.loads(paths.repair_plan.read_text(encoding="utf-8"))
    device = resolve_device(repair["device"])
    model = model_adapter.load_checkpoint(paths.repaired_checkpoint, device)
    check_weight_mode(model, weight_mode, paths.repaired_checkpoint)
    sample_shape = (model_spec.input_channels, *model_spec.input_size)
    fitted = fit_batch_size(model, search["batch_size"], device, training=False, sample_shape=sample_shape)
    if fitted != search["batch_size"]:
        print(f"Batch size reduced {search['batch_size']} -> {fitted} to fit GPU memory")
        search["batch_size"] = fitted
    loader = dataset_adapter.raw_test_loader(
        repair["data_dir"], search["batch_size"], search["num_workers"]
    )
    mean, std = dataset_adapter.normalization
    class_names = dataset_adapter.class_names
    sample_count = len(loader.dataset)
    levels = FailureLevels(**search["failure_drops"])

    def scan(name: str, severity: float) -> dict:
        return evaluate_perturbation(
            model, loader, device, name, severity, repair["seed"], class_names, mean, std
        )

    repaired_baseline = scan("rotation", 0.0)
    experiments = {}
    print(f"Comparing repaired model on {device}...")
    for name, severities in severities_for(search, calibration).items():
        if selected and name not in selected:
            continue
        results = []
        for severity in severities:
            severity = scale_severity(name, float(severity), args.severity_scale)
            metrics = scan(name, severity)
            drop = repaired_baseline["accuracy"] - metrics["accuracy"]
            results.append(
                {
                    "severity": float(severity),
                    "strength": calibration.strength(name, severity),
                    "rmse": calibration.rmse(name, severity),
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
        "model": type(model).__name__,
        "model_id": args.model,
        "dataset": f"{dataset_adapter.spec.display_name} test split",
        "dataset_id": args.dataset,
        "sample_count": sample_count,
        "failure_drops": search["failure_drops"],
        "calibration": {
            "metric": calibration.metric,
            "reference_rmse": calibration.reference_rmse,
            "sample_count": calibration.sample_count,
        },
        "baseline": repaired_baseline,
        "experiments": experiments,
    }
    comparison = {
        "repair_plan": repair_plan,
        "baseline": baseline_analysis,
        "repaired": repaired_analysis,
    }
    save_comparison_json(paths.repair_comparison, comparison)
    save_comparison_plot(paths.repair_comparison_plot, baseline_analysis, repaired_analysis)
    write_comparison_report(paths.repair_report, comparison)
    print(f"Comparison report written to: {paths.repair_report}")


if __name__ == "__main__":
    main()
