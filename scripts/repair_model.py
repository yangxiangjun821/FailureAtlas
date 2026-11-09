"""Let the repair agent choose augmentations and fine-tune the baseline model."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
import yaml
from torch import nn

from failure_atlas.adapters import get_dataset_adapter, get_model_adapter
from failure_atlas.catalog import resolve_weight_mode, validate_selection
from failure_atlas.config import check_weight_mode, effective_batch_size, effective_learning_rate
from failure_atlas.config import TrainConfig
from failure_atlas.paths import experiment_paths
from failure_atlas.repair_agent import TargetedAugment, create_repair_plan
from failure_atlas.training import (
    evaluate,
    fit_batch_size,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    set_seed,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/repair.yaml")
    parser.add_argument("--model", default="fashion_cnn")
    parser.add_argument("--dataset", default="fashion_mnist")
    parser.add_argument("--weight-mode", choices=("scratch", "pretrained"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_selection(args.model, args.dataset)
    model_adapter = get_model_adapter(args.model)
    dataset_adapter = get_dataset_adapter(args.dataset)
    settings = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["repair"]
    if args.seed is not None:
        settings["seed"] = args.seed
    weight_mode = resolve_weight_mode(args.model, args.weight_mode)
    settings["batch_size"] = effective_batch_size(
        args.batch_size or settings["batch_size"],
        weight_mode,
        explicit=args.batch_size is not None,
    )
    settings["learning_rate"] = effective_learning_rate(settings["learning_rate"], weight_mode)
    paths = experiment_paths(args.model, args.dataset, weight_mode)
    analysis = json.loads(paths.failure_analysis.read_text(encoding="utf-8"))
    plan = create_repair_plan(analysis, settings["top_k"])
    plan_path = paths.repair_plan
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    print("Agent selected targeted repairs:")
    for action in plan["actions"]:
        print(
            f"  {action['perturbation']}: score={action['vulnerability_score']:.4f}, "
            f"p={action['probability']}, max={action['max_severity']} "
            f"({action['severity_source']})"
        )

    set_seed(settings["seed"])
    device = resolve_device(settings["device"])
    model, baseline_payload = load_checkpoint(paths.baseline_checkpoint, device)
    check_weight_mode(model, weight_mode, paths.baseline_checkpoint)
    baseline_config = baseline_payload["config"]
    if "val_fraction" not in baseline_config:
        raise RuntimeError(
            "This baseline was trained on the full training split, so it has seen every "
            "validation image and validation accuracy cannot guide checkpoint selection. "
            "Retrain it: python -m scripts.train_baseline "
            f"--model {args.model} --dataset {args.dataset}"
        )
    # The split must match the baseline's exactly, otherwise validation leaks.
    val_fraction = baseline_config["val_fraction"]
    split_seed = baseline_config["seed"]
    fitted = fit_batch_size(model, settings["batch_size"], device, training=True)
    if fitted != settings["batch_size"]:
        print(f"Batch size reduced {settings['batch_size']} -> {fitted} to fit GPU memory")
        settings["batch_size"] = fitted
    mean, std = dataset_adapter.normalization
    # Checkpoint selection uses a validation split carved from the training data,
    # so the test split is only touched afterwards by compare_repair.
    train_loader, val_loader = dataset_adapter.repair_loaders(
        settings["data_dir"],
        TargetedAugment(plan, mean, std),
        settings["batch_size"],
        settings["num_workers"],
        val_fraction,
        split_seed,
    )
    print(
        f"Repair split: {len(train_loader.dataset)} train / "
        f"{len(val_loader.dataset)} validation images"
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    train_config = replace(
        TrainConfig(**baseline_payload["config"]),
        epochs=settings["epochs"],
        learning_rate=settings["learning_rate"],
        batch_size=settings["batch_size"],
    )
    best_accuracy = -1.0
    checkpoint_path = paths.repaired_checkpoint
    for epoch in range(1, settings["epochs"] + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = model_adapter.evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch:02d}/{settings['epochs']}: loss={loss:.4f}, "
            f"val_accuracy={metrics['accuracy']:.4f}"
        )
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            save_checkpoint(
                checkpoint_path,
                model,
                train_config,
                metrics,
                model_id=args.model,
                dataset_id=args.dataset,
            )
    restored, payload = load_checkpoint(checkpoint_path, device)
    restored_metrics = evaluate(restored, val_loader, device)
    print(
        f"Saved best repaired model: {checkpoint_path} "
        f"(validation accuracy={restored_metrics['accuracy']:.4f})"
    )
    if abs(restored_metrics["accuracy"] - payload["metrics"]["accuracy"]) > 1e-9:
        raise RuntimeError("Repaired checkpoint reload verification failed.")


if __name__ == "__main__":
    main()
