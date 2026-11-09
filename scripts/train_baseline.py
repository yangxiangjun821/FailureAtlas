"""Train and verify the Phase 1 baseline model."""

from __future__ import annotations

import argparse
from dataclasses import replace

import torch
from torch import nn

from failure_atlas.adapters import get_dataset_adapter, get_model_adapter
from failure_atlas.catalog import resolve_weight_mode, validate_selection
from failure_atlas.config import effective_batch_size, effective_learning_rate, load_train_config
from failure_atlas.paths import experiment_paths
from failure_atlas.model import pretrained_weight_cache_path, pretrained_weight_cached
from failure_atlas.training import (
    evaluate,
    fit_batch_size,
    load_checkpoint,
    measure_inference_speed,
    resolve_device,
    save_checkpoint,
    set_seed,
    train_one_epoch,
    write_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--epochs", type=int, help="Override epochs for a smoke test.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--weight-mode", choices=("scratch", "pretrained"))
    parser.add_argument("--model", default="fashion_cnn")
    parser.add_argument("--dataset", default="fashion_mnist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_spec, _, _ = validate_selection(args.model, args.dataset)
    model_adapter = get_model_adapter(args.model)
    dataset_adapter = get_dataset_adapter(args.dataset)
    config = load_train_config(args.config)
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    if args.batch_size is not None:
        config = replace(config, batch_size=args.batch_size)
    weight_mode = resolve_weight_mode(args.model, args.weight_mode)
    config = replace(
        config,
        weight_mode=weight_mode,
        learning_rate=effective_learning_rate(config.learning_rate, weight_mode),
        batch_size=effective_batch_size(
            config.batch_size, weight_mode, explicit=args.batch_size is not None
        ),
    )
    if weight_mode == "pretrained":
        print(
            f"Pretrained fine-tuning: learning rate {config.learning_rate:g}, "
            f"batch size {config.batch_size}"
        )

    set_seed(config.seed)
    device = resolve_device(config.device)
    print(f"Training on: {device}")
    if config.weight_mode == "pretrained" and not pretrained_weight_cached(args.model):
        print(
            "Official pretrained weights are not cached; TorchVision will download "
            f"them once to: {pretrained_weight_cache_path(args.model)}"
        )
    try:
        model = model_adapter.build({"weight_mode": config.weight_mode}).to(device)
    except Exception as error:
        if config.weight_mode == "pretrained":
            raise RuntimeError(
                "Unable to load official pretrained weights. Check the network, or "
                f"place the checkpoint in {pretrained_weight_cache_path(args.model)}."
            ) from error
        raise
    # Shrink the batch if a measured training step would not fit in free GPU memory. An
    # explicit --batch-size is shrunk too: a size that spills into shared memory is never
    # what anyone wants. The fitted size is what gets recorded in the checkpoint config.
    sample_shape = (model_spec.input_channels, *model_spec.input_size)
    fitted = fit_batch_size(model, config.batch_size, device, training=True, sample_shape=sample_shape)
    if fitted != config.batch_size:
        print(f"Batch size reduced {config.batch_size} -> {fitted} to fit GPU memory")
        config = replace(config, batch_size=fitted)
    # prepare() also downloads the dataset, so it runs before the holdout split.
    _, test_loader = dataset_adapter.prepare(
        config.data_dir, config.batch_size, config.num_workers, download=True
    )
    train_loader, val_loader = dataset_adapter.holdout_loaders(
        config.data_dir,
        config.batch_size,
        config.num_workers,
        config.val_fraction,
        config.seed,
    )
    print(
        f"Baseline split: {len(train_loader.dataset)} train / "
        f"{len(val_loader.dataset)} validation held out"
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Per-epoch monitoring uses the held-out validation split; the test split is
    # evaluated exactly once, after training has finished.
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": loss, **metrics})
        print(
            f"Epoch {epoch:02d}/{config.epochs}: loss={loss:.4f}, "
            f"val_accuracy={metrics['accuracy']:.4f}, val_macro_f1={metrics['macro_f1']:.4f}"
        )

    paths = experiment_paths(args.model, args.dataset, weight_mode)
    checkpoint_path = paths.baseline_checkpoint
    metrics_path = paths.baseline_metrics
    final_metrics = evaluate(model, test_loader, device)
    final_metrics.update(measure_inference_speed(model, test_loader, device))
    print(
        f"Test accuracy (single final evaluation): {final_metrics['accuracy']:.4f}, "
        f"macro_f1={final_metrics['macro_f1']:.4f}"
    )
    save_checkpoint(
        checkpoint_path,
        model,
        config,
        final_metrics,
        model_id=args.model,
        dataset_id=args.dataset,
    )
    write_metrics(
        metrics_path,
        {
            "model": model_adapter.export_metadata(model),
            "dataset": dataset_adapter.metadata(),
            "split": {
                "train_samples": len(train_loader.dataset),
                "validation_samples": len(val_loader.dataset),
                "val_fraction": config.val_fraction,
                "seed": config.seed,
            },
            "history": history,
            "test": final_metrics,
        },
    )

    reloaded_model, payload = load_checkpoint(checkpoint_path, device)
    reload_metrics = evaluate(reloaded_model, test_loader, device)
    if abs(reload_metrics["accuracy"] - final_metrics["accuracy"]) > 1e-9:
        raise RuntimeError("Reloaded model metrics do not match the saved model.")
    print(f"Saved and reloaded: {checkpoint_path}")
    print(f"Verified accuracy: {payload['metrics']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
