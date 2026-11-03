"""Train and verify the Phase 1 baseline model."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from failure_atlas.config import load_train_config
from failure_atlas.data import build_loaders
from failure_atlas.model import FashionCNN
from failure_atlas.training import (
    evaluate,
    load_checkpoint,
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config)
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)

    set_seed(config.seed)
    device = resolve_device(config.device)
    print(f"Training on: {device}")
    train_loader, test_loader = build_loaders(
        config.data_dir, config.batch_size, config.num_workers, download=True
    )
    model = FashionCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, test_loader, device)
        epoch_metrics = {"epoch": epoch, "train_loss": loss, **metrics}
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch:02d}/{config.epochs}: loss={loss:.4f}, "
            f"accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}"
        )

    artifact_dir = Path(config.artifact_dir)
    checkpoint_path = artifact_dir / "baseline_fashion_cnn.pt"
    final_metrics = history[-1]
    save_checkpoint(checkpoint_path, model, config, final_metrics)
    write_metrics(artifact_dir / "baseline_metrics.json", {"history": history})

    reloaded_model, payload = load_checkpoint(checkpoint_path, device)
    reload_metrics = evaluate(reloaded_model, test_loader, device)
    if abs(reload_metrics["accuracy"] - final_metrics["accuracy"]) > 1e-9:
        raise RuntimeError("Reloaded model metrics do not match the saved model.")
    print(f"Saved and reloaded: {checkpoint_path}")
    print(f"Verified accuracy: {payload['metrics']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
