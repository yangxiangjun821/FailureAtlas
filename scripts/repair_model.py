"""Let the repair agent choose augmentations and fine-tune the baseline model."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from failure_atlas.config import TrainConfig
from failure_atlas.data import NORMALIZE_MEAN, NORMALIZE_STD
from failure_atlas.repair_agent import TargetedAugment, create_repair_plan
from failure_atlas.training import (
    evaluate,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    set_seed,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/repair.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["repair"]
    analysis = json.loads(Path(settings["analysis"]).read_text(encoding="utf-8"))
    plan = create_repair_plan(analysis, settings["top_k"])
    plan_path = Path(settings["repair_plan"])
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    print("Agent selected targeted repairs:")
    for action in plan["actions"]:
        print(
            f"  {action['perturbation']}: score={action['vulnerability_score']:.4f}, "
            f"p={action['probability']}, max={action['max_severity']}"
        )

    set_seed(settings["seed"])
    device = resolve_device(settings["device"])
    model, baseline_payload = load_checkpoint(settings["baseline_checkpoint"], device)
    train_dataset = datasets.FashionMNIST(
        settings["data_dir"],
        train=True,
        transform=TargetedAugment(plan),
        download=False,
    )
    clean_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)]
    )
    test_dataset = datasets.FashionMNIST(
        settings["data_dir"], train=False, transform=clean_transform, download=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings["batch_size"],
        shuffle=True,
        num_workers=settings["num_workers"],
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=settings["batch_size"],
        shuffle=False,
        num_workers=settings["num_workers"],
        pin_memory=device.type == "cuda",
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
    checkpoint_path = Path(settings["repaired_checkpoint"])
    for epoch in range(1, settings["epochs"] + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch:02d}/{settings['epochs']}: loss={loss:.4f}, "
            f"clean_accuracy={metrics['accuracy']:.4f}"
        )
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            save_checkpoint(checkpoint_path, model, train_config, metrics)
    restored, payload = load_checkpoint(checkpoint_path, device)
    restored_metrics = evaluate(restored, test_loader, device)
    print(
        f"Saved best repaired model: {checkpoint_path} "
        f"(clean accuracy={restored_metrics['accuracy']:.4f})"
    )
    if abs(restored_metrics["accuracy"] - payload["metrics"]["accuracy"]) > 1e-9:
        raise RuntimeError("Repaired checkpoint reload verification failed.")


if __name__ == "__main__":
    main()
