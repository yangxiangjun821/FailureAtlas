"""Stable adapter interfaces between the agent and experiment implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .catalog import DATASET_REGISTRY, MODEL_REGISTRY, DatasetSpec, ModelSpec
from .config import TrainConfig
from .data import FASHION_MNIST_CLASSES, build_loaders
from .model import FashionCNN
from .training import evaluate, load_checkpoint, train_one_epoch


class ModelAdapter(ABC):
    @property
    @abstractmethod
    def spec(self) -> ModelSpec: ...

    @abstractmethod
    def build(self, config: Mapping | None = None) -> nn.Module: ...

    @abstractmethod
    def load_checkpoint(self, path: str | Path, device: torch.device) -> nn.Module: ...

    @abstractmethod
    def train(
        self,
        model: nn.Module,
        loader: DataLoader,
        config: TrainConfig,
        device: torch.device,
    ) -> list[dict[str, float]]: ...

    @abstractmethod
    def predict(self, model: nn.Module, batch: torch.Tensor, device: torch.device) -> torch.Tensor: ...

    @abstractmethod
    def evaluate(
        self, model: nn.Module, loader: DataLoader, device: torch.device
    ) -> dict[str, float]: ...

    @abstractmethod
    def export_metadata(self, model: nn.Module) -> dict: ...


class DatasetAdapter(ABC):
    @property
    @abstractmethod
    def spec(self) -> DatasetSpec: ...

    @abstractmethod
    def download(self, root: str | Path) -> None: ...

    @abstractmethod
    def prepare(
        self,
        root: str | Path,
        batch_size: int,
        num_workers: int = 0,
        *,
        download: bool = True,
    ) -> tuple[DataLoader, DataLoader]: ...

    @abstractmethod
    def metadata(self) -> dict: ...

    def compatible_models(self) -> tuple[str, ...]:
        return self.spec.models

    def perturbations(self) -> tuple[str, ...]:
        return self.spec.perturbations

    @abstractmethod
    def sample_preview(self, root: str | Path, count: int = 8) -> list[dict]: ...


class FashionCNNAdapter(ModelAdapter):
    @property
    def spec(self) -> ModelSpec:
        return MODEL_REGISTRY.get("fashion_cnn")

    def build(self, config: Mapping | None = None) -> nn.Module:
        num_classes = int((config or {}).get("num_classes", self.spec.num_classes))
        return FashionCNN(num_classes=num_classes)

    def load_checkpoint(self, path: str | Path, device: torch.device) -> nn.Module:
        model, _ = load_checkpoint(path, device)
        return model

    def train(
        self,
        model: nn.Module,
        loader: DataLoader,
        config: TrainConfig,
        device: torch.device,
    ) -> list[dict[str, float]]:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        history = []
        for epoch in range(1, config.epochs + 1):
            loss = train_one_epoch(model, loader, optimizer, criterion, device)
            history.append({"epoch": float(epoch), "train_loss": loss})
        return history

    @torch.inference_mode()
    def predict(self, model: nn.Module, batch: torch.Tensor, device: torch.device) -> torch.Tensor:
        model.eval()
        return model(batch.to(device)).argmax(dim=1).cpu()

    def evaluate(
        self, model: nn.Module, loader: DataLoader, device: torch.device
    ) -> dict[str, float]:
        return evaluate(model, loader, device)

    def export_metadata(self, model: nn.Module) -> dict:
        return {
            "model_id": self.spec.id,
            "display_name": self.spec.display_name,
            "architecture": self.spec.architecture,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "input_channels": self.spec.input_channels,
            "input_size": list(self.spec.input_size),
            "num_classes": self.spec.num_classes,
        }


class FashionMNISTAdapter(DatasetAdapter):
    @property
    def spec(self) -> DatasetSpec:
        return DATASET_REGISTRY.get("fashion_mnist")

    def download(self, root: str | Path) -> None:
        build_loaders(root, batch_size=128, download=True)

    def prepare(
        self,
        root: str | Path,
        batch_size: int,
        num_workers: int = 0,
        *,
        download: bool = True,
    ) -> tuple[DataLoader, DataLoader]:
        return build_loaders(root, batch_size, num_workers, download)

    def metadata(self) -> dict:
        return {
            "dataset_id": self.spec.id,
            "display_name": self.spec.display_name,
            "task": self.spec.task,
            "input_channels": self.spec.input_channels,
            "input_size": list(self.spec.input_size),
            "num_classes": self.spec.num_classes,
            "class_names": list(FASHION_MNIST_CLASSES),
            "train_samples": 60_000,
            "test_samples": 10_000,
            "source": "torchvision.datasets.FashionMNIST",
        }

    def sample_preview(self, root: str | Path, count: int = 8) -> list[dict]:
        _, test_loader = self.prepare(root, batch_size=max(1, count), download=False)
        images, labels = next(iter(test_loader))
        return [
            {
                "index": index,
                "shape": list(images[index].shape),
                "label": int(labels[index]),
                "class_name": FASHION_MNIST_CLASSES[int(labels[index])],
            }
            for index in range(min(count, len(labels)))
        ]


def get_model_adapter(model_id: str) -> ModelAdapter:
    if model_id == "fashion_cnn":
        return FashionCNNAdapter()
    raise NotImplementedError(f"No adapter implemented for model '{model_id}'.")


def get_dataset_adapter(dataset_id: str) -> DatasetAdapter:
    if dataset_id == "fashion_mnist":
        return FashionMNISTAdapter()
    raise NotImplementedError(f"No adapter implemented for dataset '{dataset_id}'.")
