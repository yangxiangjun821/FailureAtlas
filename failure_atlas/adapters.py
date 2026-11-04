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
from .data import (
    DATASET_PROFILES,
    FASHION_MNIST_CLASSES,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    build_dataset_loaders,
    build_holdout_loaders,
    build_loaders,
    build_raw_test_loader,
    build_repair_loaders,
)
from .model import (
    ConvNeXtTinyCIFAR,
    EfficientNetB0CIFAR,
    FashionCNN,
    LeNet,
    MobileNetV3CIFAR,
    ResNet18CIFAR,
    ViTCIFAR,
)
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

    @property
    @abstractmethod
    def class_names(self) -> tuple[str, ...]: ...

    @property
    @abstractmethod
    def normalization(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Per-channel (mean, std) applied after perturbations."""

    @abstractmethod
    def raw_test_loader(
        self, root: str | Path, batch_size: int, num_workers: int = 0
    ) -> DataLoader:
        """Un-normalized [0, 1] test images for perturbation scans."""

    @abstractmethod
    def holdout_loaders(
        self,
        root: str | Path,
        batch_size: int,
        num_workers: int,
        val_fraction: float,
        seed: int,
    ) -> tuple[DataLoader, DataLoader]:
        """Clean train and validation loaders; the validation split is held out of training."""

    @abstractmethod
    def repair_loaders(
        self,
        root: str | Path,
        train_transform,
        batch_size: int,
        num_workers: int,
        val_fraction: float,
        seed: int,
    ) -> tuple[DataLoader, DataLoader]:
        """Augmented train loader and a clean validation loader split from train."""

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


class LeNetAdapter(FashionCNNAdapter):
    @property
    def spec(self) -> ModelSpec:
        return MODEL_REGISTRY.get("lenet")

    def build(self, config: Mapping | None = None) -> nn.Module:
        num_classes = int((config or {}).get("num_classes", self.spec.num_classes))
        return LeNet(num_classes=num_classes)


class ResNet18Adapter(FashionCNNAdapter):
    @property
    def spec(self) -> ModelSpec:
        return MODEL_REGISTRY.get("resnet18")

    def build(self, config: Mapping | None = None) -> nn.Module:
        num_classes = int((config or {}).get("num_classes", self.spec.num_classes))
        return ResNet18CIFAR(num_classes=num_classes)


class MobileNetV3Adapter(FashionCNNAdapter):
    @property
    def spec(self) -> ModelSpec:
        return MODEL_REGISTRY.get("mobilenet_v3")

    def build(self, config: Mapping | None = None) -> nn.Module:
        return MobileNetV3CIFAR(int((config or {}).get("num_classes", self.spec.num_classes)))


class EfficientNetB0Adapter(FashionCNNAdapter):
    @property
    def spec(self) -> ModelSpec:
        return MODEL_REGISTRY.get("efficientnet_b0")

    def build(self, config: Mapping | None = None) -> nn.Module:
        return EfficientNetB0CIFAR(int((config or {}).get("num_classes", self.spec.num_classes)))


class ConvNeXtTinyAdapter(FashionCNNAdapter):
    @property
    def spec(self) -> ModelSpec:
        return MODEL_REGISTRY.get("convnext_tiny")

    def build(self, config: Mapping | None = None) -> nn.Module:
        return ConvNeXtTinyCIFAR(int((config or {}).get("num_classes", self.spec.num_classes)))


class ViTAdapter(FashionCNNAdapter):
    @property
    def spec(self) -> ModelSpec:
        return MODEL_REGISTRY.get("vit")

    def build(self, config: Mapping | None = None) -> nn.Module:
        return ViTCIFAR(int((config or {}).get("num_classes", self.spec.num_classes)))


class FashionMNISTAdapter(DatasetAdapter):
    @property
    def spec(self) -> DatasetSpec:
        return DATASET_REGISTRY.get("fashion_mnist")

    def download(self, root: str | Path) -> None:
        build_dataset_loaders(self.spec.id, root, batch_size=128, download=True)

    def prepare(
        self,
        root: str | Path,
        batch_size: int,
        num_workers: int = 0,
        *,
        download: bool = True,
    ) -> tuple[DataLoader, DataLoader]:
        return build_dataset_loaders(
            self.spec.id, root, batch_size, num_workers, download
        )

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(DATASET_PROFILES[self.spec.id]["classes"])

    @property
    def normalization(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        profile = DATASET_PROFILES[self.spec.id]
        return tuple(profile["mean"]), tuple(profile["std"])

    def raw_test_loader(
        self, root: str | Path, batch_size: int, num_workers: int = 0
    ) -> DataLoader:
        return build_raw_test_loader(root, batch_size, num_workers, self.spec.id)

    def holdout_loaders(
        self,
        root: str | Path,
        batch_size: int,
        num_workers: int,
        val_fraction: float,
        seed: int,
    ) -> tuple[DataLoader, DataLoader]:
        return build_holdout_loaders(
            root, batch_size, num_workers, val_fraction, seed, self.spec.id
        )

    def repair_loaders(
        self,
        root: str | Path,
        train_transform,
        batch_size: int,
        num_workers: int,
        val_fraction: float,
        seed: int,
    ) -> tuple[DataLoader, DataLoader]:
        return build_repair_loaders(
            root,
            train_transform,
            batch_size,
            num_workers,
            val_fraction,
            seed,
            self.spec.id,
        )

    def metadata(self) -> dict:
        profile = DATASET_PROFILES[self.spec.id]
        return {
            "dataset_id": self.spec.id,
            "display_name": self.spec.display_name,
            "task": self.spec.task,
            "input_channels": self.spec.input_channels,
            "input_size": list(self.spec.input_size),
            "num_classes": self.spec.num_classes,
            "class_names": list(self.class_names),
            "normalization_mean": list(self.normalization[0]),
            "normalization_std": list(self.normalization[1]),
            "train_samples": profile["train_samples"],
            "test_samples": profile["test_samples"],
            "source": f"torchvision.datasets.{profile['class'].__name__}",
        }

    def sample_preview(self, root: str | Path, count: int = 8) -> list[dict]:
        _, test_loader = self.prepare(root, batch_size=max(1, count), download=False)
        images, labels = next(iter(test_loader))
        return [
            {
                "index": index,
                "shape": list(images[index].shape),
                "label": int(labels[index]),
                "class_name": self.class_names[int(labels[index])],
            }
            for index in range(min(count, len(labels)))
        ]


class MNISTAdapter(FashionMNISTAdapter):
    @property
    def spec(self) -> DatasetSpec:
        return DATASET_REGISTRY.get("mnist")


class CIFAR10Adapter(FashionMNISTAdapter):
    @property
    def spec(self) -> DatasetSpec:
        return DATASET_REGISTRY.get("cifar10")


def get_model_adapter(model_id: str) -> ModelAdapter:
    adapters: dict[str, type[ModelAdapter]] = {
        "fashion_cnn": FashionCNNAdapter,
        "lenet": LeNetAdapter,
        "resnet18": ResNet18Adapter,
        "mobilenet_v3": MobileNetV3Adapter,
        "efficientnet_b0": EfficientNetB0Adapter,
        "convnext_tiny": ConvNeXtTinyAdapter,
        "vit": ViTAdapter,
    }
    if model_id in adapters:
        return adapters[model_id]()
    raise NotImplementedError(f"No adapter implemented for model '{model_id}'.")


def get_dataset_adapter(dataset_id: str) -> DatasetAdapter:
    adapters: dict[str, type[DatasetAdapter]] = {
        "fashion_mnist": FashionMNISTAdapter,
        "mnist": MNISTAdapter,
        "cifar10": CIFAR10Adapter,
    }
    if dataset_id in adapters:
        return adapters[dataset_id]()
    raise NotImplementedError(f"No adapter implemented for dataset '{dataset_id}'.")
