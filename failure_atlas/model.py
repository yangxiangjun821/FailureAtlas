"""Baseline convolutional classifier."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ViT_B_16_Weights,
    convnext_tiny,
    efficientnet_b0,
    mobilenet_v3_small,
    resnet18,
    vit_b_16,
)
from torchvision.models.vision_transformer import VisionTransformer


PRETRAINED_WEIGHTS = {
    "mobilenet_v3": MobileNet_V3_Small_Weights.DEFAULT,
    "efficientnet_b0": EfficientNet_B0_Weights.DEFAULT,
    "convnext_tiny": ConvNeXt_Tiny_Weights.DEFAULT,
    "vit_b16": ViT_B_16_Weights.DEFAULT,
}


def pretrained_weight_cache_path(model_id: str) -> Path | None:
    weights = PRETRAINED_WEIGHTS.get(model_id)
    if weights is None:
        return None
    filename = Path(urlparse(weights.url).path).name
    return Path(torch.hub.get_dir()) / "checkpoints" / filename


def pretrained_weight_cached(model_id: str) -> bool:
    path = pretrained_weight_cache_path(model_id)
    return bool(path and path.is_file())


def pretrained_weight_metadata(model_id: str) -> dict | None:
    """Return the exact official weight contract without loading the checkpoint."""

    weights = PRETRAINED_WEIGHTS.get(model_id)
    if weights is None:
        return None
    transforms = weights.transforms()
    return {
        "source": "torchvision",
        "enum": f"{type(weights).__name__}.{weights.name}",
        "url": weights.url,
        "cache_path": str(pretrained_weight_cache_path(model_id)),
        "input_size": [224, 224],
        "resize_size": list(transforms.resize_size),
        "crop_size": list(transforms.crop_size),
        "interpolation": str(transforms.interpolation),
        "mean": list(transforms.mean),
        "std": list(transforms.std),
    }


class ImageNetInputAdapter(nn.Module):
    """Convert CIFAR-normalized 32×32 inputs to the official ImageNet contract."""

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network
        self.register_buffer("source_mean", torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1))
        self.register_buffer("source_std", torch.tensor((0.2470, 0.2435, 0.2616)).view(1, 3, 1, 1))
        self.register_buffer("target_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("target_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raw = inputs * self.source_std + self.source_mean
        resized = F.interpolate(raw, size=(224, 224), mode="bilinear", align_corners=False)
        normalized = (resized - self.target_mean) / self.target_std
        return self.network(normalized)


def _weights_for(model_id: str, weight_mode: str, initialize_weights: bool):
    if weight_mode == "scratch":
        return None
    if weight_mode != "pretrained":
        raise ValueError("weight_mode must be 'scratch' or 'pretrained'.")
    try:
        weights = PRETRAINED_WEIGHTS[model_id]
    except KeyError as error:
        raise ValueError(f"Model '{model_id}' does not provide pretrained weights.") from error
    return weights if initialize_weights else None


class FashionCNN(nn.Module):
    """A compact CNN that is fast enough for local failure analysis."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class LeNet(nn.Module):
    """LeNet-style classifier for 28×28 grayscale datasets."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class ResNet18CIFAR(nn.Module):
    """ResNet-18 adjusted for 32×32 RGB inputs without ImageNet downsampling."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        network = resnet18(weights=None, num_classes=num_classes)
        network.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        network.maxpool = nn.Identity()
        self.network = network

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class MobileNetV3CIFAR(nn.Module):
    """MobileNetV3-Small classifier using the official TorchVision backbone."""

    def __init__(
        self,
        num_classes: int = 10,
        weight_mode: str = "scratch",
        initialize_weights: bool = True,
    ) -> None:
        super().__init__()
        self.weight_mode = weight_mode
        weights = _weights_for("mobilenet_v3", weight_mode, initialize_weights)
        network = mobilenet_v3_small(weights=weights)
        network.classifier[3] = nn.Linear(network.classifier[3].in_features, num_classes)
        self.network = ImageNetInputAdapter(network) if weight_mode == "pretrained" else network

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class EfficientNetB0CIFAR(nn.Module):
    """EfficientNet-B0 classifier adapted to the CIFAR-10 label space."""

    def __init__(
        self,
        num_classes: int = 10,
        weight_mode: str = "scratch",
        initialize_weights: bool = True,
    ) -> None:
        super().__init__()
        self.weight_mode = weight_mode
        weights = _weights_for("efficientnet_b0", weight_mode, initialize_weights)
        network = efficientnet_b0(weights=weights)
        network.classifier[1] = nn.Linear(network.classifier[1].in_features, num_classes)
        self.network = ImageNetInputAdapter(network) if weight_mode == "pretrained" else network

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class ConvNeXtTinyCIFAR(nn.Module):
    """ConvNeXt-Tiny classifier adapted to 32×32 RGB inputs."""

    def __init__(
        self,
        num_classes: int = 10,
        weight_mode: str = "scratch",
        initialize_weights: bool = True,
    ) -> None:
        super().__init__()
        self.weight_mode = weight_mode
        weights = _weights_for("convnext_tiny", weight_mode, initialize_weights)
        network = convnext_tiny(weights=weights)
        network.classifier[2] = nn.Linear(network.classifier[2].in_features, num_classes)
        self.network = ImageNetInputAdapter(network) if weight_mode == "pretrained" else network

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class ViTCIFAR(nn.Module):
    """Compact Vision Transformer with 4×4 patches for CIFAR-sized images.

    Defined for training from scratch only. The official ImageNet ViT is a different
    network and lives under its own id (``vit_b16``).
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.weight_mode = "scratch"
        self.network = VisionTransformer(
            image_size=32,
            patch_size=4,
            num_layers=6,
            num_heads=6,
            hidden_dim=384,
            mlp_dim=1536,
            dropout=0.0,
            attention_dropout=0.0,
            num_classes=num_classes,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class ViTB16CIFAR(nn.Module):
    """Official ImageNet ViT-B/16 with a new CIFAR-10 head; pretrained weights only."""

    def __init__(
        self,
        num_classes: int = 10,
        weight_mode: str = "pretrained",
        initialize_weights: bool = True,
    ) -> None:
        super().__init__()
        if weight_mode != "pretrained":
            raise ValueError("vit_b16 is only defined with official pretrained weights.")
        self.weight_mode = weight_mode
        weights = _weights_for("vit_b16", weight_mode, initialize_weights)
        network = vit_b_16(weights=weights)
        network.heads.head = nn.Linear(network.heads.head.in_features, num_classes)
        self.network = ImageNetInputAdapter(network)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


MODEL_BUILDERS: dict[str, type[nn.Module]] = {
    "fashion_cnn": FashionCNN,
    "lenet": LeNet,
    "resnet18": ResNet18CIFAR,
    "mobilenet_v3": MobileNetV3CIFAR,
    "efficientnet_b0": EfficientNetB0CIFAR,
    "convnext_tiny": ConvNeXtTinyCIFAR,
    "vit": ViTCIFAR,
    "vit_b16": ViTB16CIFAR,
}


def build_model(
    model_id: str,
    num_classes: int = 10,
    weight_mode: str = "scratch",
    initialize_weights: bool = True,
) -> nn.Module:
    try:
        model_class = MODEL_BUILDERS[model_id]
    except KeyError as error:
        raise KeyError(
            f"Unknown model '{model_id}'. Available: {', '.join(MODEL_BUILDERS)}"
        ) from error
    if model_id in PRETRAINED_WEIGHTS:
        return model_class(
            num_classes=num_classes,
            weight_mode=weight_mode,
            initialize_weights=initialize_weights,
        )
    if weight_mode != "scratch":
        raise ValueError(f"Model '{model_id}' does not provide pretrained weights.")
    return model_class(num_classes=num_classes)
