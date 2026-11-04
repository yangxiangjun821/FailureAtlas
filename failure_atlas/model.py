"""Baseline convolutional classifier."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import convnext_tiny, efficientnet_b0, mobilenet_v3_small, resnet18
from torchvision.models.vision_transformer import VisionTransformer


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

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.network = mobilenet_v3_small(weights=None, num_classes=num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class EfficientNetB0CIFAR(nn.Module):
    """EfficientNet-B0 classifier adapted to the CIFAR-10 label space."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.network = efficientnet_b0(weights=None, num_classes=num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class ConvNeXtTinyCIFAR(nn.Module):
    """ConvNeXt-Tiny classifier adapted to 32×32 RGB inputs."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.network = convnext_tiny(weights=None, num_classes=num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class ViTCIFAR(nn.Module):
    """Compact Vision Transformer with 4×4 patches for CIFAR-sized images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
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


MODEL_BUILDERS: dict[str, type[nn.Module]] = {
    "fashion_cnn": FashionCNN,
    "lenet": LeNet,
    "resnet18": ResNet18CIFAR,
    "mobilenet_v3": MobileNetV3CIFAR,
    "efficientnet_b0": EfficientNetB0CIFAR,
    "convnext_tiny": ConvNeXtTinyCIFAR,
    "vit": ViTCIFAR,
}


def build_model(model_id: str, num_classes: int = 10) -> nn.Module:
    try:
        model_class = MODEL_BUILDERS[model_id]
    except KeyError as error:
        raise KeyError(
            f"Unknown model '{model_id}'. Available: {', '.join(MODEL_BUILDERS)}"
        ) from error
    return model_class(num_classes=num_classes)
