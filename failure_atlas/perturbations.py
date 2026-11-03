"""Deterministic image perturbations used to probe model failure boundaries."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torchvision.transforms import functional as vision_functional


TensorPerturbation = Callable[[torch.Tensor, float, torch.Generator | None], torch.Tensor]


def rotation(
    images: torch.Tensor, severity: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    if severity == 0:
        return images
    return vision_functional.rotate(images, angle=float(severity), fill=0.0)


def gaussian_noise(
    images: torch.Tensor, severity: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    if severity == 0:
        return images
    noise = torch.randn(
        images.shape, device=images.device, dtype=images.dtype, generator=generator
    )
    return (images + noise * float(severity)).clamp(0.0, 1.0)


def blur(
    images: torch.Tensor, severity: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    if severity == 0:
        return images
    sigma = max(float(severity), 0.1)
    kernel_size = max(3, int(math.ceil(sigma * 4)) | 1)
    return vision_functional.gaussian_blur(
        images, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma]
    )


def occlusion(
    images: torch.Tensor, severity: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Cover a centered square whose area is the requested image fraction."""
    if severity == 0:
        return images
    result = images.clone()
    height, width = images.shape[-2:]
    side = max(1, round(math.sqrt(float(severity)) * min(height, width)))
    top = (height - side) // 2
    left = (width - side) // 2
    result[..., top : top + side, left : left + side] = 0.0
    return result


def brightness(
    images: torch.Tensor, severity: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Apply a brightness factor: 1.0 is clean and smaller values darken."""
    return vision_functional.adjust_brightness(images, float(severity)).clamp(0.0, 1.0)


PERTURBATIONS: dict[str, TensorPerturbation] = {
    "rotation": rotation,
    "gaussian_noise": gaussian_noise,
    "blur": blur,
    "occlusion": occlusion,
    "brightness": brightness,
}
