"""Fashion-MNIST dataset loading."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


FASHION_MNIST_CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

NORMALIZE_MEAN = (0.2860,)
NORMALIZE_STD = (0.3530,)


def build_loaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int = 0,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Return deterministic train and test loaders for Fashion-MNIST."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)]
    )
    root = Path(data_dir)
    train_dataset = datasets.FashionMNIST(
        root=root, train=True, transform=transform, download=download
    )
    test_dataset = datasets.FashionMNIST(
        root=root, train=False, transform=transform, download=download
    )
    pin_memory = True
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader
