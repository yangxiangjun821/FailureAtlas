"""Fashion-MNIST dataset loading."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
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

MNIST_CLASSES = tuple(str(index) for index in range(10))
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

NORMALIZE_MEAN = (0.2860,)
NORMALIZE_STD = (0.3530,)

DATASET_PROFILES = {
    "fashion_mnist": {
        "class": datasets.FashionMNIST,
        "classes": FASHION_MNIST_CLASSES,
        "mean": NORMALIZE_MEAN,
        "std": NORMALIZE_STD,
        "train_samples": 60_000,
        "test_samples": 10_000,
    },
    "mnist": {
        "class": datasets.MNIST,
        "classes": MNIST_CLASSES,
        "mean": (0.1307,),
        "std": (0.3081,),
        "train_samples": 60_000,
        "test_samples": 10_000,
    },
    "cifar10": {
        "class": datasets.CIFAR10,
        "classes": CIFAR10_CLASSES,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "train_samples": 50_000,
        "test_samples": 10_000,
    },
}


def _pin_memory() -> bool:
    return torch.cuda.is_available()


def build_loaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int = 0,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Return deterministic train and test loaders for Fashion-MNIST."""
    return build_dataset_loaders(
        "fashion_mnist", data_dir, batch_size, num_workers, download=download
    )


def build_dataset_loaders(
    dataset_id: str,
    data_dir: str | Path,
    batch_size: int,
    num_workers: int = 0,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Return normalized train/test loaders for a registered image dataset."""
    try:
        profile = DATASET_PROFILES[dataset_id]
    except KeyError as error:
        raise KeyError(
            f"Unknown dataset '{dataset_id}'. Available: {', '.join(DATASET_PROFILES)}"
        ) from error
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(profile["mean"], profile["std"])]
    )
    root = Path(data_dir)
    dataset_class = profile["class"]
    train_dataset = dataset_class(
        root=root, train=True, transform=transform, download=download
    )
    test_dataset = dataset_class(
        root=root, train=False, transform=transform, download=download
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=_pin_memory(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin_memory(),
    )
    return train_loader, test_loader


def build_raw_test_loader(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int = 0,
    dataset_id: str = "fashion_mnist",
) -> DataLoader:
    """Test loader yielding un-normalized [0, 1] images for perturbation scans."""
    profile = DATASET_PROFILES[dataset_id]
    dataset = profile["class"](
        data_dir, train=False, transform=transforms.ToTensor(), download=False
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin_memory(),
    )


def split_indices(size: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Deterministically split ``range(size)`` into train and validation indices."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed)).tolist()
    val_size = max(1, round(size * val_fraction))
    return order[val_size:], order[:val_size]


def build_repair_loaders(
    data_dir: str | Path,
    train_transform,
    batch_size: int,
    num_workers: int,
    val_fraction: float,
    seed: int,
    dataset_id: str = "fashion_mnist",
) -> tuple[DataLoader, DataLoader]:
    """Augmented train loader plus a clean validation loader carved from the train split.

    Checkpoint selection uses the validation loader so the test split is never
    consulted while the repair is being fitted.
    """
    profile = DATASET_PROFILES[dataset_id]
    clean_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(profile["mean"], profile["std"])]
    )
    dataset_class = profile["class"]
    augmented: Dataset = dataset_class(
        data_dir, train=True, transform=train_transform, download=False
    )
    clean: Dataset = dataset_class(
        data_dir, train=True, transform=clean_transform, download=False
    )
    train_indices, val_indices = split_indices(len(augmented), val_fraction, seed)
    train_loader = DataLoader(
        Subset(augmented, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=_pin_memory(),
    )
    val_loader = DataLoader(
        Subset(clean, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin_memory(),
    )
    return train_loader, val_loader


def build_holdout_loaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int,
    val_fraction: float,
    seed: int,
    dataset_id: str = "fashion_mnist",
) -> tuple[DataLoader, DataLoader]:
    """Clean train and validation loaders for baseline training.

    Uses the same split as ``build_repair_loaders`` for equal ``val_fraction`` and
    ``seed``, so the validation images are unseen by both the baseline and the repair.
    """
    profile = DATASET_PROFILES[dataset_id]
    clean_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(profile["mean"], profile["std"])]
    )
    return build_repair_loaders(
        data_dir, clean_transform, batch_size, num_workers, val_fraction, seed, dataset_id
    )


def load_unlabeled_sample(
    data_dir: str | Path, count: int, dataset_id: str = "fashion_mnist"
) -> torch.Tensor:
    """First ``count`` training images in [0, 1], labels discarded.

    Used to measure pixel-space distortion, which needs no labels, so no test image
    is involved.
    """
    profile = DATASET_PROFILES[dataset_id]
    dataset = profile["class"](
        data_dir, train=True, transform=transforms.ToTensor(), download=False
    )
    return torch.stack([dataset[index][0] for index in range(min(count, len(dataset)))])
