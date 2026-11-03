from pathlib import Path

import pytest
import torch

from failure_atlas.adapters import FashionCNNAdapter, FashionMNISTAdapter
from failure_atlas.catalog import (
    DATASET_REGISTRY,
    MODEL_REGISTRY,
    ModelSpec,
    Registry,
    validate_selection,
)


def test_catalog_distinguishes_ready_and_planned_entries() -> None:
    assert [spec.id for spec in MODEL_REGISTRY.values(implemented_only=True)] == [
        "fashion_cnn"
    ]
    assert DATASET_REGISTRY.get("fashion_mnist").implemented is True
    assert DATASET_REGISTRY.get("mnist").implemented is False


def test_registry_rejects_duplicate_ids() -> None:
    registry: Registry[ModelSpec] = Registry("model")
    spec = ModelSpec("tiny", "Tiny", "image", "cnn", 1, (28, 28), 10, ("demo",))
    registry.register(spec)
    with pytest.raises(ValueError, match="Duplicate model id"):
        registry.register(spec)


def test_selection_rejects_incompatible_or_unimplemented_combinations() -> None:
    validate_selection("fashion_cnn", "fashion_mnist")
    with pytest.raises(ValueError, match="not compatible"):
        validate_selection("fashion_cnn", "cifar10", require_implemented=False)
    with pytest.raises(NotImplementedError, match="model:lenet"):
        validate_selection("lenet", "mnist")


def test_fashion_adapters_export_stable_metadata() -> None:
    adapter = FashionCNNAdapter()
    model = adapter.build()
    metadata = adapter.export_metadata(model)
    assert metadata["model_id"] == "fashion_cnn"
    assert metadata["parameter_count"] == sum(p.numel() for p in model.parameters())
    assert adapter.predict(model, torch.zeros(2, 1, 28, 28), torch.device("cpu")).shape == (2,)

    dataset = FashionMNISTAdapter()
    assert dataset.compatible_models() == ("fashion_cnn", "lenet")
    assert dataset.metadata()["test_samples"] == 10_000
    assert isinstance(Path("data"), Path)
