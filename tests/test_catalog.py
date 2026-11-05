import pytest
import torch

from failure_atlas.adapters import (
    FashionCNNAdapter,
    FashionMNISTAdapter,
    get_dataset_adapter,
    get_model_adapter,
)
from failure_atlas.catalog import (
    DATASET_REGISTRY,
    MODEL_REGISTRY,
    ModelSpec,
    Registry,
    validate_selection,
)


def test_catalog_distinguishes_ready_and_planned_entries() -> None:
    assert [spec.id for spec in MODEL_REGISTRY.values(implemented_only=True)] == [
        "fashion_cnn",
        "lenet",
        "resnet18",
        "mobilenet_v3",
        "efficientnet_b0",
        "convnext_tiny",
        "vit",
        "vit_b16",
    ]
    assert DATASET_REGISTRY.get("fashion_mnist").implemented is True
    assert DATASET_REGISTRY.get("mnist").implemented is True
    assert all(
        MODEL_REGISTRY.get(model_id).pretrained_available
        for model_id in ("mobilenet_v3", "efficientnet_b0", "convnext_tiny", "vit_b16")
    )
    assert MODEL_REGISTRY.get("resnet18").pretrained_available is False
    # One id names one architecture: the compact ViT is scratch-only and the official
    # ViT-B/16 is pretrained-only, so weight modes of one id stay comparable.
    assert MODEL_REGISTRY.get("vit").pretrained_available is False
    assert MODEL_REGISTRY.get("vit_b16").scratch_available is False


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


@pytest.mark.parametrize(
    ("model_id", "shape"),
    [
        ("fashion_cnn", (2, 1, 28, 28)),
        ("lenet", (2, 1, 28, 28)),
        ("resnet18", (2, 3, 32, 32)),
        ("mobilenet_v3", (2, 3, 32, 32)),
        ("efficientnet_b0", (2, 3, 32, 32)),
        ("convnext_tiny", (2, 3, 32, 32)),
        ("vit", (2, 3, 32, 32)),
    ],
)
def test_all_p0_models_produce_ten_class_logits(model_id: str, shape: tuple[int, ...]) -> None:
    model = get_model_adapter(model_id).build().eval()
    with torch.inference_mode():
        assert model(torch.zeros(shape)).shape == (2, 10)


def test_all_p0_datasets_expose_preprocessing_metadata() -> None:
    for dataset_id in ("mnist", "fashion_mnist", "cifar10"):
        adapter = get_dataset_adapter(dataset_id)
        metadata = adapter.metadata()
        assert len(metadata["normalization_mean"]) == adapter.spec.input_channels
        assert len(metadata["normalization_std"]) == adapter.spec.input_channels


def test_pretrained_adapter_contract_does_not_require_download() -> None:
    adapter = get_model_adapter("mobilenet_v3")
    model = adapter.build({"weight_mode": "pretrained", "initialize_weights": False}).eval()
    metadata = adapter.export_metadata(model)
    assert metadata["weight_mode"] == "pretrained"
    assert metadata["pretrained_weights"]["source"] == "torchvision"
    assert metadata["pretrained_weights"]["input_size"] == [224, 224]
    with torch.inference_mode():
        assert model(torch.zeros(1, 3, 32, 32)).shape == (1, 10)


def test_vit_b16_exists_only_with_pretrained_weights() -> None:
    adapter = get_model_adapter("vit_b16")
    model = adapter.build({"weight_mode": "pretrained", "initialize_weights": False}).eval()
    assert adapter.export_metadata(model)["pretrained_weights"]["input_size"] == [224, 224]
    with torch.inference_mode():
        assert model(torch.zeros(1, 3, 32, 32)).shape == (1, 10)
    with pytest.raises(ValueError, match="only defined with official pretrained"):
        adapter.build({"weight_mode": "scratch"})


def test_compact_vit_rejects_pretrained_mode() -> None:
    with pytest.raises(ValueError, match="does not provide pretrained"):
        get_model_adapter("vit").build({"weight_mode": "pretrained"})


def test_weight_mode_validation_and_defaults() -> None:
    from failure_atlas.catalog import resolve_weight_mode, validate_weight_mode

    assert resolve_weight_mode("lenet") == "scratch"
    assert resolve_weight_mode("vit_b16") == "pretrained"
    assert resolve_weight_mode("mobilenet_v3", "pretrained") == "pretrained"
    for model_id, mode in (("vit", "pretrained"), ("vit_b16", "scratch"), ("lenet", "pretrained")):
        with pytest.raises(ValueError):
            validate_weight_mode(model_id, mode)
    with pytest.raises(ValueError, match="weight_mode must be"):
        validate_weight_mode("lenet", "random")
