"""Typed registries for models, datasets, and perturbations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Generic, TypeVar


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display_name: str
    task: str
    architecture: str
    input_channels: int
    input_size: tuple[int, int]
    num_classes: int
    datasets: tuple[str, ...]
    implemented: bool = True


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    display_name: str
    task: str
    input_channels: int
    input_size: tuple[int, int]
    num_classes: int
    models: tuple[str, ...]
    perturbations: tuple[str, ...]
    normalization_mean: tuple[float, ...]
    normalization_std: tuple[float, ...]
    implemented: bool = True


@dataclass(frozen=True)
class PerturbationSpec:
    id: str
    display_name: str
    task: str
    severity_levels: tuple[float, ...]
    implemented: bool = True


SpecT = TypeVar("SpecT", ModelSpec, DatasetSpec, PerturbationSpec)


class Registry(Generic[SpecT]):
    """Small deterministic registry with duplicate and unknown-ID checks."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, SpecT] = {}

    def register(self, spec: SpecT) -> SpecT:
        if spec.id in self._entries:
            raise ValueError(f"Duplicate {self.kind} id: {spec.id}")
        self._entries[spec.id] = spec
        return spec

    def get(self, item_id: str) -> SpecT:
        try:
            return self._entries[item_id]
        except KeyError as error:
            choices = ", ".join(self._entries) or "none"
            raise KeyError(f"Unknown {self.kind} '{item_id}'. Available: {choices}") from error

    def values(self, *, implemented_only: bool = False) -> tuple[SpecT, ...]:
        values = tuple(self._entries.values())
        if implemented_only:
            return tuple(spec for spec in values if spec.implemented)
        return values

    def as_dicts(self) -> list[dict]:
        return [asdict(spec) for spec in self.values()]


MODEL_REGISTRY: Registry[ModelSpec] = Registry("model")
DATASET_REGISTRY: Registry[DatasetSpec] = Registry("dataset")
PERTURBATION_REGISTRY: Registry[PerturbationSpec] = Registry("perturbation")


MODEL_REGISTRY.register(
    ModelSpec(
        id="fashion_cnn",
        display_name="FashionCNN",
        task="image_classification",
        architecture="compact_cnn",
        input_channels=1,
        input_size=(28, 28),
        num_classes=10,
        datasets=("fashion_mnist", "mnist"),
    )
)
MODEL_REGISTRY.register(
    ModelSpec(
        id="lenet",
        display_name="LeNet",
        task="image_classification",
        architecture="classic_cnn",
        input_channels=1,
        input_size=(28, 28),
        num_classes=10,
        datasets=("mnist", "fashion_mnist"),
    )
)
MODEL_REGISTRY.register(
    ModelSpec(
        id="resnet18",
        display_name="ResNet-18",
        task="image_classification",
        architecture="residual_cnn",
        input_channels=3,
        input_size=(32, 32),
        num_classes=10,
        datasets=("cifar10",),
    )
)
for modern_model in (
    ModelSpec(
        id="mobilenet_v3",
        display_name="MobileNetV3-Small",
        task="image_classification",
        architecture="mobile_cnn",
        input_channels=3,
        input_size=(32, 32),
        num_classes=10,
        datasets=("cifar10",),
    ),
    ModelSpec(
        id="efficientnet_b0",
        display_name="EfficientNet-B0",
        task="image_classification",
        architecture="compound_scaled_cnn",
        input_channels=3,
        input_size=(32, 32),
        num_classes=10,
        datasets=("cifar10",),
    ),
    ModelSpec(
        id="convnext_tiny",
        display_name="ConvNeXt-Tiny",
        task="image_classification",
        architecture="modern_convnet",
        input_channels=3,
        input_size=(32, 32),
        num_classes=10,
        datasets=("cifar10",),
    ),
    ModelSpec(
        id="vit",
        display_name="ViT-CIFAR",
        task="image_classification",
        architecture="vision_transformer",
        input_channels=3,
        input_size=(32, 32),
        num_classes=10,
        datasets=("cifar10",),
    ),
):
    MODEL_REGISTRY.register(modern_model)


IMAGE_PERTURBATIONS = ("rotation", "gaussian_noise", "blur", "occlusion", "brightness")

DATASET_REGISTRY.register(
    DatasetSpec(
        id="fashion_mnist",
        display_name="Fashion-MNIST",
        task="image_classification",
        input_channels=1,
        input_size=(28, 28),
        num_classes=10,
        models=("fashion_cnn", "lenet"),
        perturbations=IMAGE_PERTURBATIONS,
        normalization_mean=(0.2860,),
        normalization_std=(0.3530,),
    )
)
DATASET_REGISTRY.register(
    DatasetSpec(
        id="mnist",
        display_name="MNIST",
        task="image_classification",
        input_channels=1,
        input_size=(28, 28),
        num_classes=10,
        models=("fashion_cnn", "lenet"),
        perturbations=IMAGE_PERTURBATIONS,
        normalization_mean=(0.1307,),
        normalization_std=(0.3081,),
    )
)
DATASET_REGISTRY.register(
    DatasetSpec(
        id="cifar10",
        display_name="CIFAR-10",
        task="image_classification",
        input_channels=3,
        input_size=(32, 32),
        num_classes=10,
        models=(
            "resnet18",
            "mobilenet_v3",
            "efficientnet_b0",
            "convnext_tiny",
            "vit",
        ),
        perturbations=IMAGE_PERTURBATIONS,
        normalization_mean=(0.4914, 0.4822, 0.4465),
        normalization_std=(0.2470, 0.2435, 0.2616),
    )
)


for perturbation in (
    PerturbationSpec(
        "rotation", "Rotation", "image_classification", (0, 5, 10, 15, 20, 30, 45, 60)
    ),
    PerturbationSpec(
        "gaussian_noise",
        "Gaussian noise",
        "image_classification",
        (0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80),
    ),
    PerturbationSpec(
        "blur", "Gaussian blur", "image_classification", (0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0)
    ),
    PerturbationSpec(
        "occlusion",
        "Center occlusion",
        "image_classification",
        (0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60),
    ),
    PerturbationSpec(
        "brightness",
        "Brightness reduction",
        "image_classification",
        (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2),
    ),
):
    PERTURBATION_REGISTRY.register(perturbation)


def ensure_compatible(model: ModelSpec, dataset: DatasetSpec) -> None:
    if model.task != dataset.task:
        raise ValueError(
            f"Task mismatch: model '{model.id}' uses {model.task}, "
            f"dataset '{dataset.id}' uses {dataset.task}."
        )
    if dataset.id not in model.datasets or model.id not in dataset.models:
        raise ValueError(f"Model '{model.id}' is not compatible with dataset '{dataset.id}'.")
    if model.input_channels != dataset.input_channels or model.input_size != dataset.input_size:
        raise ValueError(
            f"Input mismatch between model '{model.id}' and dataset '{dataset.id}'."
        )


def validate_selection(
    model_id: str,
    dataset_id: str,
    perturbation_ids: tuple[str, ...] | None = None,
    *,
    require_implemented: bool = True,
) -> tuple[ModelSpec, DatasetSpec, tuple[PerturbationSpec, ...]]:
    model = MODEL_REGISTRY.get(model_id)
    dataset = DATASET_REGISTRY.get(dataset_id)
    ensure_compatible(model, dataset)
    selected_ids = perturbation_ids or dataset.perturbations
    perturbations = tuple(PERTURBATION_REGISTRY.get(item_id) for item_id in selected_ids)
    unsupported = [spec.id for spec in perturbations if spec.id not in dataset.perturbations]
    if unsupported:
        raise ValueError(f"Dataset '{dataset.id}' does not support: {', '.join(unsupported)}")
    if require_implemented:
        pending = [
            label
            for label, implemented in (
                (f"model:{model.id}", model.implemented),
                (f"dataset:{dataset.id}", dataset.implemented),
                *((f"perturbation:{spec.id}", spec.implemented) for spec in perturbations),
            )
            if not implemented
        ]
        if pending:
            raise NotImplementedError(f"Planned but not implemented: {', '.join(pending)}")
    return model, dataset, perturbations
