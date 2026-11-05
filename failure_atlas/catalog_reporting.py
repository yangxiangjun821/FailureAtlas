"""Generate a deterministic model/dataset compatibility report."""

from __future__ import annotations

from pathlib import Path

from .adapters import get_dataset_adapter, get_model_adapter
from .catalog import DATASET_REGISTRY, MODEL_REGISTRY, ensure_compatible


def compatibility_markdown() -> str:
    models = MODEL_REGISTRY.values()
    datasets = DATASET_REGISTRY.values()
    lines = [
        "# FailureAtlas Model Compatibility Matrix",
        "",
        "Generated from the model and dataset registries. A check means the input shape, "
        "task, adapter, and preprocessing contract are compatible.",
        "",
        "| Model | " + " | ".join(dataset.display_name for dataset in datasets) + " |",
        "|---|" + "---:|" * len(datasets),
    ]
    for model in models:
        cells = []
        for dataset in datasets:
            try:
                ensure_compatible(model, dataset)
                cells.append("✓")
            except ValueError:
                cells.append("—")
        lines.append(f"| {model.display_name} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Model baselines",
            "",
            "| ID | Architecture | Input | Parameters | Compatible datasets |",
            "|---|---|---:|---:|---|",
        ]
    )
    for model in models:
        built = get_model_adapter(model.id).build({"initialize_weights": False})
        parameter_count = sum(parameter.numel() for parameter in built.parameters())
        lines.append(
            f"| `{model.id}` | {model.architecture} | "
            f"{model.input_channels}×{model.input_size[0]}×{model.input_size[1]} | "
            f"{parameter_count:,} | {', '.join(model.datasets)} |"
        )

    lines.extend(
        [
            "",
            "## Dataset preprocessing",
            "",
            "| ID | Train / test | Input | Mean | Std | Perturbations |",
            "|---|---:|---:|---|---|---|",
        ]
    )
    for dataset in datasets:
        metadata = get_dataset_adapter(dataset.id).metadata()
        lines.append(
            f"| `{dataset.id}` | {metadata['train_samples']:,} / "
            f"{metadata['test_samples']:,} | {dataset.input_channels}×"
            f"{dataset.input_size[0]}×{dataset.input_size[1]} | "
            f"{list(dataset.normalization_mean)} | {list(dataset.normalization_std)} | "
            f"{', '.join(dataset.perturbations)} |"
        )

    lines.extend(["", "## Baseline commands", "", "```powershell"])
    for model in models:
        lines.append(
            "python -m scripts.train_baseline "
            f"--model {model.id} --dataset {model.datasets[0]}"
        )
    lines.extend(
        [
            "```",
            "",
            "Weights and generated metrics remain local under `artifacts/`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_compatibility_report(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(compatibility_markdown(), encoding="utf-8")
    return output
