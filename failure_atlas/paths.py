"""Derive checkpoint and report locations from a model/dataset selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .catalog import validate_selection


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts"
REPORT_ROOT = ROOT / "reports" / "generated"

# The original FashionCNN / Fashion-MNIST study keeps its flat, pre-registry layout
# so existing artifacts, the dashboard and exported bundles keep working.
LEGACY_SELECTION = ("fashion_cnn", "fashion_mnist")


@dataclass(frozen=True)
class ExperimentPaths:
    model_id: str
    dataset_id: str
    artifact_dir: Path
    report_dir: Path
    baseline_checkpoint: Path
    baseline_metrics: Path
    repaired_checkpoint: Path
    repair_plan: Path

    @property
    def failure_analysis(self) -> Path:
        return self.report_dir / "failure_analysis.json"

    @property
    def failure_report(self) -> Path:
        return self.report_dir / "failure_report.md"

    @property
    def failure_curves(self) -> Path:
        return self.report_dir / "failure_curves.png"

    @property
    def repair_comparison(self) -> Path:
        return self.report_dir / "repair_comparison.json"

    @property
    def repair_comparison_plot(self) -> Path:
        return self.report_dir / "repair_comparison.png"

    @property
    def repair_report(self) -> Path:
        return self.report_dir / "repair_report.md"

    @property
    def agent_report(self) -> Path:
        return self.report_dir / "agent_report.md"


def experiment_paths(
    model_id: str,
    dataset_id: str,
    *,
    artifact_root: Path = ARTIFACT_ROOT,
    report_root: Path = REPORT_ROOT,
) -> ExperimentPaths:
    """Validate the selection and return its isolated artifact/report locations."""
    validate_selection(model_id, dataset_id)
    if (model_id, dataset_id) == LEGACY_SELECTION:
        return ExperimentPaths(
            model_id,
            dataset_id,
            artifact_dir=artifact_root,
            report_dir=report_root,
            baseline_checkpoint=artifact_root / "baseline_fashion_cnn.pt",
            baseline_metrics=artifact_root / "baseline_metrics.json",
            repaired_checkpoint=artifact_root / "repaired_fashion_cnn.pt",
            repair_plan=artifact_root / "repair_plan.json",
        )
    name = f"{model_id}_{dataset_id}"
    artifact_dir = artifact_root / name
    return ExperimentPaths(
        model_id,
        dataset_id,
        artifact_dir=artifact_dir,
        report_dir=report_root / name,
        baseline_checkpoint=artifact_dir / "baseline.pt",
        baseline_metrics=artifact_dir / "baseline_metrics.json",
        repaired_checkpoint=artifact_dir / "repaired.pt",
        repair_plan=artifact_dir / "repair_plan.json",
    )
