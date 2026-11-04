import pytest

from failure_atlas.agent import steps_for
from failure_atlas.paths import ARTIFACT_ROOT, REPORT_ROOT, experiment_paths


def test_default_selection_keeps_legacy_layout() -> None:
    paths = experiment_paths("fashion_cnn", "fashion_mnist")
    assert paths.baseline_checkpoint == ARTIFACT_ROOT / "baseline_fashion_cnn.pt"
    assert paths.repaired_checkpoint == ARTIFACT_ROOT / "repaired_fashion_cnn.pt"
    assert paths.repair_plan == ARTIFACT_ROOT / "repair_plan.json"
    assert paths.failure_analysis == REPORT_ROOT / "failure_analysis.json"
    assert paths.repair_comparison == REPORT_ROOT / "repair_comparison.json"


def test_other_selections_are_isolated() -> None:
    lenet = experiment_paths("lenet", "mnist")
    cnn_mnist = experiment_paths("fashion_cnn", "mnist")
    assert lenet.artifact_dir == ARTIFACT_ROOT / "lenet_mnist"
    assert lenet.report_dir == REPORT_ROOT / "lenet_mnist"
    legacy = experiment_paths("fashion_cnn", "fashion_mnist")
    outputs = lambda p: {
        p.baseline_checkpoint, p.repaired_checkpoint, p.repair_plan,
        p.failure_analysis, p.repair_comparison, p.agent_report,
    }
    assert outputs(lenet).isdisjoint(outputs(legacy))
    assert outputs(lenet).isdisjoint(outputs(cnn_mnist))
    assert outputs(cnn_mnist).isdisjoint(outputs(legacy))


def test_invalid_selection_is_rejected() -> None:
    with pytest.raises(ValueError):
        experiment_paths("resnet18", "mnist")


def test_pipeline_outputs_follow_selection() -> None:
    default = {p for step in steps_for("full") for p in step.outputs}
    lenet = {p for step in steps_for("full", "lenet", "mnist") for p in step.outputs}
    assert default.isdisjoint(lenet)
    assert any("lenet_mnist" in str(path) for path in lenet)


def test_analyze_archive_excludes_repair_evidence() -> None:
    from pathlib import Path

    from failure_atlas.agent import archive_sources

    paths = experiment_paths("lenet", "mnist")
    analyze = archive_sources(paths, Path("report.md"), "analyze")
    full = archive_sources(paths, Path("report.md"), "full")
    repair_files = {"repair_plan.json", "repair_comparison.json", "plots/repair_comparison.png"}
    assert repair_files.isdisjoint(analyze)
    assert repair_files <= set(full)
