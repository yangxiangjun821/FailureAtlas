import json

from failure_atlas.agent import (
    generate_agent_report,
    generate_baseline_report,
    command_for,
    run_pipeline,
    steps_for,
)


def sample_analysis() -> dict:
    result = {"severity": 0.0, "accuracy": 0.9}
    return {
        "model": "TinyCNN",
        "dataset": "Synthetic test split",
        "sample_count": 10,
        "baseline": {
            "accuracy": 0.9,
            "macro_f1": 0.89,
            "ece": 0.02,
            "high_confidence_errors": 0,
        },
        "experiments": {
            "rotation": {
                "boundaries": {"mild": 5.0, "obvious": 10.0, "severe": 20.0},
                "results": [result, {"severity": 20.0, "accuracy": 0.6}],
            }
        },
    }


def test_pipeline_stages_are_ordered() -> None:
    assert [step.key for step in steps_for("baseline")] == ["data", "baseline"]
    assert [step.key for step in steps_for("analyze")] == ["data", "baseline", "scan"]
    assert [step.key for step in steps_for("full")][-2:] == ["repair", "compare"]


def test_pipeline_command_carries_reproducibility_controls() -> None:
    scan = steps_for("analyze", "lenet", "mnist")[-1]
    command = command_for(
        scan,
        model_id="lenet",
        dataset_id="mnist",
        seed=7,
        batch_size=64,
        perturbations=("rotation", "blur"),
        severity_scale=0.5,
    )
    assert command[command.index("--seed") + 1] == "7"
    assert command[command.index("--batch-size") + 1] == "64"
    assert command.count("--perturbation") == 2
    assert command[command.index("--severity-scale") + 1] == "0.5"


def test_dry_run_does_not_create_workspace(tmp_path) -> None:
    assert run_pipeline("analyze", dry_run=True, runs_root=tmp_path) is None
    assert list(tmp_path.iterdir()) == []


def test_agent_generates_standalone_markdown_report(tmp_path) -> None:
    analysis = tmp_path / "analysis.json"
    report = tmp_path / "agent_report.md"
    analysis.write_text(json.dumps(sample_analysis()), encoding="utf-8")
    generated = generate_agent_report(
        analysis_path=analysis,
        comparison_path=tmp_path / "missing-comparison.json",
        output_path=report,
    )
    content = generated.read_text(encoding="utf-8")
    assert "FailureAtlas Local Agent Report" in content
    assert "Agent diagnosis and proposed repair" in content
    assert "Rotation" in content
    assert report.with_name("agent_run.json").exists()


def test_agent_generates_baseline_only_report(tmp_path) -> None:
    metrics = tmp_path / "baseline_metrics.json"
    report = tmp_path / "agent_report.md"
    metrics.write_text(
        json.dumps(
            {
                "model": {"display_name": "LeNet", "parameter_count": 61706},
                "dataset": {"display_name": "MNIST"},
                "split": {"seed": 42},
                "test": {"accuracy": 0.98, "macro_f1": 0.979},
            }
        ),
        encoding="utf-8",
    )
    generated = generate_baseline_report(metrics, report)
    content = generated.read_text(encoding="utf-8")
    assert "# FailureAtlas" in content
    assert "Accuracy：0.9800" in content
