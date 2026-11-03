import json

from failure_atlas.agent import generate_agent_report, steps_for


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
    assert [step.key for step in steps_for("analyze")] == ["data", "baseline", "scan"]
    assert [step.key for step in steps_for("full")][-2:] == ["repair", "compare"]


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
