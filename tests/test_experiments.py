import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from failure_atlas.experiments import (
    archive_run_artifacts,
    create_run_workspace,
    finalize_run,
    new_run_id,
    validate_run_artifacts,
)


def test_run_id_and_workspace_are_isolated(tmp_path: Path) -> None:
    run_id = new_run_id(datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc))
    assert run_id.startswith("20261103T120000Z-")

    workspace = create_run_workspace(
        tmp_path,
        model_id="fashion_cnn",
        dataset_id="fashion_mnist",
        stage="analyze",
        run_id="baseline-study-01",
    )
    config = yaml.safe_load((workspace.path / "config.yaml").read_text(encoding="utf-8"))
    assert config["model"] == "fashion_cnn"
    assert config["dataset"] == "fashion_mnist"
    assert (workspace.path / "plots").is_dir()

    with pytest.raises(FileExistsError):
        create_run_workspace(
            tmp_path,
            model_id="fashion_cnn",
            dataset_id="fashion_mnist",
            stage="analyze",
            run_id="baseline-study-01",
        )
    with pytest.raises(ValueError, match="run_id"):
        create_run_workspace(
            tmp_path,
            model_id="fashion_cnn",
            dataset_id="fashion_mnist",
            stage="analyze",
            run_id="../escape",
        )


def test_artifact_archive_and_integrity_validation(tmp_path: Path) -> None:
    workspace = create_run_workspace(
        tmp_path / "runs",
        model_id="fashion_cnn",
        dataset_id="fashion_mnist",
        stage="full",
        run_id="repair-study",
    )
    source = tmp_path / "source.json"
    source.write_text('{"accuracy": 0.9}', encoding="utf-8")
    archived = archive_run_artifacts(workspace, {"baseline_metrics.json": source})
    assert archived == ["baseline_metrics.json"]
    validate_run_artifacts(workspace, ("config.yaml", "baseline_metrics.json"))
    with pytest.raises(RuntimeError, match="missing required artifacts"):
        validate_run_artifacts(workspace, ("failure_analysis.json",))

    finalize_run(workspace, status="completed", artifacts=archived)
    metadata = json.loads((workspace.path / "metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((workspace.path / "agent_run.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert manifest["run_id"] == "repair-study"


def test_pretrained_workspace_records_official_weight_contract(tmp_path: Path) -> None:
    workspace = create_run_workspace(
        tmp_path,
        model_id="mobilenet_v3",
        dataset_id="cifar10",
        stage="baseline",
        weight_mode="pretrained",
        run_id="pretrained-study",
    )
    metadata = json.loads((workspace.path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["weight_mode"] == "pretrained"
    assert metadata["pretrained_weights"]["source"] == "torchvision"
    assert metadata["pretrained_weights"]["input_size"] == [224, 224]
