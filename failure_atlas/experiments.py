"""Create isolated, reproducible experiment workspaces."""

from __future__ import annotations

import json
import platform
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch
import yaml

from .catalog import validate_selection


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


@dataclass(frozen=True)
class RunWorkspace:
    run_id: str
    path: Path
    model_id: str
    dataset_id: str
    stage: str


def new_run_id(now: datetime | None = None) -> str:
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


def create_run_workspace(
    runs_root: str | Path,
    *,
    model_id: str,
    dataset_id: str,
    perturbation_ids: tuple[str, ...] | None = None,
    stage: str,
    seed: int = 42,
    run_id: str | None = None,
) -> RunWorkspace:
    model, dataset, perturbations = validate_selection(
        model_id, dataset_id, perturbation_ids, require_implemented=True
    )
    selected_id = run_id or new_run_id()
    if not RUN_ID_PATTERN.fullmatch(selected_id):
        raise ValueError(
            "run_id must start with a letter or number and contain only letters, "
            "numbers, '.', '_' or '-'."
        )
    root = Path(runs_root).resolve()
    workspace_path = (root / selected_id).resolve()
    if workspace_path.parent != root:
        raise ValueError("run_id resolved outside the runs directory.")
    workspace_path.mkdir(parents=True, exist_ok=False)
    (workspace_path / "plots").mkdir()

    config = {
        "run_id": selected_id,
        "model": model.id,
        "dataset": dataset.id,
        "perturbations": [spec.id for spec in perturbations],
        "stage": stage,
        "seed": seed,
    }
    (workspace_path / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    metadata = {
        **config,
        "status": "created",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "executable": sys.executable,
    }
    _write_json(workspace_path / "metadata.json", metadata)
    return RunWorkspace(selected_id, workspace_path, model.id, dataset.id, stage)


def archive_run_artifacts(workspace: RunWorkspace, sources: dict[str, Path]) -> list[str]:
    archived: list[str] = []
    for relative_name, source in sources.items():
        if not source.exists():
            continue
        destination = (workspace.path / relative_name).resolve()
        if workspace.path.resolve() not in destination.parents:
            raise ValueError(f"Artifact target escapes run directory: {relative_name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)
        archived.append(relative_name)
    return archived


def finalize_run(
    workspace: RunWorkspace,
    *,
    status: str,
    artifacts: list[str] | None = None,
    error: str | None = None,
) -> None:
    metadata_path = workspace.path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "status": status,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": artifacts or [],
        }
    )
    if error:
        metadata["error"] = error
    _write_json(metadata_path, metadata)
    _write_json(
        workspace.path / "agent_run.json",
        {
            "run_id": workspace.run_id,
            "status": status,
            "model": workspace.model_id,
            "dataset": workspace.dataset_id,
            "stage": workspace.stage,
            "artifacts": artifacts or [],
        },
    )


def validate_run_artifacts(workspace: RunWorkspace, required: tuple[str, ...]) -> None:
    missing = [name for name in required if not (workspace.path / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Run '{workspace.run_id}' is missing required artifacts: {', '.join(missing)}"
        )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
