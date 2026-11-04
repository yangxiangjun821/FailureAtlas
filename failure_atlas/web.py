"""Local FastAPI dashboard for FailureAtlas experiments."""

from __future__ import annotations

import asyncio
import json
import os
import platform
import re
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .adapters import get_dataset_adapter, get_model_adapter
from .catalog import (
    DATASET_REGISTRY,
    MODEL_REGISTRY,
    PERTURBATION_REGISTRY,
    ensure_compatible,
    validate_selection,
)
from .paths import experiment_paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "web"
REPORT_ROOT = PROJECT_ROOT / "reports" / "generated"
COMPARISON_PATH = REPORT_ROOT / "repair_comparison.json"
PLAN_PATH = PROJECT_ROOT / "artifacts" / "repair_plan.json"
WORKSPACE_ROOT = PROJECT_ROOT / "workspaces"
RUN_ROOT = PROJECT_ROOT / "runs"
DEPLOY_SNAPSHOT_PATH = PROJECT_ROOT / "deploy" / "dashboard.json"
DEMO_MODE = os.getenv("FAILUREATLAS_DEMO", "0") == "1"

app = FastAPI(title="FailureAtlas", version="0.2.0")
app.mount("/static", StaticFiles(directory=WEB_ROOT / "static"), name="static")
REPORT_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/reports", StaticFiles(directory=REPORT_ROOT), name="reports")

job_state: dict = {
    "status": "idle",
    "action": None,
    "started_at": None,
    "finished_at": None,
    "return_code": None,
    "logs": [],
}
job_task: asyncio.Task | None = None


class ExperimentRequest(BaseModel):
    model: str
    dataset: str
    stage: Literal["baseline", "analyze", "full"] = "analyze"
    run_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
    epochs: int | None = Field(default=None, ge=1, le=100)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    batch_size: int | None = Field(default=None, ge=1, le=4096)
    perturbations: list[str] | None = Field(default=None, min_length=1, max_length=5)
    severity_scale: float = Field(default=1.0, ge=0.25, le=2.0)
    reuse_artifacts: bool = True


class ComparisonRequest(BaseModel):
    run_ids: list[str] = Field(min_length=2, max_length=6)


def safe_run_directory(run_id: str) -> Path:
    """Resolve a run without allowing paths to escape the local run archive."""
    candidate = (RUN_ROOT / run_id).resolve()
    try:
        candidate.relative_to(RUN_ROOT.resolve())
    except ValueError as error:
        raise HTTPException(status_code=404, detail="实验记录不存在。") from error
    if not candidate.is_dir():
        raise HTTPException(status_code=404, detail=f"实验 '{run_id}' 不存在。")
    return candidate


def read_optional_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_json(path: Path) -> dict:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Required artifact is missing: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_dashboard_payload() -> dict:
    if DEMO_MODE:
        if not DEPLOY_SNAPSHOT_PATH.exists():
            raise HTTPException(status_code=503, detail="Deployment snapshot is missing.")
        return read_json(DEPLOY_SNAPSHOT_PATH)
    comparison = read_json(COMPARISON_PATH)
    plan = read_json(PLAN_PATH)
    before = comparison["baseline"]
    after = comparison["repaired"]
    experiments = []
    for name, baseline_experiment in before["experiments"].items():
        repaired_experiment = after["experiments"][name]
        experiments.append(
            {
                "name": name,
                "baseline": baseline_experiment["results"],
                "repaired": repaired_experiment["results"],
                "boundaries_before": baseline_experiment["boundaries"],
                "boundaries_after": repaired_experiment["boundaries"],
            }
        )
    dataset_id = after.get("dataset_id", "fashion_mnist")
    model_id = after.get("model_id", "fashion_cnn")
    model = get_model_adapter(model_id).build()
    baseline_metrics = read_optional_json(
        experiment_paths(model_id, dataset_id).baseline_metrics
    ) or {}
    return {
        "deployment_mode": "local",
        "model": after.get("model", "FashionCNN"),
        "dataset": DATASET_REGISTRY.get(dataset_id).display_name,
        "sample_count": after["sample_count"],
        "clean_accuracy_before": before["baseline"]["accuracy"],
        "clean_accuracy_after": after["baseline"]["accuracy"],
        "clean_ece_before": before["baseline"]["ece"],
        "clean_ece_after": after["baseline"]["ece"],
        "clean_macro_f1_after": after["baseline"].get("macro_f1"),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "inference_ms_per_sample": baseline_metrics.get("test", {}).get(
            "inference_ms_per_sample"
        ),
        "actions": plan["actions"],
        "ranking": plan["ranking"],
        "experiments": experiments,
    }


def dataset_cached(dataset_id: str) -> bool:
    markers = {
        "fashion_mnist": PROJECT_ROOT / "data/FashionMNIST/raw/train-images-idx3-ubyte",
        "mnist": PROJECT_ROOT / "data/MNIST/raw/train-images-idx3-ubyte",
        "cifar10": PROJECT_ROOT / "data/cifar-10-batches-py/data_batch_1",
    }
    marker = markers.get(dataset_id)
    return bool(marker and marker.exists())


def build_catalog_payload() -> dict:
    run_records = build_runs_payload()["runs"]
    datasets = []
    for spec in DATASET_REGISTRY.values():
        metadata = get_dataset_adapter(spec.id).metadata()
        datasets.append(
            {
                **metadata,
                "models": list(spec.models),
                "cached": dataset_cached(spec.id),
                "experiment_count": sum(run["dataset"] == spec.id for run in run_records),
                "perturbations": [item.id for item in PERTURBATION_REGISTRY.values()],
            }
        )
    models = []
    for spec in MODEL_REGISTRY.values():
        model = get_model_adapter(spec.id).build()
        weights_ready = any(
            experiment_paths(spec.id, dataset_id).baseline_checkpoint.is_file()
            for dataset_id in spec.datasets
        )
        models.append(
            {
                "id": spec.id,
                "name": spec.display_name,
                "architecture": spec.architecture,
                "input_shape": [spec.input_channels, *spec.input_size],
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "datasets": list(spec.datasets),
                "weights_ready": weights_ready,
                "experiment_count": sum(run["model"] == spec.id for run in run_records),
            }
        )
    compatibility = []
    for model in MODEL_REGISTRY.values():
        for dataset in DATASET_REGISTRY.values():
            try:
                ensure_compatible(model, dataset)
                compatible = True
            except ValueError:
                compatible = False
            compatibility.append(
                {"model": model.id, "dataset": dataset.id, "compatible": compatible}
            )
    return {
        "models": models,
        "datasets": datasets,
        "perturbations": PERTURBATION_REGISTRY.as_dicts(),
        "compatibility": compatibility,
    }


def run_metric_summary(directory: Path) -> dict:
    baseline_metrics = read_optional_json(directory / "baseline_metrics.json") or {}
    analysis = read_optional_json(directory / "failure_analysis.json") or {}
    comparison = read_optional_json(directory / "repair_comparison.json") or {}
    baseline_test = baseline_metrics.get("test", {})
    analysis_baseline = analysis.get("baseline", {})
    repaired_baseline = comparison.get("repaired", {}).get("baseline", {})
    model = baseline_metrics.get("model", {})
    accuracy = analysis_baseline.get("accuracy", baseline_test.get("accuracy"))
    repaired_accuracy = repaired_baseline.get("accuracy")
    return {
        "accuracy": accuracy,
        "macro_f1": analysis_baseline.get("macro_f1", baseline_test.get("macro_f1")),
        "ece": analysis_baseline.get("ece"),
        "parameters": model.get("parameter_count"),
        "inference_ms_per_sample": baseline_test.get("inference_ms_per_sample"),
        "repaired_accuracy": repaired_accuracy,
        "clean_delta": (
            repaired_accuracy - accuracy
            if isinstance(repaired_accuracy, (int, float)) and isinstance(accuracy, (int, float))
            else None
        ),
    }


def build_run_record(directory: Path) -> dict | None:
    metadata = read_optional_json(directory / "metadata.json")
    if metadata is None:
        return None
    artifacts = metadata.get("artifacts", [])
    return {
        "run_id": metadata.get("run_id", directory.name),
        "model": metadata.get("model", "unknown"),
        "dataset": metadata.get("dataset", "unknown"),
        "stage": metadata.get("stage", "unknown"),
        "status": metadata.get("status", "unknown"),
        "created_at": metadata.get("created_at"),
        "completed_at": metadata.get("completed_at"),
        "error": metadata.get("error"),
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "report_available": (directory / "agent_report.md").is_file(),
        "metrics": run_metric_summary(directory),
    }


def build_runs_payload() -> dict:
    runs = []
    if RUN_ROOT.exists():
        for directory in RUN_ROOT.iterdir():
            if not directory.is_dir():
                continue
            record = build_run_record(directory)
            if record is not None:
                runs.append(record)
    runs.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return {"runs": runs}


def build_run_detail(run_id: str) -> dict:
    directory = safe_run_directory(run_id)
    record = build_run_record(directory)
    if record is None:
        raise HTTPException(status_code=404, detail="实验元数据不存在或无法读取。")
    analysis = read_optional_json(directory / "failure_analysis.json") or {}
    repair_plan = read_optional_json(directory / "repair_plan.json") or {}
    comparison = read_optional_json(directory / "repair_comparison.json") or {}
    experiments = []
    for name, result in analysis.get("experiments", {}).items():
        repaired = comparison.get("repaired", {}).get("experiments", {}).get(name, {})
        experiments.append(
            {
                "name": name,
                "boundaries_before": result.get("boundaries", {}),
                "boundaries_after": repaired.get("boundaries"),
                "endpoint_before": (result.get("results") or [{}])[-1].get("accuracy"),
                "endpoint_after": (repaired.get("results") or [{}])[-1].get("accuracy"),
                "results_before": result.get("results", []),
                "results_after": repaired.get("results", []),
            }
        )
    artifacts = []
    for relative in record["artifacts"]:
        path = directory / relative
        if path.is_file():
            artifacts.append(
                {
                    "name": relative,
                    "size": path.stat().st_size,
                    "url": f"/api/runs/{run_id}/artifacts/{relative}",
                }
            )
    return {
        **record,
        "metadata": read_optional_json(directory / "metadata.json"),
        "config": (
            yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
            if (directory / "config.yaml").is_file()
            else {}
        ),
        "baseline": analysis.get("baseline", {}),
        "experiments": experiments,
        "repair_actions": repair_plan.get("actions", []),
        "ranking": repair_plan.get("ranking", []),
        "artifacts": artifacts,
    }


def build_comparison_payload(run_ids: list[str]) -> dict:
    if len(set(run_ids)) != len(run_ids):
        raise HTTPException(status_code=422, detail="请选择不同的实验进行对比。")
    details = [build_run_detail(run_id) for run_id in run_ids]
    datasets = {detail["dataset"] for detail in details}
    comparable = len(datasets) == 1
    return {
        "dataset": details[0]["dataset"] if comparable else "多个数据集",
        "comparable": comparable,
        "limitation": None
        if comparable
        else "所选实验来自不同数据集，仅并列展示原始指标，不进行高低排名。",
        "runs": [
            {
                "run_id": detail["run_id"],
                "model": detail["model"],
                "stage": detail["stage"],
                "metrics": detail["metrics"],
                "experiments": detail["experiments"],
            }
            for detail in details
        ],
    }


def build_system_payload() -> dict:
    disk = shutil.disk_usage(PROJECT_ROOT)
    try:
        import torch

        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else None
        torch_version = torch.__version__
    except ImportError:
        cuda, gpu, torch_version = False, None, None
    return {
        "python": platform.python_version(),
        "torch": torch_version,
        "cuda": cuda,
        "gpu": gpu,
        "disk_free_bytes": disk.free,
        "mode": "showcase" if DEMO_MODE else "local",
    }


def build_preflight_payload(request: ExperimentRequest) -> dict:
    perturbations = tuple(request.perturbations) if request.perturbations else None
    try:
        validate_selection(request.model, request.dataset, perturbations)
    except (KeyError, ValueError, NotImplementedError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    from .agent import steps_for

    steps = []
    for step in steps_for(request.stage, request.model, request.dataset):
        will_run = not request.reuse_artifacts or not step.ready
        steps.append(
            {
                "key": step.key,
                "title": step.title,
                "ready": step.ready,
                "will_run": will_run,
            }
        )
    base_minutes = {"fashion_cnn": 3, "lenet": 2, "resnet18": 12}.get(request.model, 5)
    stage_factor = {"baseline": 1.0, "analyze": 1.8, "full": 3.0}[request.stage]
    selected_count = len(request.perturbations or PERTURBATION_REGISTRY.as_dicts())
    estimate = base_minutes * stage_factor * max(0.35, selected_count / 5)
    if request.reuse_artifacts:
        runnable = sum(step["will_run"] for step in steps)
        estimate *= max(0.15, runnable / max(1, len(steps)))
    paths = experiment_paths(request.model, request.dataset)
    return {
        "compatible": True,
        "dataset_cached": dataset_cached(request.dataset),
        "weights_ready": paths.baseline_checkpoint.is_file(),
        "estimated_minutes": max(1, round(estimate)),
        "steps": steps,
        "selected_perturbations": request.perturbations
        or [item.id for item in PERTURBATION_REGISTRY.values()],
        "reuse_artifacts": request.reuse_artifacts,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/dashboard/overview")
@app.get("/dashboard/new")
@app.get("/dashboard/runs")
@app.get("/dashboard/catalog")
@app.get("/dashboard/comparison")
@app.get("/dashboard/reports")
@app.get("/dashboard/system")
def dashboard_page() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/runs/{run_id}")
def run_page(run_id: str) -> FileResponse:
    safe_run_directory(run_id)
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/api/dashboard")
def dashboard() -> dict:
    return build_dashboard_payload()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "mode": "showcase" if DEMO_MODE else "local"}


@app.get("/api/catalog")
def catalog() -> dict:
    return build_catalog_payload()


@app.get("/api/runs")
def runs() -> dict:
    return build_runs_payload()


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str) -> dict:
    return build_run_detail(run_id)


@app.get("/api/runs/{run_id}/artifacts/{artifact_path:path}")
def run_artifact(run_id: str, artifact_path: str, download: bool = False) -> FileResponse:
    directory = safe_run_directory(run_id)
    target = (directory / artifact_path).resolve()
    try:
        target.relative_to(directory.resolve())
    except ValueError as error:
        raise HTTPException(status_code=404, detail="实验产物不存在。") from error
    if not target.is_file():
        raise HTTPException(status_code=404, detail="实验产物不存在。")
    media_type = "text/markdown" if target.suffix.lower() == ".md" else None
    return FileResponse(
        target,
        media_type=media_type,
        filename=target.name if download else None,
    )


@app.post("/api/comparison")
def comparison(request: ComparisonRequest) -> dict:
    return build_comparison_payload(request.run_ids)


@app.post("/api/preflight")
def preflight(request: ExperimentRequest) -> dict:
    return build_preflight_payload(request)


@app.get("/api/system")
def system() -> dict:
    return build_system_payload()


@app.get("/api/jobs/current")
def current_job() -> dict:
    return job_state


@app.get("/api/projects")
def projects() -> dict:
    imported = []
    if not DEMO_MODE:
        from .bundle import list_imported_projects

        imported = list_imported_projects(WORKSPACE_ROOT)
    payload = build_dashboard_payload()
    built_in = {
        "id": "fashion-mnist-built-in",
        "name": f"{payload['dataset']} repaired {payload['model']}",
        "task": "image_classification",
        "sample_count": payload["sample_count"],
        "class_count": 10,
        "input_shape": [-1, 1, 28, 28],
        "clean_accuracy": payload["clean_accuracy_after"],
        "status": "analyzed",
        "built_in": True,
    }
    return {"projects": [built_in, *imported]}


@app.post("/api/projects/import", status_code=201)
async def import_project(request: Request) -> dict:
    if DEMO_MODE:
        raise HTTPException(status_code=403, detail="在线演示为只读模式，请在本地版导入模型包。")
    from .bundle import MAX_BUNDLE_BYTES, import_bundle_bytes

    content_length = request.headers.get("content-length")
    if content_length:
        try:
            declared = int(content_length)
        except ValueError as error:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header.") from error
        if declared > MAX_BUNDLE_BYTES:
            raise HTTPException(status_code=413, detail="模型包超过 100 MB 限制。")
    data = await request.body()
    try:
        return await asyncio.to_thread(import_bundle_bytes, data, WORKSPACE_ROOT)
    except (ValueError, json.JSONDecodeError, zipfile.BadZipFile) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


async def run_commands(action: str, modules: list[str]) -> None:
    job_state.update(
        status="running",
        action=action,
        started_at=datetime.now(timezone.utc).isoformat(),
        finished_at=None,
        return_code=None,
        logs=[],
    )
    for module in modules:
        job_state["logs"].append(f"$ python -m {module}")
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            "-m",
            module,
            cwd=PROJECT_ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert process.stdout is not None
        async for line in process.stdout:
            job_state["logs"].append(line.decode(errors="replace").rstrip())
            job_state["logs"] = job_state["logs"][-120:]
        return_code = await process.wait()
        if return_code != 0:
            job_state.update(status="error", return_code=return_code)
            break
    else:
        job_state.update(status="success", return_code=0)
    job_state["finished_at"] = datetime.now(timezone.utc).isoformat()


async def run_agent_experiment(request: ExperimentRequest) -> None:
    command = [
        sys.executable,
        "-u",
        "-m",
        "failure_atlas.agent",
        "run",
        "--stage",
        request.stage,
        "--model",
        request.model,
        "--dataset",
        request.dataset,
    ]
    if request.run_id:
        command.extend(["--run-id", request.run_id])
    if request.epochs:
        command.extend(["--baseline-epochs", str(request.epochs)])
    command.extend(["--seed", str(request.seed)])
    if request.batch_size:
        command.extend(["--batch-size", str(request.batch_size)])
    command.extend(["--severity-scale", str(request.severity_scale)])
    for perturbation in request.perturbations or []:
        command.extend(["--perturbation", perturbation])
    if not request.reuse_artifacts:
        command.append("--force")
    job_state.update(
        status="running",
        action="experiment",
        started_at=datetime.now(timezone.utc).isoformat(),
        finished_at=None,
        return_code=None,
        logs=["$ " + " ".join(command[1:])],
        selection=request.model_dump(exclude={"run_id"}),
    )
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=PROJECT_ROOT,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert process.stdout is not None
    async for line in process.stdout:
        job_state["logs"].append(line.decode(errors="replace").rstrip())
        job_state["logs"] = job_state["logs"][-120:]
    return_code = await process.wait()
    resolved_run_id = request.run_id
    if resolved_run_id is None:
        for line in reversed(job_state["logs"]):
            match = re.search(r"(?:run_id=|workspace=.*[\\/])([A-Za-z0-9][A-Za-z0-9._-]{0,79})$", line)
            if match:
                resolved_run_id = match.group(1)
                break
    if resolved_run_id and (RUN_ROOT / resolved_run_id).is_dir():
        run_directory = RUN_ROOT / resolved_run_id
        log_path = run_directory / "execution.log"
        log_path.write_text("\n".join(job_state["logs"]) + "\n", encoding="utf-8")
        metadata_path = run_directory / "metadata.json"
        metadata = read_optional_json(metadata_path)
        if metadata is not None:
            artifacts = metadata.setdefault("artifacts", [])
            if "execution.log" not in artifacts:
                artifacts.append("execution.log")
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    job_state.update(
        status="success" if return_code == 0 else "error",
        return_code=return_code,
        finished_at=datetime.now(timezone.utc).isoformat(),
        run_id=resolved_run_id,
    )


@app.post("/api/jobs/{action}", status_code=202)
async def start_job(action: Literal["rescan", "repair"]) -> dict:
    global job_task
    if DEMO_MODE:
        raise HTTPException(status_code=403, detail="在线演示不运行训练任务，请使用本地 GPU 版本。")
    if job_task is not None and not job_task.done():
        raise HTTPException(status_code=409, detail="Another experiment is already running.")
    modules = {
        "rescan": ["scripts.search_failures", "scripts.compare_repair"],
        "repair": ["scripts.repair_model", "scripts.compare_repair"],
    }[action]
    job_task = asyncio.create_task(run_commands(action, modules))
    return {"accepted": True, "action": action}


@app.post("/api/jobs/experiment", status_code=202)
async def start_experiment(request: ExperimentRequest) -> dict:
    global job_task
    if DEMO_MODE:
        raise HTTPException(status_code=403, detail="在线演示为只读模式，请使用本地版本运行实验。")
    if job_task is not None and not job_task.done():
        raise HTTPException(status_code=409, detail="已有实验正在运行，请等待完成。")
    try:
        validate_selection(
            request.model,
            request.dataset,
            tuple(request.perturbations) if request.perturbations else None,
        )
    except (KeyError, ValueError, NotImplementedError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if request.run_id and (RUN_ROOT / request.run_id).exists():
        raise HTTPException(status_code=409, detail=f"run_id '{request.run_id}' 已存在，请更换名称。")
    job_task = asyncio.create_task(run_agent_experiment(request))
    return {"accepted": True, "action": "experiment", "selection": request.model_dump()}
