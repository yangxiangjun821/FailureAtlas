"""Local FastAPI dashboard for FailureAtlas experiments."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "web"
REPORT_ROOT = PROJECT_ROOT / "reports" / "generated"
COMPARISON_PATH = REPORT_ROOT / "repair_comparison.json"
PLAN_PATH = PROJECT_ROOT / "artifacts" / "repair_plan.json"
WORKSPACE_ROOT = PROJECT_ROOT / "workspaces"
DEPLOY_SNAPSHOT_PATH = PROJECT_ROOT / "deploy" / "dashboard.json"
DEMO_MODE = os.getenv("FAILUREATLAS_DEMO", "0") == "1"

app = FastAPI(title="FailureAtlas", version="0.2.0")
app.mount("/static", StaticFiles(directory=WEB_ROOT / "static"), name="static")
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
    return {
        "deployment_mode": "local",
        "model": "FashionCNN",
        "dataset": "Fashion-MNIST",
        "sample_count": after["sample_count"],
        "clean_accuracy_before": before["baseline"]["accuracy"],
        "clean_accuracy_after": after["baseline"]["accuracy"],
        "clean_ece_before": before["baseline"]["ece"],
        "clean_ece_after": after["baseline"]["ece"],
        "actions": plan["actions"],
        "ranking": plan["ranking"],
        "experiments": experiments,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/api/dashboard")
def dashboard() -> dict:
    return build_dashboard_payload()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "mode": "showcase" if DEMO_MODE else "local"}


@app.get("/api/jobs/current")
def current_job() -> dict:
    return job_state


@app.get("/api/projects")
def projects() -> dict:
    imported = []
    if not DEMO_MODE:
        from .bundle import list_imported_projects

        imported = list_imported_projects(WORKSPACE_ROOT)
    built_in = {
        "id": "fashion-mnist-built-in",
        "name": "Fashion-MNIST repaired CNN",
        "task": "image_classification",
        "sample_count": 10_000,
        "class_count": 10,
        "input_shape": [-1, 1, 28, 28],
        "clean_accuracy": build_dashboard_payload()["clean_accuracy_after"],
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
    if content_length and int(content_length) > MAX_BUNDLE_BYTES:
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
