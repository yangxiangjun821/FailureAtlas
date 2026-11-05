import json

import pytest

from failure_atlas import web
from failure_atlas.web import (
    build_catalog_payload,
    build_comparison_payload,
    build_preflight_payload,
    build_dashboard_payload,
    build_run_detail,
    build_runs_payload,
    build_system_payload,
    dashboard,
    dashboard_page,
    index,
    ExperimentRequest,
    projects,
)


def _results(accuracy: float) -> list[dict]:
    return [
        {"severity": 0.0, "accuracy": accuracy},
        {"severity": 1.0, "accuracy": accuracy - 0.3},
    ]


@pytest.fixture
def fixture_artifacts(tmp_path, monkeypatch):
    """Point the dashboard at a tiny synthetic experiment, never at local artifacts."""
    names = ("rotation", "blur")
    boundaries = {"mild": 1.0, "obvious": None, "severe": None}

    def analysis(accuracy: float, extra: dict) -> dict:
        return {
            **extra,
            "sample_count": 100,
            "baseline": {"accuracy": accuracy, "ece": 0.02},
            "experiments": {
                name: {"boundaries": boundaries, "results": _results(accuracy)}
                for name in names
            },
        }

    comparison = {
        "baseline": analysis(0.90, {}),
        "repaired": analysis(0.89, {"model": "FashionCNN", "dataset_id": "fashion_mnist"}),
    }
    plan = {
        "actions": [{"perturbation": "rotation", "max_severity": 15.0}],
        "ranking": [{"perturbation": "rotation", "vulnerability_score": 0.3}],
    }
    comparison_path = tmp_path / "repair_comparison.json"
    plan_path = tmp_path / "repair_plan.json"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(web, "COMPARISON_PATH", comparison_path)
    monkeypatch.setattr(web, "PLAN_PATH", plan_path)
    monkeypatch.setattr(web, "DEMO_MODE", False)


def test_dashboard_payload_reflects_comparison_artifacts(fixture_artifacts) -> None:
    payload = build_dashboard_payload()
    # The payload mirrors the artifacts; it must not assume the repair improved anything.
    assert payload["clean_accuracy_before"] == 0.90
    assert payload["clean_accuracy_after"] == 0.89
    assert [item["name"] for item in payload["experiments"]] == ["rotation", "blur"]
    assert len(payload["actions"]) == 1
    assert payload["dataset"] == "Fashion-MNIST"
    assert payload["model"] == "FashionCNN"


def test_dashboard_entry_and_api_function(fixture_artifacts) -> None:
    response = index()
    assert response.status_code == 200
    assert response.path.name == "index.html"
    page = response.path.read_text(encoding="utf-8")
    assert 'id="roadmap"' in page
    assert 'id="roadmap-model-catalog"' in page
    assert 'id="roadmap-dataset-library"' in page
    assert 'id="experiment-config"' in page
    assert 'id="experiment-weight-mode"' in page
    assert 'id="model-catalog"' in page
    assert 'id="runs-table"' in page
    assert 'id="system-check"' in page
    assert 'id="run-detail"' in page
    assert 'id="comparison"' in page
    assert 'id="report-center"' in page
    assert 'href="/dashboard/overview"' in page
    assert 'href="/dashboard/system"' in page
    assert 'src="/static/app.js"' in page
    assert "待办" not in page, "finished catalog work must not be listed as to-do"
    assert dashboard()["dataset"] == "Fashion-MNIST"
    assert dashboard_page().path.name == "index.html"
    built_in = projects()["projects"][0]
    assert built_in["status"] == "analyzed"
    assert built_in["sample_count"] == 100


def test_dashboard_catalog_and_system_are_registry_backed() -> None:
    catalog = build_catalog_payload()
    assert [model["id"] for model in catalog["models"]] == [
        "fashion_cnn",
        "lenet",
        "resnet18",
        "mobilenet_v3",
        "efficientnet_b0",
        "convnext_tiny",
        "vit",
        "vit_b16",
    ]
    assert [dataset["dataset_id"] for dataset in catalog["datasets"]] == [
        "fashion_mnist",
        "mnist",
        "cifar10",
    ]
    assert any(
        item == {"model": "resnet18", "dataset": "cifar10", "compatible": True}
        for item in catalog["compatibility"]
    )
    modern = {model["id"]: model for model in catalog["models"]}
    assert modern["mobilenet_v3"]["pretrained_available"] is True
    assert modern["resnet18"]["pretrained_available"] is False
    assert modern["vit"]["pretrained_available"] is False
    assert modern["vit_b16"]["scratch_available"] is False
    assert modern["vit_b16"]["parameters"] > 80_000_000
    system = build_system_payload()
    assert system["python"]
    assert system["disk_free_bytes"] > 0


def test_run_history_reads_isolated_metadata(tmp_path, monkeypatch) -> None:
    run = tmp_path / "study-01"
    run.mkdir()
    (run / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": "study-01",
                "model": "lenet",
                "dataset": "mnist",
                "stage": "analyze",
                "status": "completed",
                "created_at": "2026-11-03T12:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(web, "RUN_ROOT", tmp_path)
    payload = build_runs_payload()
    assert payload["runs"][0]["run_id"] == "study-01"
    assert payload["runs"][0]["model"] == "lenet"


def _write_run(root, run_id: str, dataset: str = "mnist", accuracy: float = 0.9) -> None:
    run = root / run_id
    run.mkdir()
    metadata = {
        "run_id": run_id,
        "model": "lenet",
        "dataset": dataset,
        "stage": "analyze",
        "status": "completed",
        "created_at": "2026-11-03T12:00:00+00:00",
        "artifacts": ["failure_analysis.json", "agent_report.md"],
    }
    analysis = {
        "baseline": {"accuracy": accuracy, "macro_f1": accuracy - 0.01, "ece": 0.03},
        "experiments": {
            "rotation": {
                "boundaries": {"severe": 20.0},
                "results": [{"severity": 0.0, "accuracy": accuracy}, {"severity": 30.0, "accuracy": 0.5}],
            }
        },
    }
    (run / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run / "failure_analysis.json").write_text(json.dumps(analysis), encoding="utf-8")
    (run / "agent_report.md").write_text("# Report", encoding="utf-8")


def test_run_detail_exposes_metrics_boundaries_and_artifacts(tmp_path, monkeypatch) -> None:
    _write_run(tmp_path, "study-01")
    monkeypatch.setattr(web, "RUN_ROOT", tmp_path)
    detail = build_run_detail("study-01")
    assert detail["metrics"]["accuracy"] == 0.9
    assert detail["experiments"][0]["boundaries_before"]["severe"] == 20.0
    assert detail["artifacts"][1]["name"] == "agent_report.md"


def test_comparison_keeps_cross_dataset_runs_unranked(tmp_path, monkeypatch) -> None:
    _write_run(tmp_path, "mnist-01")
    _write_run(tmp_path, "fashion-01", dataset="fashion_mnist")
    monkeypatch.setattr(web, "RUN_ROOT", tmp_path)
    payload = build_comparison_payload(["mnist-01", "fashion-01"])
    assert payload["comparable"] is False
    assert "不进行高低排名" in payload["limitation"]


def test_comparison_returns_compatible_runs(tmp_path, monkeypatch) -> None:
    _write_run(tmp_path, "mnist-01", accuracy=0.91)
    _write_run(tmp_path, "mnist-02", accuracy=0.93)
    monkeypatch.setattr(web, "RUN_ROOT", tmp_path)
    payload = build_comparison_payload(["mnist-01", "mnist-02"])
    assert payload["dataset"] == "mnist"
    assert [run["metrics"]["accuracy"] for run in payload["runs"]] == [0.91, 0.93]


def test_preflight_reports_steps_and_selected_perturbations() -> None:
    payload = build_preflight_payload(
        ExperimentRequest(
            model="lenet",
            dataset="mnist",
            stage="analyze",
            perturbations=["rotation", "blur"],
            seed=7,
            batch_size=64,
            severity_scale=0.5,
        )
    )
    assert payload["compatible"] is True
    assert payload["selected_perturbations"] == ["rotation", "blur"]
    assert [step["key"] for step in payload["steps"]] == ["data", "baseline", "scan"]


def test_preflight_reports_pretrained_download_and_rejects_unsupported(monkeypatch) -> None:
    monkeypatch.setattr(web, "pretrained_weight_cached", lambda model_id: False)
    payload = build_preflight_payload(
        ExperimentRequest(
            model="mobilenet_v3",
            dataset="cifar10",
            stage="baseline",
            weight_mode="pretrained",
            # Do not depend on whichever artifacts happen to exist on this machine.
            reuse_artifacts=False,
        )
    )
    assert payload["weight_mode"] == "pretrained"
    assert payload["download_required"] is True
    assert all(step["will_run"] for step in payload["steps"])

    with pytest.raises(web.HTTPException, match="does not provide pretrained weights"):
        build_preflight_payload(
            ExperimentRequest(
                model="lenet",
                dataset="mnist",
                stage="baseline",
                weight_mode="pretrained",
            )
        )


def test_static_routes_are_not_shadowed_by_parameterized_routes() -> None:
    """Starlette serves the first matching route, so declaration order matters."""
    from starlette.routing import Match

    methods_by_path: list[tuple[str, str, str]] = []
    for route in web.app.routes:
        path = getattr(route, "path", "")
        if "{" in path or not path.startswith("/api/"):
            continue
        for method in getattr(route, "methods", None) or ():
            if method != "HEAD":
                methods_by_path.append((path, method, route.endpoint.__name__))
    assert methods_by_path, "expected static API routes"
    for path, method, endpoint in methods_by_path:
        scope = {"type": "http", "path": path, "method": method, "root_path": ""}
        first = next(
            route for route in web.app.routes if route.matches(scope)[0] == Match.FULL
        )
        assert first.endpoint.__name__ == endpoint, (
            f"{method} {path} is handled by {first.path} instead of {endpoint}"
        )


def test_experiment_job_route_reaches_its_own_handler() -> None:
    from starlette.routing import Match

    scope = {"type": "http", "path": "/api/jobs/experiment", "method": "POST", "root_path": ""}
    first = next(route for route in web.app.routes if route.matches(scope)[0] == Match.FULL)
    assert first.endpoint.__name__ == "start_experiment"


def test_preflight_resolves_the_natural_weight_mode_for_pretrained_only_models() -> None:
    payload = build_preflight_payload(
        ExperimentRequest(model="vit_b16", dataset="cifar10", stage="baseline")
    )
    assert payload["weight_mode"] == "pretrained"
    with pytest.raises(web.HTTPException, match="only defined with official pretrained"):
        build_preflight_payload(
            ExperimentRequest(
                model="vit_b16", dataset="cifar10", stage="baseline", weight_mode="scratch"
            )
        )


def test_pretrained_runs_are_estimated_slower_than_scratch_runs() -> None:
    scratch = build_preflight_payload(
        ExperimentRequest(
            model="mobilenet_v3", dataset="cifar10", stage="full", reuse_artifacts=False
        )
    )
    pretrained = build_preflight_payload(
        ExperimentRequest(
            model="mobilenet_v3",
            dataset="cifar10",
            stage="full",
            reuse_artifacts=False,
            weight_mode="pretrained",
        )
    )
    assert pretrained["estimated_minutes"] > scratch["estimated_minutes"]
