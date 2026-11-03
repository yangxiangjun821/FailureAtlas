"""Local-first orchestration agent for FailureAtlas experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .catalog import (
    DATASET_REGISTRY,
    MODEL_REGISTRY,
    PERTURBATION_REGISTRY,
    validate_selection,
)
from .comparison import boundary_delta
from .experiments import (
    RunWorkspace,
    archive_run_artifacts,
    create_run_workspace,
    finalize_run,
    validate_run_artifacts,
)
from .repair_agent import create_repair_plan
from .reporting import DISPLAY_NAMES


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PipelineStep:
    key: str
    title: str
    module: str
    outputs: tuple[Path, ...]

    @property
    def ready(self) -> bool:
        return bool(self.outputs) and all(path.exists() for path in self.outputs)


PIPELINE = (
    PipelineStep(
        "data",
        "Prepare the Fashion-MNIST dataset",
        "scripts.download_data",
        (ROOT / "data/FashionMNIST/raw/train-images-idx3-ubyte",),
    ),
    PipelineStep(
        "baseline",
        "Train and verify the baseline model",
        "scripts.train_baseline",
        (
            ROOT / "artifacts/baseline_fashion_cnn.pt",
            ROOT / "artifacts/baseline_metrics.json",
        ),
    ),
    PipelineStep(
        "scan",
        "Discover failure boundaries",
        "scripts.search_failures",
        (
            ROOT / "reports/generated/failure_analysis.json",
            ROOT / "reports/generated/failure_report.md",
        ),
    ),
    PipelineStep(
        "repair",
        "Select targeted repairs and fine-tune",
        "scripts.repair_model",
        (
            ROOT / "artifacts/repaired_fashion_cnn.pt",
            ROOT / "artifacts/repair_plan.json",
        ),
    ),
    PipelineStep(
        "compare",
        "Re-scan and compare the repaired model",
        "scripts.compare_repair",
        (
            ROOT / "reports/generated/repair_comparison.json",
            ROOT / "reports/generated/repair_report.md",
        ),
    ),
)


def steps_for(stage: str) -> tuple[PipelineStep, ...]:
    return PIPELINE[:3] if stage == "analyze" else PIPELINE


def command_for(step: PipelineStep, baseline_epochs: int | None = None) -> list[str]:
    command = [sys.executable, "-m", step.module]
    if step.key == "baseline" and baseline_epochs is not None:
        command.extend(["--epochs", str(baseline_epochs)])
    return command


def print_plan(
    stage: str,
    baseline_epochs: int | None = None,
    model_id: str = "fashion_cnn",
    dataset_id: str = "fashion_mnist",
) -> None:
    validate_selection(model_id, dataset_id)
    print(f"FailureAtlas local agent plan ({stage})")
    print(f"  selection. model={model_id}, dataset={dataset_id}")
    for index, step in enumerate(steps_for(stage), start=1):
        status = "ready" if step.ready else "pending"
        command = " ".join(command_for(step, baseline_epochs))
        print(f"  {index}. [{status}] {step.title}")
        print(f"     {command}")
    print("  report. Generate reports/generated/agent_report.md")


def run_pipeline(
    stage: str,
    *,
    force: bool = False,
    dry_run: bool = False,
    baseline_epochs: int | None = None,
    model_id: str = "fashion_cnn",
    dataset_id: str = "fashion_mnist",
    run_id: str | None = None,
    runs_root: str | Path | None = None,
) -> Path | None:
    validate_selection(model_id, dataset_id)
    print_plan(stage, baseline_epochs, model_id, dataset_id)
    if dry_run:
        return None

    workspace = create_run_workspace(
        runs_root or ROOT / "runs",
        model_id=model_id,
        dataset_id=dataset_id,
        stage=stage,
        run_id=run_id,
    )
    print(f"[run]  workspace={workspace.path}")
    try:
        for step in steps_for(stage):
            if step.ready and not force:
                print(f"[skip] {step.title}: expected outputs already exist")
                continue
            command = command_for(step, baseline_epochs)
            print(f"[run]  {step.title}")
            subprocess.run(command, cwd=ROOT, check=True)
            if not step.ready:
                missing = [str(path.relative_to(ROOT)) for path in step.outputs if not path.exists()]
                raise RuntimeError(f"Step '{step.key}' completed without outputs: {missing}")

        report = generate_agent_report()
        archived = _archive_current_outputs(workspace, report)
        required = ("config.yaml", "metadata.json", "failure_analysis.json", "agent_report.md")
        validate_run_artifacts(workspace, required)
        finalize_run(workspace, status="completed", artifacts=archived)
        run_report = workspace.path / "agent_report.md"
        print(f"[complete] run_id={workspace.run_id}")
        return run_report
    except Exception as error:
        finalize_run(workspace, status="failed", error=str(error))
        raise


def _archive_current_outputs(workspace: RunWorkspace, report: Path) -> list[str]:
    sources = {
        "baseline_metrics.json": ROOT / "artifacts/baseline_metrics.json",
        "failure_analysis.json": ROOT / "reports/generated/failure_analysis.json",
        "repair_plan.json": ROOT / "artifacts/repair_plan.json",
        "repair_comparison.json": ROOT / "reports/generated/repair_comparison.json",
        "agent_report.md": report,
        "plots/failure_curves.png": ROOT / "reports/generated/failure_curves.png",
        "plots/repair_comparison.png": ROOT / "reports/generated/repair_comparison.png",
    }
    return archive_run_artifacts(workspace, sources)


def print_catalog() -> None:
    print("Models")
    for spec in MODEL_REGISTRY.values():
        status = "ready" if spec.implemented else "planned"
        print(f"  {spec.id:<16} {status:<8} {spec.display_name}")
    print("Datasets")
    for spec in DATASET_REGISTRY.values():
        status = "ready" if spec.implemented else "planned"
        print(f"  {spec.id:<16} {status:<8} {spec.display_name}")
    print("Perturbations")
    for spec in PERTURBATION_REGISTRY.values():
        status = "ready" if spec.implemented else "planned"
        print(f"  {spec.id:<16} {status:<8} {spec.display_name}")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _display_path(path: Path) -> str:
    """Prefer repository-relative paths while allowing external report targets."""
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _boundary(value: float | None) -> str:
    return "not reached" if value is None else f"{value:g}"


def generate_agent_report(
    analysis_path: Path | None = None,
    comparison_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    analysis_path = analysis_path or ROOT / "reports/generated/failure_analysis.json"
    comparison_path = comparison_path or ROOT / "reports/generated/repair_comparison.json"
    output_path = output_path or ROOT / "reports/generated/agent_report.md"
    if not analysis_path.exists():
        raise FileNotFoundError(
            "No failure analysis found. Run `python -m failure_atlas.agent run --stage analyze` first."
        )

    analysis = _load_json(analysis_path)
    plan = create_repair_plan(analysis)
    baseline = analysis["baseline"]
    lines = [
        "# FailureAtlas Local Agent Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Experiment identity",
        "",
        f"- Model: {analysis['model']}",
        f"- Dataset: {analysis['dataset']}",
        f"- Test samples: {analysis['sample_count']:,}",
        f"- Clean accuracy: {baseline['accuracy']:.2%}",
        f"- Macro-F1: {baseline['macro_f1']:.2%}",
        f"- Expected calibration error: {baseline['ece']:.4f}",
        "",
        "## Failure boundary summary",
        "",
        "| Perturbation | Mild | Obvious | Severe | Max-severity accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, experiment in analysis["experiments"].items():
        boundaries = experiment["boundaries"]
        endpoint = experiment["results"][-1]["accuracy"]
        lines.append(
            f"| {DISPLAY_NAMES[name]} | {_boundary(boundaries['mild'])} | "
            f"{_boundary(boundaries['obvious'])} | {_boundary(boundaries['severe'])} | "
            f"{endpoint:.2%} |"
        )

    lines.extend(["", "## Agent diagnosis and proposed repair", ""])
    for rank, action in enumerate(plan["actions"], start=1):
        lines.append(
            f"{rank}. **{DISPLAY_NAMES[action['perturbation']]}** — vulnerability "
            f"score {action['vulnerability_score']:.3f}. {action['rationale']}"
        )

    comparison = _load_json(comparison_path) if comparison_path.exists() else None
    if comparison:
        repaired = comparison["repaired"]
        clean_delta = repaired["baseline"]["accuracy"] - baseline["accuracy"]
        lines.extend(
            [
                "",
                "## Repair verification",
                "",
                f"- Repaired clean accuracy: {repaired['baseline']['accuracy']:.2%}",
                f"- Clean-accuracy change: {clean_delta:+.2%}",
                "",
                "| Perturbation | Severe boundary before → after | Endpoint change |",
                "|---|---:|---:|",
            ]
        )
        for name, before in analysis["experiments"].items():
            after = repaired["experiments"][name]
            movement = boundary_delta(
                before["boundaries"]["severe"], after["boundaries"]["severe"]
            )
            endpoint_delta = after["results"][-1]["accuracy"] - before["results"][-1]["accuracy"]
            lines.append(f"| {DISPLAY_NAMES[name]} | {movement} | {endpoint_delta:+.2%} |")

    lines.extend(
        [
            "",
            "## Reproduce locally",
            "",
            "```powershell",
            "python -m failure_atlas.agent plan --stage full",
            "python -m failure_atlas.agent run --stage full",
            "```",
            "",
            "Generated artifacts remain local by default under `artifacts/` and "
            "`reports/generated/` so model weights and datasets are not accidentally committed.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": "full" if comparison else "analyze",
        "model": analysis["model"],
        "dataset": analysis["dataset"],
        "report": _display_path(output_path),
        "source_artifacts": [
            _display_path(analysis_path),
            *([_display_path(comparison_path)] if comparison else []),
        ],
    }
    output_path.with_name("agent_run.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[report] {output_path}")
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="failure-atlas",
        description="Run the local FailureAtlas diagnosis and repair agent.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "run"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--stage", choices=("analyze", "full"), default="full")
        subparser.add_argument("--baseline-epochs", type=int)
        subparser.add_argument("--model", default="fashion_cnn")
        subparser.add_argument("--dataset", default="fashion_mnist")
        if command == "run":
            subparser.add_argument("--force", action="store_true")
            subparser.add_argument("--dry-run", action="store_true")
            subparser.add_argument("--run-id")
    subparsers.add_parser("catalog")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--model", required=True)
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--perturbation", action="append", dest="perturbations")
    report = subparsers.add_parser("report")
    report.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "plan":
        print_plan(args.stage, args.baseline_epochs, args.model, args.dataset)
    elif args.command == "run":
        run_pipeline(
            args.stage,
            force=args.force,
            dry_run=args.dry_run,
            baseline_epochs=args.baseline_epochs,
            model_id=args.model,
            dataset_id=args.dataset,
            run_id=args.run_id,
        )
    elif args.command == "catalog":
        print_catalog()
    elif args.command == "validate":
        validate_selection(
            args.model,
            args.dataset,
            tuple(args.perturbations) if args.perturbations else None,
        )
        print(f"Valid selection: model={args.model}, dataset={args.dataset}")
    else:
        generate_agent_report(output_path=args.output)


if __name__ == "__main__":
    main()
