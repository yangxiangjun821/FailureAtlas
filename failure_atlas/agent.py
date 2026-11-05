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
    resolve_weight_mode,
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
from .paths import ExperimentPaths, experiment_paths
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


DATA_MARKERS = {
    "fashion_mnist": ROOT / "data/FashionMNIST/raw/train-images-idx3-ubyte",
    "mnist": ROOT / "data/MNIST/raw/train-images-idx3-ubyte",
    "cifar10": ROOT / "data/cifar-10-batches-py/data_batch_1",
}


def pipeline_for(
    model_id: str = "fashion_cnn",
    dataset_id: str = "fashion_mnist",
    weight_mode: str | None = None,
) -> tuple[PipelineStep, ...]:
    """Pipeline whose expected outputs are isolated per model, dataset and weight mode."""
    paths = experiment_paths(model_id, dataset_id, weight_mode)
    dataset = DATASET_REGISTRY.get(dataset_id)
    return (
        PipelineStep(
            "data",
            f"Prepare the {dataset.display_name} dataset",
            "scripts.download_data",
            (DATA_MARKERS[dataset_id],),
        ),
        PipelineStep(
            "baseline",
            "Train and verify the baseline model",
            "scripts.train_baseline",
            (paths.baseline_checkpoint, paths.baseline_metrics),
        ),
        PipelineStep(
            "scan",
            "Discover failure boundaries",
            "scripts.search_failures",
            (paths.failure_analysis, paths.failure_report),
        ),
        PipelineStep(
            "repair",
            "Select targeted repairs and fine-tune",
            "scripts.repair_model",
            (paths.repaired_checkpoint, paths.repair_plan),
        ),
        PipelineStep(
            "compare",
            "Re-scan and compare the repaired model",
            "scripts.compare_repair",
            (paths.repair_comparison, paths.repair_report),
        ),
    )


def steps_for(
    stage: str,
    model_id: str = "fashion_cnn",
    dataset_id: str = "fashion_mnist",
    weight_mode: str | None = None,
) -> tuple[PipelineStep, ...]:
    pipeline = pipeline_for(model_id, dataset_id, weight_mode)
    if stage == "baseline":
        return pipeline[:2]
    return pipeline[:3] if stage == "analyze" else pipeline


def generate_baseline_report(metrics_path: Path, output_path: Path) -> Path:
    metrics = _load_json(metrics_path)
    model = metrics.get("model", {})
    dataset = metrics.get("dataset", {})
    test = metrics.get("test", {})
    split = metrics.get("split", {})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                "# FailureAtlas 基准训练报告",
                "",
                f"- 模型：{model.get('display_name', model.get('model_id', 'unknown'))}",
                f"- 数据集：{dataset.get('display_name', dataset.get('dataset_id', 'unknown'))}",
                f"- Accuracy：{test.get('accuracy', 0):.4f}",
                f"- Macro-F1：{test.get('macro_f1', 0):.4f}",
                f"- 参数量：{model.get('parameter_count', 'unknown')}",
                f"- 随机种子：{split.get('seed', 'unknown')}",
                "",
                "该运行只建立基准模型；失效边界、修复方案与修复验证尚未执行。",
            ]
        ),
        encoding="utf-8",
    )
    return output_path


def command_for(
    step: PipelineStep,
    baseline_epochs: int | None = None,
    model_id: str = "fashion_cnn",
    dataset_id: str = "fashion_mnist",
    seed: int = 42,
    batch_size: int | None = None,
    perturbations: tuple[str, ...] | None = None,
    severity_scale: float = 1.0,
    weight_mode: str | None = None,
) -> list[str]:
    command = [sys.executable, "-m", step.module, "--model", model_id, "--dataset", dataset_id]
    if step.key == "baseline" and baseline_epochs is not None:
        command.extend(["--epochs", str(baseline_epochs)])
    if step.key in {"baseline", "scan", "repair", "compare"}:
        command.extend(["--weight-mode", resolve_weight_mode(model_id, weight_mode)])
        command.extend(["--seed", str(seed)])
    if batch_size is not None and step.key in {"baseline", "scan", "repair", "compare"}:
        command.extend(["--batch-size", str(batch_size)])
    if step.key in {"scan", "compare"}:
        command.extend(["--severity-scale", str(severity_scale)])
        for perturbation in perturbations or ():
            command.extend(["--perturbation", perturbation])
    return command


def print_plan(
    stage: str,
    baseline_epochs: int | None = None,
    model_id: str = "fashion_cnn",
    dataset_id: str = "fashion_mnist",
    seed: int = 42,
    batch_size: int | None = None,
    perturbations: tuple[str, ...] | None = None,
    severity_scale: float = 1.0,
    weight_mode: str | None = None,
) -> None:
    validate_selection(model_id, dataset_id, perturbations)
    weight_mode = resolve_weight_mode(model_id, weight_mode)
    paths = experiment_paths(model_id, dataset_id, weight_mode)
    print(f"FailureAtlas local agent plan ({stage})")
    print(f"  selection. model={model_id}, dataset={dataset_id}, weights={weight_mode}")
    for index, step in enumerate(steps_for(stage, model_id, dataset_id, weight_mode), start=1):
        status = "ready" if step.ready else "pending"
        command = " ".join(
            command_for(
                step,
                baseline_epochs,
                model_id,
                dataset_id,
                seed,
                batch_size,
                perturbations,
                severity_scale,
                weight_mode,
            )
        )
        print(f"  {index}. [{status}] {step.title}")
        print(f"     {command}")
    print(f"  report. Generate {_display_path(paths.agent_report)}")


def run_pipeline(
    stage: str,
    *,
    force: bool = False,
    dry_run: bool = False,
    baseline_epochs: int | None = None,
    model_id: str = "fashion_cnn",
    dataset_id: str = "fashion_mnist",
    seed: int = 42,
    batch_size: int | None = None,
    perturbations: tuple[str, ...] | None = None,
    severity_scale: float = 1.0,
    weight_mode: str | None = None,
    run_id: str | None = None,
    runs_root: str | Path | None = None,
) -> Path | None:
    validate_selection(model_id, dataset_id, perturbations)
    weight_mode = resolve_weight_mode(model_id, weight_mode)
    print_plan(
        stage,
        baseline_epochs,
        model_id,
        dataset_id,
        seed,
        batch_size,
        perturbations,
        severity_scale,
        weight_mode,
    )
    if dry_run:
        return None

    paths = experiment_paths(model_id, dataset_id, weight_mode)
    workspace = create_run_workspace(
        runs_root or ROOT / "runs",
        model_id=model_id,
        dataset_id=dataset_id,
        stage=stage,
        seed=seed,
        batch_size=batch_size,
        severity_scale=severity_scale,
        weight_mode=weight_mode,
        perturbation_ids=perturbations,
        run_id=run_id,
    )
    print(f"[run]  workspace={workspace.path}")
    # Once a step re-runs, every later step is stale and must re-run too.
    rerun = force
    try:
        for step in steps_for(stage, model_id, dataset_id, weight_mode):
            if step.ready and not rerun:
                print(f"[skip] {step.title}: expected outputs already exist")
                continue
            rerun = True
            command = command_for(
                step,
                baseline_epochs,
                model_id,
                dataset_id,
                seed,
                batch_size,
                perturbations,
                severity_scale,
                weight_mode,
            )
            print(f"[run]  {step.title}")
            subprocess.run(command, cwd=ROOT, check=True)
            if not step.ready:
                missing = [_display_path(path) for path in step.outputs if not path.exists()]
                raise RuntimeError(f"Step '{step.key}' completed without outputs: {missing}")

        report = (
            generate_baseline_report(paths.baseline_metrics, paths.agent_report)
            if stage == "baseline"
            else generate_agent_report(
                paths.failure_analysis,
                paths.repair_comparison,
                paths.agent_report,
                model_id=model_id,
                dataset_id=dataset_id,
                weight_mode=weight_mode,
                include_comparison=stage == "full",
            )
        )
        archived = _archive_current_outputs(workspace, report, paths, stage)
        required = (
            ("config.yaml", "metadata.json", "baseline_metrics.json", "agent_report.md")
            if stage == "baseline"
            else ("config.yaml", "metadata.json", "failure_analysis.json", "agent_report.md")
        )
        validate_run_artifacts(workspace, required)
        finalize_run(workspace, status="completed", artifacts=archived)
        run_report = workspace.path / "agent_report.md"
        print(f"[complete] run_id={workspace.run_id}")
        return run_report
    except Exception as error:
        finalize_run(workspace, status="failed", error=str(error))
        raise


def archive_sources(paths: ExperimentPaths, report: Path, stage: str) -> dict[str, Path]:
    """Files a run archives. Repair evidence belongs only to full runs: an analyze
    run must not pick up leftovers from an earlier full run of the same selection."""
    sources = {
        "baseline_metrics.json": paths.baseline_metrics,
        "agent_report.md": report,
    }
    if stage == "baseline":
        return sources
    sources.update(
        {
            "failure_analysis.json": paths.failure_analysis,
            "plots/failure_curves.png": paths.failure_curves,
        }
    )
    if stage == "full":
        sources.update(
            {
                "repair_plan.json": paths.repair_plan,
                "repair_comparison.json": paths.repair_comparison,
                "plots/repair_comparison.png": paths.repair_comparison_plot,
            }
        )
    return sources


def _archive_current_outputs(
    workspace: RunWorkspace, report: Path, paths: ExperimentPaths, stage: str
) -> list[str]:
    return archive_run_artifacts(workspace, archive_sources(paths, report, stage))


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


LIMITATIONS = [
    "## Limitations",
    "",
    "- Repair augmentations come from the same perturbation families that the scan "
    "measures, so boundary movement shows robustness to *known* shifts, not "
    "generalization to unseen ones.",
    "- Failure boundaries are measured on the test split with a single fixed seed; "
    "no confidence intervals are reported.",
    "- A validation split is held out of both baseline training and repair; "
    "repair checkpoints are selected on it, and the test split is used only for "
    "final reporting. The baseline therefore trains on less data than the full "
    "training split.",
    "- A better severe boundary does not imply every metric improved; check the clean "
    "accuracy change and per-perturbation endpoint changes above.",
]


def _boundary(value: float | None) -> str:
    return "not reached" if value is None else f"{value:g}"


def generate_agent_report(
    analysis_path: Path | None = None,
    comparison_path: Path | None = None,
    output_path: Path | None = None,
    *,
    model_id: str = "fashion_cnn",
    dataset_id: str = "fashion_mnist",
    weight_mode: str | None = None,
    include_comparison: bool = True,
) -> Path:
    weight_mode = resolve_weight_mode(model_id, weight_mode)
    paths = experiment_paths(model_id, dataset_id, weight_mode)
    analysis_path = analysis_path or paths.failure_analysis
    comparison_path = comparison_path or paths.repair_comparison
    output_path = output_path or paths.agent_report
    if not analysis_path.exists():
        raise FileNotFoundError(
            f"No failure analysis found. Run `python -m failure_atlas.agent run --stage analyze "
            f"--model {model_id} --dataset {dataset_id} --weight-mode {weight_mode}` first."
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

    comparison = (
        _load_json(comparison_path)
        if include_comparison and comparison_path.exists()
        else None
    )
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

    lines.extend(["", *LIMITATIONS])
    lines.extend(
        [
            "",
            "## Reproduce locally",
            "",
            "```powershell",
            f"python -m failure_atlas.agent plan --stage full --model {model_id} --dataset {dataset_id} --weight-mode {weight_mode}",
            f"python -m failure_atlas.agent run --stage full --model {model_id} --dataset {dataset_id} --weight-mode {weight_mode}",
            "```",
            "",
            "Generated artifacts remain local by default under "
            f"`{_display_path(paths.artifact_dir)}` and `{_display_path(paths.report_dir)}` "
            "so model weights and datasets are not accidentally committed.",
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
        subparser.add_argument("--stage", choices=("baseline", "analyze", "full"), default="full")
        subparser.add_argument("--baseline-epochs", type=int)
        subparser.add_argument("--model", default="fashion_cnn")
        subparser.add_argument("--dataset", default="fashion_mnist")
        subparser.add_argument("--seed", type=int, default=42)
        subparser.add_argument("--batch-size", type=int)
        subparser.add_argument("--perturbation", action="append", dest="perturbations")
        subparser.add_argument("--severity-scale", type=float, default=1.0)
        subparser.add_argument("--weight-mode", choices=("scratch", "pretrained"))
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
    report.add_argument("--model", default="fashion_cnn")
    report.add_argument("--dataset", default="fashion_mnist")
    report.add_argument("--weight-mode", choices=("scratch", "pretrained"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "plan":
        print_plan(
            args.stage,
            args.baseline_epochs,
            args.model,
            args.dataset,
            args.seed,
            args.batch_size,
            tuple(args.perturbations) if args.perturbations else None,
            args.severity_scale,
            args.weight_mode,
        )
    elif args.command == "run":
        run_pipeline(
            args.stage,
            force=args.force,
            dry_run=args.dry_run,
            baseline_epochs=args.baseline_epochs,
            model_id=args.model,
            dataset_id=args.dataset,
            seed=args.seed,
            batch_size=args.batch_size,
            perturbations=tuple(args.perturbations) if args.perturbations else None,
            severity_scale=args.severity_scale,
            weight_mode=args.weight_mode,
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
        generate_agent_report(
            output_path=args.output,
            model_id=args.model,
            dataset_id=args.dataset,
            weight_mode=args.weight_mode,
        )


if __name__ == "__main__":
    main()
