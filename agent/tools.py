"""Wrap the existing FailureAtlas functions as four small tools an LLM can call.

Each tool returns a compact dict (a few numbers, no tensors or confusion matrices) so
the conversation stays short. Every result says which data it was computed on
(``source`` and ``sample_count``), so the model's summary can stay honest about it.

Demo mode keeps a full agent run within a few minutes:
- evaluation and boundary scans use a fixed random subset of the test split;
- the vulnerability ranking reuses the cached full-test-set scan when one exists;
- the repair fine-tunes on a small slice of the training split for one epoch.
All numbers still come from real model runs; demo mode only uses less data.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset

from failure_atlas.adapters import get_dataset_adapter
from failure_atlas.calibration import load_calibration
from failure_atlas.catalog import (
    DATASET_REGISTRY,
    MODEL_REGISTRY,
    WEIGHT_MODES,
    resolve_weight_mode,
    validate_selection,
)
from failure_atlas.config import effective_batch_size, effective_learning_rate
from failure_atlas.config import TrainConfig
from failure_atlas.failure_search import evaluate_perturbation
from failure_atlas.paths import ROOT, ExperimentPaths, experiment_paths
from failure_atlas.repair_agent import (
    ACTION_SETTINGS,
    TargetedAugment,
    score_vulnerability,
    target_severity,
)
from failure_atlas.severity import severities_for
from failure_atlas.training import (
    fit_batch_size,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    set_seed,
    train_one_epoch,
)


SEARCH_CONFIG = ROOT / "configs" / "failure_search.yaml"
REPAIR_CONFIG = ROOT / "configs" / "repair.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "agent_runs"
PERTURBATION_IDS = ("rotation", "gaussian_noise", "blur", "occlusion", "brightness")


@dataclass
class ToolSettings:
    """How much data the tools use. ``demo=False`` uses the full splits."""

    demo: bool = True
    test_samples: int = 1000
    train_samples: int = 2000
    repair_epochs: int = 1
    seed: int = 42
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


class FailureAtlasTools:
    """The tool box. It remembers the repaired checkpoint so later calls can re-test it."""

    def __init__(self, settings: ToolSettings | None = None) -> None:
        self.settings = settings or ToolSettings()
        self.search = yaml.safe_load(SEARCH_CONFIG.read_text(encoding="utf-8"))["search"]
        self.repair = yaml.safe_load(REPAIR_CONFIG.read_text(encoding="utf-8"))["repair"]
        self.device = resolve_device(self.search["device"])
        # (model, dataset) -> (checkpoint path, perturbation it was repaired for)
        self.repaired: dict[tuple[str, str], tuple[Path, str]] = {}
        self._test_sets: dict[str, Subset] = {}

    # ------------------------------------------------------------------ dispatch
    @property
    def schemas(self) -> list[dict]:
        return TOOL_SCHEMAS

    def call(self, name: str, arguments: str | dict | None) -> dict:
        """Run one tool. Errors come back as ``{"error": ...}`` instead of raising."""
        handlers = {
            "evaluate_model": self.evaluate_model,
            "find_failure_boundary": self.find_failure_boundary,
            "rank_vulnerability": self.rank_vulnerability,
            "apply_targeted_repair": self.apply_targeted_repair,
        }
        if name not in handlers:
            return {"error": f"未知工具 '{name}'，可用工具：{', '.join(handlers)}"}
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
            if not isinstance(args, dict):
                raise ValueError("参数必须是 JSON 对象")
        except (json.JSONDecodeError, ValueError) as error:
            return {"error": f"参数不是合法的 JSON 对象：{error}"}
        try:
            return handlers[name](**args)
        except TypeError as error:
            return {"error": f"参数错误：{error}"}
        except Exception as error:  # the model sees the message and can adjust
            return {"error": f"{type(error).__name__}: {error}"}

    # ------------------------------------------------------------------ helpers
    def _locate(self, model: str, dataset: str) -> ExperimentPaths:
        """Find the baseline checkpoint, trying the model's natural weight mode first."""
        validate_selection(model, dataset)
        natural = resolve_weight_mode(model)
        modes = [natural] + [mode for mode in WEIGHT_MODES if mode != natural]
        for mode in modes:
            if not MODEL_REGISTRY.get(model).supports_weight_mode(mode):
                continue
            paths = experiment_paths(model, dataset, mode)
            if paths.baseline_checkpoint.is_file():
                return paths
        available = ", ".join(f"{m}/{d}" for m, d in available_selections()) or "无"
        raise FileNotFoundError(
            f"没有 {model} 在 {dataset} 上训练好的基准模型。已有基准模型的组合：{available}"
        )

    def _load(self, paths: ExperimentPaths, checkpoint: str) -> nn.Module:
        if checkpoint == "baseline":
            path = paths.baseline_checkpoint
        elif checkpoint == "repaired":
            key = (paths.model_id, paths.dataset_id)
            if key not in self.repaired:
                raise ValueError("本次会话还没有修复过该模型，请先调用 apply_targeted_repair。")
            path = self.repaired[key][0]
        else:
            raise ValueError("checkpoint 只能是 'baseline' 或 'repaired'。")
        model, _ = load_checkpoint(path, self.device)
        return model

    def _test_loader(self, dataset: str, model: nn.Module) -> DataLoader:
        if dataset not in self._test_sets:
            adapter = get_dataset_adapter(dataset)
            full = adapter.raw_test_loader(self.search["data_dir"], 1).dataset
            indices = list(range(len(full)))
            if self.settings.demo:
                generator = torch.Generator().manual_seed(self.settings.seed)
                order = torch.randperm(len(full), generator=generator).tolist()
                indices = sorted(order[: self.settings.test_samples])
            self._test_sets[dataset] = Subset(full, indices)
        weight_mode = getattr(model, "weight_mode", "scratch")
        batch = effective_batch_size(self.search["batch_size"], weight_mode)
        batch = fit_batch_size(model, batch, self.device, training=False,
                               sample_shape=_sample_shape(dataset))
        return DataLoader(self._test_sets[dataset], batch_size=batch, shuffle=False)

    def _scan(self, model: nn.Module, loader: DataLoader, dataset: str,
              perturbation: str, severity: float) -> dict:
        adapter = get_dataset_adapter(dataset)
        mean, std = adapter.normalization
        return evaluate_perturbation(
            model, loader, self.device, perturbation, severity,
            self.search["seed"], adapter.class_names, mean, std,
        )

    def _source(self) -> str:
        return "live_demo_subset" if self.settings.demo else "live_full_test"

    def _grid(self, dataset: str, perturbation: str) -> list[float]:
        if perturbation not in PERTURBATION_IDS:
            raise ValueError(f"未知扰动 '{perturbation}'，可选：{', '.join(PERTURBATION_IDS)}")
        grids = severities_for(self.search, load_calibration(dataset))
        return grids[perturbation]

    # ------------------------------------------------------------------ tools
    def evaluate_model(self, model: str, dataset: str, checkpoint: str = "baseline") -> dict:
        paths = self._locate(model, dataset)
        network = self._load(paths, checkpoint)
        loader = self._test_loader(dataset, network)
        metrics = self._scan(network, loader, dataset, "rotation", 0.0)
        return {
            "model": model,
            "dataset": dataset,
            "weight_mode": paths.weight_mode,
            "checkpoint": checkpoint,
            "accuracy": _round(metrics["accuracy"]),
            "ece": _round(metrics["ece"]),
            "macro_f1": _round(metrics["macro_f1"]),
            "sample_count": len(loader.dataset),
            "source": self._source(),
        }

    def find_failure_boundary(
        self,
        model: str,
        dataset: str,
        perturbation: str,
        threshold: float = 0.10,
        checkpoint: str = "baseline",
    ) -> dict:
        threshold = float(threshold)
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold 是准确率下降量，必须在 0 到 1 之间，例如 0.10。")
        grid = self._grid(dataset, perturbation)
        paths = self._locate(model, dataset)
        network = self._load(paths, checkpoint)
        loader = self._test_loader(dataset, network)
        clean = self._scan(network, loader, dataset, perturbation, grid[0])["accuracy"]
        curve, boundary = [], None
        for severity in grid[1:]:
            accuracy = self._scan(network, loader, dataset, perturbation, severity)["accuracy"]
            curve.append([_round(severity), _round(accuracy)])
            if boundary is None and clean - accuracy >= threshold:
                boundary = (severity, accuracy)
        return {
            "model": model,
            "dataset": dataset,
            "checkpoint": checkpoint,
            "perturbation": perturbation,
            "threshold": threshold,
            "clean_accuracy": _round(clean),
            "boundary_severity": _round(boundary[0]) if boundary else None,
            "accuracy_at_boundary": _round(boundary[1]) if boundary else None,
            "boundary_found": boundary is not None,
            "max_scanned_severity": _round(grid[-1]),
            "curve": curve,
            "sample_count": len(loader.dataset),
            "source": self._source(),
        }

    def rank_vulnerability(self, model: str, dataset: str) -> dict:
        paths = self._locate(model, dataset)
        if paths.failure_analysis.is_file():
            analysis = json.loads(paths.failure_analysis.read_text(encoding="utf-8"))
            source = "cached_full_scan"
        else:
            analysis = self._live_analysis(paths)
            source = self._source()
        baseline_accuracy = analysis["baseline"]["accuracy"]
        perturbations = sorted(
            (
                {
                    "perturbation": name,
                    "score": _round(score_vulnerability(experiment, baseline_accuracy)),
                    "obvious_boundary": _round(experiment["boundaries"].get("obvious")),
                    "severe_boundary": _round(experiment["boundaries"].get("severe")),
                }
                for name, experiment in analysis["experiments"].items()
            ),
            key=lambda item: item["score"],
            reverse=True,
        )
        return {
            "model": model,
            "dataset": dataset,
            "weight_mode": paths.weight_mode,
            "clean_accuracy": _round(baseline_accuracy),
            "perturbation_ranking": perturbations,
            "most_vulnerable_classes": _class_ranking(analysis)[:3],
            "sample_count": analysis.get("sample_count"),
            "source": source,
        }

    def apply_targeted_repair(self, model: str, dataset: str, perturbation: str) -> dict:
        if perturbation not in ACTION_SETTINGS:
            raise ValueError(f"未知扰动 '{perturbation}'，可选：{', '.join(ACTION_SETTINGS)}")
        started = time.perf_counter()
        paths = self._locate(model, dataset)
        network, payload = load_checkpoint(paths.baseline_checkpoint, self.device)
        baseline_config = payload["config"]
        weight_mode = baseline_config.get("weight_mode", "scratch")

        # Same augmentation rule as the existing repair agent, restricted to one perturbation.
        experiment = {"boundaries": {}}
        if paths.failure_analysis.is_file():
            analysis = json.loads(paths.failure_analysis.read_text(encoding="utf-8"))
            experiment = analysis["experiments"].get(perturbation, experiment)
        max_severity, severity_source = target_severity(perturbation, experiment)
        probability = ACTION_SETTINGS[perturbation][0]
        plan = {"actions": [{
            "perturbation": perturbation,
            "probability": probability,
            "max_severity": max_severity,
        }]}

        # Measure before and after on the same test images, clean and at the target severity.
        loader = self._test_loader(dataset, network)
        before = self._before_after(network, loader, dataset, perturbation, max_severity)

        adapter = get_dataset_adapter(dataset)
        mean, std = adapter.normalization
        set_seed(self.settings.seed)
        batch = effective_batch_size(self.repair["batch_size"], weight_mode)
        batch = fit_batch_size(network, batch, self.device, training=True,
                               sample_shape=_sample_shape(dataset))
        train_loader, _ = adapter.repair_loaders(
            self.repair["data_dir"], TargetedAugment(plan, mean, std), batch,
            self.repair["num_workers"], baseline_config.get("val_fraction", 0.1),
            baseline_config.get("seed", 42),
        )
        train_set = train_loader.dataset
        epochs = self.repair["epochs"]
        if self.settings.demo:
            train_set = Subset(train_set, range(min(self.settings.train_samples, len(train_set))))
            epochs = self.settings.repair_epochs
        train_loader = DataLoader(
            train_set, batch_size=batch, shuffle=True,
            generator=torch.Generator().manual_seed(self.settings.seed),
        )
        learning_rate = effective_learning_rate(self.repair["learning_rate"], weight_mode)
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        losses = [
            train_one_epoch(network, train_loader, optimizer, criterion, self.device)
            for _ in range(epochs)
        ]
        after = self._before_after(network, loader, dataset, perturbation, max_severity)

        name = f"{paths.artifact_dir.name}_repaired_{perturbation}.pt"
        checkpoint_path = self.settings.output_dir / "checkpoints" / name
        train_config = TrainConfig(**{
            **{k: v for k, v in baseline_config.items() if k in TrainConfig.__dataclass_fields__},
            "epochs": epochs, "learning_rate": learning_rate, "batch_size": batch,
        })
        save_checkpoint(checkpoint_path, network, train_config, after["clean"],
                        model_id=model, dataset_id=dataset)
        self.repaired[(model, dataset)] = (checkpoint_path, perturbation)
        return {
            "model": model,
            "dataset": dataset,
            "perturbation": perturbation,
            "augmentation": {
                "probability": probability,
                "max_severity": _round(max_severity),
                "severity_source": severity_source,
            },
            "train_samples": len(train_set),
            "epochs": epochs,
            "final_train_loss": _round(losses[-1]),
            "evaluated_at_severity": _round(max_severity),
            "before": {k: {m: _round(v) for m, v in d.items()} for k, d in before.items()},
            "after": {k: {m: _round(v) for m, v in d.items()} for k, d in after.items()},
            "sample_count": len(loader.dataset),
            "source": self._source(),
            "seconds": round(time.perf_counter() - started, 1),
            "note": "修复后的模型已保存，可用 checkpoint='repaired' 复测。",
        }

    def _before_after(self, network, loader, dataset, perturbation, severity) -> dict:
        clean_setting = 1.0 if perturbation == "brightness" else 0.0
        result = {}
        for key, value in (("clean", clean_setting), ("perturbed", severity)):
            metrics = self._scan(network, loader, dataset, perturbation, value)
            result[key] = {"accuracy": metrics["accuracy"], "ece": metrics["ece"]}
        return result

    def _live_analysis(self, paths: ExperimentPaths) -> dict:
        """Scan every perturbation when no cached analysis exists (slow, but real)."""
        network = self._load(paths, "baseline")
        loader = self._test_loader(paths.dataset_id, network)
        calibration = load_calibration(paths.dataset_id)
        baseline = self._scan(network, loader, paths.dataset_id, "rotation", 0.0)
        experiments = {}
        for name, grid in severities_for(self.search, calibration).items():
            results = []
            for severity in grid:
                metrics = self._scan(network, loader, paths.dataset_id, name, severity)
                results.append({
                    "severity": severity,
                    "strength": calibration.strength(name, severity),
                    **metrics,
                })
            drops = {"obvious": 0.10, "severe": 0.20}
            boundaries = {
                level: next((r["severity"] for r in results
                             if baseline["accuracy"] - r["accuracy"] >= drop), None)
                for level, drop in drops.items()
            }
            experiments[name] = {"boundaries": boundaries, "results": results}
        return {"baseline": baseline, "experiments": experiments,
                "sample_count": len(loader.dataset)}


def _sample_shape(dataset: str) -> tuple[int, ...]:
    spec = DATASET_REGISTRY.get(dataset)
    return (spec.input_channels, *spec.input_size)


def _class_ranking(analysis: dict) -> list[dict]:
    """Mean per-class accuracy drop over every scanned perturbed setting."""
    clean = analysis["baseline"].get("per_class_accuracy", {})
    drops: dict[str, list[float]] = {name: [] for name in clean}
    for experiment in analysis["experiments"].values():
        for result in experiment["results"][1:]:
            for name, accuracy in result.get("per_class_accuracy", {}).items():
                if name in drops:
                    drops[name].append(max(0.0, clean[name] - accuracy))
    ranking = [
        {"class": name, "mean_accuracy_drop": _round(sum(values) / len(values))}
        for name, values in drops.items() if values
    ]
    return sorted(ranking, key=lambda item: item["mean_accuracy_drop"], reverse=True)


def available_selections() -> list[tuple[str, str]]:
    """Model/dataset pairs that already have a trained baseline checkpoint."""
    found = []
    for model in MODEL_REGISTRY.values(implemented_only=True):
        for dataset_id in model.datasets:
            for mode in WEIGHT_MODES:
                if not model.supports_weight_mode(mode):
                    continue
                try:
                    paths = experiment_paths(model.id, dataset_id, mode)
                except (KeyError, ValueError, NotImplementedError):
                    continue
                if paths.baseline_checkpoint.is_file():
                    found.append((model.id, dataset_id))
                    break
    return found


# ---------------------------------------------------------------------- schemas
_MODEL = {"type": "string", "description": "模型 id，例如 mobilenet_v3、resnet18、lenet、fashion_cnn"}
_DATASET = {"type": "string", "description": "数据集 id：cifar10、mnist 或 fashion_mnist"}
_PERTURBATION = {"type": "string", "enum": list(PERTURBATION_IDS), "description": "扰动类型"}
_CHECKPOINT = {
    "type": "string",
    "enum": ["baseline", "repaired"],
    "description": "评测哪个模型：baseline 为原始模型，repaired 为本次会话修复后的模型",
}


def _function(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOL_SCHEMAS = [
    _function(
        "evaluate_model",
        "在干净测试集上评测模型，返回基础准确率 accuracy、期望校准误差 ece 和 macro_f1。"
        "修复后可传 checkpoint='repaired' 复测。",
        {"model": _MODEL, "dataset": _DATASET, "checkpoint": _CHECKPOINT},
        ["model", "dataset"],
    ),
    _function(
        "find_failure_boundary",
        "沿严重度网格逐级施加某种扰动，返回准确率相对干净数据下降达到 threshold 时的"
        "严重度（失效边界）以及简化的准确率曲线。边界越小说明越脆弱。",
        {
            "model": _MODEL,
            "dataset": _DATASET,
            "perturbation": _PERTURBATION,
            "threshold": {
                "type": "number",
                "description": "判定失效的准确率下降量，0 到 1 之间，默认 0.10（下降 10 个百分点）",
            },
            "checkpoint": _CHECKPOINT,
        },
        ["model", "dataset", "perturbation"],
    ),
    _function(
        "rank_vulnerability",
        "按脆弱度得分对 5 种扰动排序（得分越高越脆弱），并给出平均准确率下降最大的 3 个类别。",
        {"model": _MODEL, "dataset": _DATASET},
        ["model", "dataset"],
    ),
    _function(
        "apply_targeted_repair",
        "针对一种扰动做定向数据增强微调，返回修复前后在干净数据和该扰动目标严重度下的"
        "准确率与 ECE。耗时较长，只在确实需要修复时调用。",
        {"model": _MODEL, "dataset": _DATASET, "perturbation": _PERTURBATION},
        ["model", "dataset", "perturbation"],
    ),
]
