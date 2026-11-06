# FailureAtlas

FailureAtlas is a local AI/ML agent that discovers where a trained model starts to
fail under controlled distribution shifts, selects targeted repairs, verifies the
result, and generates an evidence-backed Markdown report. It requires no LLM API.

## What makes it different

Instead of reporting only a single validation accuracy, FailureAtlas searches for
the boundary where model behavior becomes unreliable. It then ranks vulnerabilities,
selects a targeted augmentation policy, repairs the model, and measures whether the
failure boundary moved. The included Fashion-MNIST workflow demonstrates the full
closed loop with rotation, Gaussian noise, blur, occlusion, and brightness shifts.

## Quick start

Requirements: Python 3.12 and Git. A CUDA-capable GPU is optional.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-base.txt
```

For the pinned CUDA 13.0 PyTorch build:

```powershell
python -m pip install -r requirements-gpu.txt
```

Verify the environment and run the test suite:

```powershell
python verify_environment.py
python -m pytest
```

## Run the local agent

List the model, dataset, and perturbation catalog, then validate a combination:

```powershell
python -m failure_atlas.agent catalog
python -m failure_atlas.agent validate --model fashion_cnn --dataset fashion_mnist
```

The catalog includes FashionCNN, LeNet, ResNet-18, MobileNetV3-Small,
EfficientNet-B0, ConvNeXt-Tiny, ViT-CIFAR and ViT-B/16 with Fashion-MNIST, MNIST,
and CIFAR-10. MobileNetV3-Small, EfficientNet-B0 and ConvNeXt-Tiny support both
from-scratch CIFAR-10 training and official TorchVision ImageNet weights. ViT-CIFAR is
a compact transformer and trains from scratch only; ViT-B/16 is the official ImageNet
network and exists only with pretrained weights, so one model id always names one
architecture. See the generated
[compatibility matrix](docs/compatibility-matrix.md) and
[baseline report](docs/p0-baseline-report.md). The remaining Phase E verification
steps are tracked in the [modern-model checklist](docs/stage-e-checklist.md).

Train a selected baseline without changing Python code:

```powershell
python -m scripts.download_data --model lenet --dataset mnist
python -m scripts.train_baseline --model lenet --dataset mnist --epochs 1
```

Select official pretrained weights for a modern CIFAR-10 model with either the
Dashboard's **Initial weights** field or `--weight-mode pretrained`:

```powershell
python -m failure_atlas.agent run --stage baseline --model mobilenet_v3 --dataset cifar10 --weight-mode pretrained --baseline-epochs 1 --run-id mobile-pretrained
```

The first explicit pretrained run downloads the matching official checkpoint to
the Torch hub cache (set `TORCH_HOME` to put it elsewhere, for example when
`~/.cache` is not writable). The Dashboard preflight shows whether that file is already
cached; later runs and checkpoint restoration work offline. The run records the
weight enum, source URL, cache path, and preprocessing contract.

Pretrained runs keep their own artifacts in `artifacts/<model>_<dataset>_pretrained/`
and `reports/generated/<model>_<dataset>_pretrained/`, so a scratch baseline and a
pretrained one for the same model sit side by side. Because the pretrained networks see
upsampled 224×224 inputs and start from useful features, fine-tuning uses a learning
rate one tenth of the configured one and caps the batch at 64 unless `--batch-size` is
given.

Inspect the planned workflow without changing any files:

```powershell
python -m failure_atlas.agent plan --stage full
```

Run only data preparation, baseline training, and failure analysis:

```powershell
python -m failure_atlas.agent run --stage analyze
```

Run the complete diagnosis → repair → verification loop:

```powershell
python -m failure_atlas.agent run --stage full --run-id my-first-study
```

The agent resumes from existing artifacts by default. Use `--force` to repeat every
step, `--dry-run` to preview execution, or `--baseline-epochs 1` for a quick smoke
test. If the experiments already exist, regenerate only the report with:

```powershell
python -m failure_atlas.agent report
```

Outputs are isolated per selection: FashionCNN on Fashion-MNIST keeps the original
flat layout (`artifacts/`, `reports/generated/`), and every other combination writes to
`artifacts/<model>_<dataset>/` and `reports/generated/<model>_<dataset>/`. For example:

```powershell
python -m failure_atlas.agent run --stage full --model lenet --dataset mnist
```

Every real run is archived under `runs/<run_id>/` with its configuration,
environment metadata, metrics, failure analysis, repair evidence, plots, Markdown
report, and machine-readable manifest. The `runs/` directory is local-only and
ignored by Git. See the committed [example agent report](docs/example-agent-report.md)
for a reference run.

## Manual experiment commands

The individual pipeline stages remain available for study and debugging:

```powershell
.\.venv\Scripts\Activate.ps1
python -m scripts.download_data
python -m pytest
python -m scripts.train_baseline
```

For a fast end-to-end smoke test:

```powershell
python -m scripts.train_baseline --epochs 1
```

Outputs are written to `artifacts/` and the dataset is cached under `data/`, both of
which are excluded from Git.

## Discover failure boundaries

Run the five controlled distribution-shift experiments after training the baseline:

```powershell
python -m scripts.search_failures
```

The command scans rotation, Gaussian noise, blur, center occlusion, and brightness.
It writes a JSON result, accuracy curves, and an agent diagnosis to
`reports/generated/`.

## Automated repair loop

The repair agent ranks the observed vulnerabilities, selects targeted augmentation,
fine-tunes the baseline, and then repeats the same robustness scan:

```powershell
python -m scripts.repair_model
python -m scripts.compare_repair
```

The final evidence report is written to `reports/generated/repair_report.md`.

## LLM agent (tool calling)

`agent/` adds a small LLM-driven loop next to the rule-based agent above: the model
reads tool results and decides the next call (observe → decide → call → verify). It
uses any OpenAI-compatible `/chat/completions` endpoint with tool calling, through a
standard-library HTTP client, so it needs no extra package.

| Tool | What it returns |
| --- | --- |
| `evaluate_model` | clean accuracy, ECE, macro-F1 (`checkpoint="repaired"` re-tests) |
| `find_failure_boundary` | first severity whose accuracy drop ≥ `threshold`, plus the curve |
| `rank_vulnerability` | perturbations by vulnerability score, the 3 weakest classes |
| `apply_targeted_repair` | single-perturbation augmentation fine-tune, before/after metrics |

Configure the endpoint with three environment variables and run from the repository
root:

```powershell
$env:BASE_URL = "https://api.openai.com/v1"   # or http://localhost:11434/v1 for Ollama
$env:API_KEY  = "<your key>"                  # Ollama ignores it
$env:MODEL    = "gpt-4o-mini"                 # or e.g. qwen2.5:7b on Ollama
python -m agent.agent "帮我找出 MobileNetV3 在 CIFAR-10 上最脆弱的扰动并修复"
```

or start the dashboard (`python -m scripts.serve_dashboard`) and call
`POST /agent/run` with `{"goal": "...", "max_steps": 8}`; it returns the summary and
the full trajectory. Every step is written to `agent_runs/agent_trace.json`.

Demo mode (the default) keeps a run within a few minutes on an RTX 3060: evaluation
and boundary scans use a fixed 1,000-image test subset, the ranking reuses the cached
full-test-set scan in `reports/generated/`, and the repair fine-tunes one epoch on
2,000 training images. Each tool result names its `source` and `sample_count`. Pass
`--full` to use the complete splits. The agent tools need a trained baseline
checkpoint and never overwrite `artifacts/`; repaired checkpoints go to
`agent_runs/checkpoints/`. Tests use a scripted fake LLM:
`python -m pytest tests/test_llm_agent.py`.

### Example trace

A real run on 2026-11-06 with `deepseek-chat`, goal
「帮我找出 MobileNetV3 在 CIFAR-10 上最脆弱的扰动并修复」, demo mode on an RTX 3060.
Five LLM calls, 34 seconds end to end, `status: finished`,
`verified_after_repair: true`. Condensed from `agent_runs/agent_trace.json`:

| Step | Model's decision | Tool call | Key result |
| --- | --- | --- | --- |
| 1 | Evaluate, then rank | `evaluate_model` | accuracy 0.929, ECE 0.0429 (1,000-image subset) |
| 1 | | `rank_vulnerability` | gaussian_noise 0.5528 > blur 0.5261 > occlusion 0.1165 > rotation 0.0845 > brightness 0.0256 (cached 10k scan); weakest classes horse, automobile, dog |
| 2 | Confirm the top perturbation | `find_failure_boundary(gaussian_noise, 0.10)` | 10-point drop at σ = 0.0309 (accuracy 0.818) |
| 3 | Early failure → repair | `apply_targeted_repair(gaussian_noise)` | at σ = 0.0466: accuracy 0.675 → 0.771, ECE 0.2108 → 0.140; clean 0.929 → 0.938 |
| 4 | Must re-test | `find_failure_boundary(..., checkpoint="repaired")` | boundary σ 0.0309 → 0.0466; at σ = 0.0309 accuracy 0.818 → 0.856 |
| 5 | Final answer | — | Chinese summary with sources, sample counts and limitations |

The final summary cited every number correctly from the tool results, with one
wording slip: it called "0.818 → 0.771" a same-severity improvement, although those
two accuracies are at different severities (0.0309 and 0.0466). It also noted its own
limits: only Gaussian noise was repaired, blur (score 0.5261) was not re-tested, and
the repair is a light 1-epoch, 2,000-image fine-tune. The full run, with every tool
result, both accuracy curves and the model's verbatim summary, is in the
[LLM agent run report](docs/llm-agent-run-report.md).

## Web dashboard

The dashboard is an optional visualization layer, not a requirement for running the
agent or generating reports. Setting `FAILUREATLAS_DEMO=1` starts a read-only
showcase that serves `deploy/dashboard.json`; `render.yaml` deploys it that way. Rebuild
the snapshot with `python -m scripts.build_deploy_snapshot` after new experiments.

Start the local control console after the experiment artifacts exist:

```powershell
python -m scripts.serve_dashboard
```

Then open `http://127.0.0.1:8000/`. The console visualizes all five failure curves,
the repair policy, boundary movements, experiment logs, and local rescan/repair jobs.

## Portable model bundles

Export the current repaired model as a reference `.fatlas` bundle:

```powershell
python -m scripts.export_demo_bundle
```

A schema-v1 bundle is a ZIP archive containing `manifest.json`, a portable
PyTorch Export `model.pt2`, and a `test.npz` with `images` and `labels`. Import it from
the dashboard. FailureAtlas validates archive paths, sizes, input shape and range,
class count, model outputs, and clean accuracy before adding it to the workspace.

## Optional web deployment

The repository includes a root-level `render.yaml` and lightweight
`requirements-deploy.txt` for deploying the read-only dashboard from GitHub.
Build the showcase snapshot before pushing:

```powershell
python -m scripts.build_deploy_snapshot
```

Render runs with `FAILUREATLAS_DEMO=1`, serving the real experiment results without
installing PyTorch. Training, uploads, and repair jobs remain available only in the
local GPU version.

To rebuild the fully static Sites package locally, first ensure
`reports/generated/repair_report.md` exists and then run:

```powershell
python -m scripts.build_static_site
```

The generated `sites/failure-atlas-showcase/` directory is intentionally ignored by
Git because the hosted Sites workflow maintains its own internal repository.

## GitHub publishing checklist

This repository is published at
[github.com/yangxiangjun821/FailureAtlas](https://github.com/yangxiangjun821/FailureAtlas).
It ignores virtual environments, package caches, datasets, trained weights, generated
reports, exported model bundles, and deployment archives. The tracked files are source,
configuration, calibration tables, documentation, and the small
`deploy/dashboard.json` snapshot that `render.yaml` serves in demo mode.

Before publishing a copy or fork elsewhere, check what Git will publish and which
author identities it exposes:

```powershell
git status                      # should be clean
git log --format="%an <%ae>"    # authors that become public
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

Use your GitHub `noreply` address as the commit email if you want to keep a personal
address private. Do not commit `.env` files, API keys, downloaded datasets, or trained
weights.

## License

Released under the [MIT License](LICENSE).
