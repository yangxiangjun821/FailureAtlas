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

## Web dashboard

The dashboard is an optional visualization layer, not a requirement for running the
agent or generating reports. A read-only public demo is available at
[failure-atlas-ml-agent.grey-wren-2859.chatgpt.site](https://failure-atlas-ml-agent.grey-wren-2859.chatgpt.site/).

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

The repository ignores virtual environments, package caches, datasets, trained
weights, generated reports, exported model bundles, and deployment archives. Before
the first push, review the files Git will publish:

```powershell
git init
git add .
git status
git commit -m "Add local FailureAtlas agent"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/failure-atlas.git
git push -u origin main
```

Do not commit `.env` files, API keys, downloaded datasets, or trained weights. No
open-source license has been selected yet; add one before accepting outside reuse or
contributions.
