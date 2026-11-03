# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

FastAPI serves a dependency-free HTML, CSS, and JavaScript dashboard. The existing
PyTorch experiment code remains the source of truth for training and evaluation.

## Users

Machine-learning students and junior engineers who need to inspect, demonstrate,
and improve model robustness from one local workspace.

## Product Purpose

FailureAtlas discovers where a trained model fails under controlled distribution
shifts, turns the evidence into a targeted repair plan, retrains the model, and
measures whether the failure boundaries actually moved.

## Positioning

The product closes the loop from failure discovery to evidence-based repair instead
of stopping at aggregate accuracy or generic model monitoring.

## Operating Context

The current workflow uses a Fashion-MNIST CNN, five controlled perturbations,
locally cached datasets, GPU training, JSON experiment artifacts, and Markdown
reports. It is designed to run locally and remain useful without external APIs.

## Capabilities and Constraints

- Compare baseline and repaired models across rotation, noise, blur, occlusion, and brightness.
- Expose mild, obvious, and severe failure boundaries.
- Show clean-accuracy guardrails and repair tradeoffs.
- Trigger local rescan and repair jobs from the dashboard.
- The built-in analysis uses Fashion-MNIST; schema-v1 `.fatlas` bundles can import
  PyTorch Export image classifiers and `.npz` test sets for validation.

## Planned Experiments

- Build a model catalog and website selector covering representative image,
  sequence, and tabular architectures. Every model must run through the same
  failure-boundary protocol so comparisons remain meaningful.
- Build a versioned dataset library. Start with MNIST, Fashion-MNIST, CIFAR-10,
  CIFAR-100, and SVHN, then add text and tabular datasets after task-specific
  perturbation adapters are defined.

## Evidence on Hand

- `reports/generated/failure_analysis.json`: complete baseline scan.
- `reports/generated/repair_comparison.json`: baseline-versus-repaired evidence.
- `artifacts/repair_plan.json`: machine-selected repair policy.
- No customer claims, testimonials, or production deployment evidence exist and none should be fabricated.

## Product Principles

- Show evidence before recommendations.
- Preserve clean-data quality while improving robustness.
- Make tradeoffs visible, including regressions.
- Keep experiments deterministic and locally reproducible.
