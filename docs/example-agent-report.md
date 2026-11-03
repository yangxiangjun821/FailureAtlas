# Example FailureAtlas Agent Report

This is a committed snapshot of a real local reference run. Generated reports stay
under `reports/generated/` and are ignored by Git so repeated experiments do not
pollute the repository history.

## Experiment identity

- Model: FashionCNN
- Dataset: Fashion-MNIST test split
- Test samples: 10,000
- Clean accuracy: 91.90%
- Macro-F1: 91.86%
- Expected calibration error: 0.0191

## Failure boundary summary

| Perturbation | Mild | Obvious | Severe | Max-severity accuracy |
|---|---:|---:|---:|---:|
| Rotation | 5 | 10 | 15 | 19.29% |
| Gaussian noise | 0.1 | 0.15 | 0.2 | 13.15% |
| Gaussian blur | 0.6 | 1 | 1 | 37.72% |
| Center occlusion | 0.05 | 0.05 | 0.3 | 53.30% |
| Brightness reduction | 0.6 | 0.3 | not reached | 76.96% |

## Agent diagnosis

1. Rotation — vulnerability score 0.524. Improve orientation invariance before the severe boundary.
2. Gaussian noise — vulnerability score 0.515. Train with noise up to the observed severe boundary.
3. Gaussian blur — vulnerability score 0.356. Preserve features under the observed blur boundary.

## Repair verification

- Repaired clean accuracy: 92.28%
- Clean-accuracy change: +0.38%

| Perturbation | Severe boundary before → after | Endpoint change |
|---|---:|---:|
| Rotation | 15 → 30 (+15) | -1.04% |
| Gaussian noise | 0.2 → 0.4 (+0.2) | +11.80% |
| Gaussian blur | 1 → 2 (+1) | +14.21% |
| Center occlusion | 0.3 → 0.3 (+0) | -2.69% |
| Brightness reduction | not reached → not reached | -1.84% |

The result shows why FailureAtlas reports tradeoffs instead of claiming universal
improvement: the repair substantially improves noise and blur robustness, moves the
rotation boundary, and also exposes small regressions at some endpoints.
