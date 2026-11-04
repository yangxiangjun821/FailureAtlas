# P0 Baseline Report

This report records verified local baselines for the first model and dataset
catalog. Results from different datasets are not treated as a direct model ranking.

All baselines train for 8 epochs on 90% of the training split (seed 42). The remaining
10% is a validation split that neither the baseline nor the repair trains on; it is
used for per-epoch monitoring and for selecting the repair checkpoint. The test split
is evaluated once per baseline, after training.

| Model | Dataset | Train / validation | Test accuracy | Macro-F1 | Repaired accuracy | Run |
|---|---|---:|---:|---:|---:|---|
| FashionCNN | Fashion-MNIST | 54,000 / 6,000 | 91.94% | 91.89% | 92.20% | `fashion-cnn-calibrated` |
| LeNet | MNIST | 54,000 / 6,000 | 98.85% | 98.84% | 98.77% | `lenet-mnist-calibrated` |
| ResNet-18 | CIFAR-10 | 45,000 / 5,000 | 81.47% | 81.42% | 84.62% | `resnet18-cifar10-calibrated` |

Every combination has a complete failure scan, targeted repair and repair comparison.
Per-selection artifacts live in `artifacts/<model>_<dataset>/` and
`reports/generated/<model>_<dataset>/`; FashionCNN on Fashion-MNIST keeps the original
flat layout.

## How to read these numbers

- The ResNet-18 baseline uses no data augmentation, so 81.47% is well below what the
  architecture reaches on CIFAR-10 with standard crop and flip augmentation.
- The repair fine-tunes for 5 more epochs with targeted augmentation. The accuracy
  gain after repair (for example +3.2 points on ResNet-18) mixes the effect of the
  targeted augmentation with the effect of additional training. A control run with
  extra epochs and no targeted augmentation is needed to separate them.
- Scans use one fixed seed and report no confidence intervals.
- Severities are calibrated per dataset to equal pixel-space RMSE up to a common limit
  (`configs/calibration/`). Equal RMSE is not equal perceptual or semantic damage.

## Reproduce

```powershell
python -m scripts.calibrate_perturbations
python -m failure_atlas.agent run --stage full --model fashion_cnn --dataset fashion_mnist
python -m failure_atlas.agent run --stage full --model lenet --dataset mnist
python -m failure_atlas.agent run --stage full --model resnet18 --dataset cifar10
```

Generated weights and metric JSON files remain under `artifacts/` and are excluded from Git.
