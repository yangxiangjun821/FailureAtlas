"""Measure pixel-space distortion per perturbation and store the calibration."""

from __future__ import annotations

import argparse

from failure_atlas.calibration import STRENGTH_LEVELS, calibrate, save_calibration
from failure_atlas.catalog import DATASET_REGISTRY
from failure_atlas.data import load_unlabeled_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = args.datasets or [spec.id for spec in DATASET_REGISTRY.values(implemented_only=True)]
    for dataset_id in datasets:
        images = load_unlabeled_sample(args.data_dir, args.samples, dataset_id)
        calibration = calibrate(dataset_id, images, seed=args.seed)
        path = save_calibration(calibration)
        print(f"{dataset_id}: {len(images)} images, reference RMSE {calibration.reference_rmse:.4f}")
        for name in calibration.tables:
            grid = calibration.grid(name)
            print(f"  {name:15} max RMSE {calibration.max_rmse(name):.3f}  "
                  f"strength grid ends at {grid[-1]:g} ({len(STRENGTH_LEVELS)} levels)")
        print(f"  saved {path}")


if __name__ == "__main__":
    main()
