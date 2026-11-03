"""Verify the Phase 1 FailureAtlas development environment."""

from __future__ import annotations

import importlib
import platform
import sys


REQUIRED_MODULES = (
    "numpy",
    "pandas",
    "sklearn",
    "scipy",
    "PIL",
    "cv2",
    "matplotlib",
    "seaborn",
    "plotly",
    "optuna",
    "fastapi",
    "uvicorn",
    "pytest",
    "yaml",
    "torch",
    "torchvision",
    "captum",
)


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    missing: list[str] = []
    for name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            missing.append(name)
            continue
        version = getattr(module, "__version__", "installed")
        print(f"{name}: {version}")

    if missing:
        print("Missing modules: " + ", ".join(missing))
        return 1

    import torch

    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("GPU verification failed: CUDA is not available to PyTorch.")
        return 2

    device = torch.device("cuda")
    left = torch.tensor([[1.0, 2.0]], device=device)
    right = torch.tensor([[3.0], [4.0]], device=device)
    result = left @ right
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA tensor test: {result.item():.1f}")
    print("FailureAtlas environment is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

