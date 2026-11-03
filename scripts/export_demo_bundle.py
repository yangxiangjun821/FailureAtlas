"""Export the current FashionCNN as an example portable .fatlas bundle."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms

from failure_atlas.data import FASHION_MNIST_CLASSES, NORMALIZE_MEAN, NORMALIZE_STD
from failure_atlas.training import load_checkpoint


class PortableFashionCNN(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(NORMALIZE_MEAN).view(1, 1, 1, 1))
        self.register_buffer("std", torch.tensor(NORMALIZE_STD).view(1, 1, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model((images - self.mean) / self.std)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="exports/fashion-mnist-demo.fatlas")
    parser.add_argument("--checkpoint", default="artifacts/repaired_fashion_cnn.pt")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model, _ = load_checkpoint(args.checkpoint, torch.device("cpu"))
    batch = torch.export.Dim("batch", min=1, max=256)
    exported = torch.export.export(
        PortableFashionCNN(model).eval(),
        (torch.zeros(2, 1, 28, 28),),
        dynamic_shapes={"images": {0: batch}},
    )
    dataset = datasets.FashionMNIST("data", train=False, transform=transforms.ToTensor(), download=False)
    images = np.stack([dataset[index][0].numpy() for index in range(len(dataset))])
    labels = np.asarray([dataset[index][1] for index in range(len(dataset))], dtype=np.int64)
    manifest = {
        "schema_version": 1,
        "name": "Fashion-MNIST repaired CNN",
        "task": "image_classification",
        "model_file": "model.pt2",
        "dataset_file": "test.npz",
        "input_shape": [-1, 1, 28, 28],
        "input_range": [0.0, 1.0],
        "class_names": list(FASHION_MNIST_CLASSES),
    }
    model_buffer = io.BytesIO()
    torch.export.save(exported, model_buffer)
    dataset_buffer = io.BytesIO()
    np.savez_compressed(dataset_buffer, images=images, labels=labels)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        archive.writestr("model.pt2", model_buffer.getvalue())
        archive.writestr("test.npz", dataset_buffer.getvalue())
    print(f"Exported portable bundle: {output} ({output.stat().st_size / 1_048_576:.1f} MB)")


if __name__ == "__main__":
    main()
