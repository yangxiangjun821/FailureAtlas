import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from failure_atlas.bundle import import_bundle_bytes


class TinyClassifier(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        score = images.mean(dim=(1, 2, 3))
        return torch.stack((1 - score, score), dim=1)


def build_bundle(tmp_path: Path, unsafe: bool = False) -> bytes:
    batch = torch.export.Dim("batch", min=1, max=256)
    exported = torch.export.export(
        TinyClassifier(),
        (torch.zeros(2, 1, 4, 4),),
        dynamic_shapes={"images": {0: batch}},
    )
    model_buffer = io.BytesIO()
    torch.export.save(exported, model_buffer)
    images = np.stack(
        (np.zeros((1, 4, 4), dtype=np.float32), np.ones((1, 4, 4), dtype=np.float32))
    )
    labels = np.asarray([0, 1], dtype=np.int64)
    dataset_buffer = io.BytesIO()
    np.savez_compressed(dataset_buffer, images=images, labels=labels)
    manifest = {
        "schema_version": 1,
        "name": "Tiny model",
        "task": "image_classification",
        "model_file": "model.pt2",
        "dataset_file": "test.npz",
        "input_shape": [-1, 1, 4, 4],
        "input_range": [0.0, 1.0],
        "class_names": ["dark", "light"],
    }
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("model.pt2", model_buffer.getvalue())
        archive.writestr("test.npz", dataset_buffer.getvalue())
        if unsafe:
            archive.writestr("../escape.txt", "blocked")
    return payload.getvalue()


def test_import_bundle_validates_model_and_dataset(tmp_path: Path) -> None:
    result = import_bundle_bytes(build_bundle(tmp_path), tmp_path / "workspace")
    assert result["status"] == "validated"
    assert result["clean_accuracy"] == 1.0
    assert result["sample_count"] == 2


def test_import_bundle_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsafe archive path"):
        import_bundle_bytes(build_bundle(tmp_path, unsafe=True), tmp_path / "workspace")


def test_import_bundle_rejects_missing_files(tmp_path: Path) -> None:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("manifest.json", "{}")
    with pytest.raises(ValueError, match="Bundle is missing"):
        import_bundle_bytes(payload.getvalue(), tmp_path / "workspace")
