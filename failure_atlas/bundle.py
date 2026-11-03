"""Portable FailureAtlas model bundles and safe local validation."""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import torch


MAX_BUNDLE_BYTES = 100 * 1024 * 1024
MAX_EXTRACTED_BYTES = 300 * 1024 * 1024
REQUIRED_FILES = {"manifest.json", "model.pt2", "test.npz"}


@dataclass(frozen=True)
class BundleManifest:
    schema_version: int
    name: str
    task: str
    model_file: str
    dataset_file: str
    input_shape: list[int]
    input_range: list[float]
    class_names: list[str]

    @classmethod
    def from_dict(cls, raw: dict) -> "BundleManifest":
        required = {
            "schema_version", "name", "task", "model_file", "dataset_file",
            "input_shape", "input_range", "class_names",
        }
        missing = required - set(raw)
        if missing:
            raise ValueError(f"Manifest is missing: {', '.join(sorted(missing))}")
        manifest = cls(**{key: raw[key] for key in required})
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Only bundle schema_version 1 is supported.")
        if self.task != "image_classification":
            raise ValueError("Only image_classification bundles are supported in this phase.")
        if not self.name.strip() or len(self.name) > 80:
            raise ValueError("Bundle name must contain 1–80 characters.")
        if self.model_file != "model.pt2" or self.dataset_file != "test.npz":
            raise ValueError("Schema v1 requires model.pt2 and test.npz at the bundle root.")
        if len(self.input_shape) != 4 or self.input_shape[0] not in (-1, 0):
            raise ValueError("input_shape must be [-1, channels, height, width].")
        if any(not isinstance(value, int) or value <= 0 for value in self.input_shape[1:]):
            raise ValueError("Channels, height, and width must be positive integers.")
        if len(self.input_range) != 2 or self.input_range[0] >= self.input_range[1]:
            raise ValueError("input_range must contain an increasing [min, max] pair.")
        if len(self.class_names) < 2 or len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names must contain at least two unique labels.")


def safe_archive_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    files = []
    extracted_size = 0
    for member in archive.infolist():
        path = PurePosixPath(member.filename.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe archive path: {member.filename}")
        if member.is_dir():
            continue
        if len(path.parts) != 1:
            raise ValueError("Schema v1 bundle files must be stored at the archive root.")
        extracted_size += member.file_size
        if extracted_size > MAX_EXTRACTED_BYTES:
            raise ValueError("Extracted bundle exceeds the 300 MB safety limit.")
        files.append(member)
    names = {PurePosixPath(item.filename).name for item in files}
    missing = REQUIRED_FILES - names
    if missing:
        raise ValueError(f"Bundle is missing: {', '.join(sorted(missing))}")
    return files


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:40] or "model"


@torch.inference_mode()
def validate_extracted_bundle(directory: Path) -> dict:
    manifest = BundleManifest.from_dict(
        json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    )
    with np.load(directory / manifest.dataset_file, allow_pickle=False) as dataset:
        if set(dataset.files) != {"images", "labels"}:
            raise ValueError("test.npz must contain exactly 'images' and 'labels'.")
        images = dataset["images"]
        labels = dataset["labels"]
        expected_shape = tuple(manifest.input_shape[1:])
        if images.ndim != 4 or tuple(images.shape[1:]) != expected_shape:
            raise ValueError(f"Expected images shaped [N, {', '.join(map(str, expected_shape))}].")
        if labels.ndim != 1 or len(labels) != len(images) or len(images) == 0:
            raise ValueError("labels must be a non-empty 1D array matching images.")
        if len(images) > 100_000:
            raise ValueError("Schema v1 accepts at most 100,000 test samples.")
        if not np.issubdtype(images.dtype, np.number) or not np.isfinite(images).all():
            raise ValueError("Images must contain finite numeric values.")
        if images.min() < manifest.input_range[0] or images.max() > manifest.input_range[1]:
            raise ValueError("Image values fall outside the declared input_range.")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Labels must use an integer dtype.")
        if labels.min() < 0 or labels.max() >= len(manifest.class_names):
            raise ValueError("Labels fall outside the declared class_names range.")

        model_bytes = (directory / manifest.model_file).read_bytes()
        model = torch.export.load(io.BytesIO(model_bytes)).module()
        correct = 0
        processed = 0
        for start in range(0, len(images), 256):
            batch = torch.from_numpy(images[start : start + 256]).float()
            logits = model(batch)
            if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
                raise ValueError("The model must return a [batch, classes] tensor.")
            if logits.shape != (len(batch), len(manifest.class_names)):
                raise ValueError("Model output shape does not match class_names.")
            if not torch.isfinite(logits).all():
                raise ValueError("Model output contains non-finite values.")
            expected = torch.from_numpy(labels[start : start + 256]).long()
            correct += int(logits.argmax(dim=1).eq(expected).sum())
            processed += len(batch)

    return {
        "name": manifest.name,
        "task": manifest.task,
        "sample_count": processed,
        "class_count": len(manifest.class_names),
        "class_names": manifest.class_names,
        "input_shape": manifest.input_shape,
        "input_range": manifest.input_range,
        "format": "pytorch_export",
        "clean_accuracy": correct / processed,
        "model_bytes": len(model_bytes),
        "dataset_bytes": (directory / manifest.dataset_file).stat().st_size,
        "status": "validated",
    }


def import_bundle_bytes(data: bytes, workspace_root: Path) -> dict:
    if not data:
        raise ValueError("The uploaded bundle is empty.")
    if len(data) > MAX_BUNDLE_BYTES:
        raise ValueError("Bundle exceeds the 100 MB upload limit.")
    digest = hashlib.sha256(data).hexdigest()[:10]
    workspace_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = safe_archive_members(archive)
        manifest = BundleManifest.from_dict(
            json.loads(archive.read("manifest.json").decode("utf-8"))
        )
        project_id = f"{slugify(manifest.name)}-{digest}"
        destination = workspace_root / project_id
        if destination.exists():
            result = validate_extracted_bundle(destination)
            metadata = {"id": project_id, **result, "duplicate": True}
            (destination / "validation.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            return metadata
        with tempfile.TemporaryDirectory(dir=workspace_root) as temporary:
            temporary_path = Path(temporary)
            for member in members:
                target = temporary_path / PurePosixPath(member.filename).name
                with archive.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
            result = validate_extracted_bundle(temporary_path)
            temporary_path.rename(destination)
    metadata = {"id": project_id, **result, "duplicate": False}
    (destination / "validation.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def list_imported_projects(workspace_root: Path) -> list[dict]:
    if not workspace_root.exists():
        return []
    projects = []
    for path in sorted(workspace_root.glob("*/validation.json")):
        try:
            project = json.loads(path.read_text(encoding="utf-8"))
            if project.get("format") == "pytorch_export":
                projects.append(project)
        except (OSError, json.JSONDecodeError):
            continue
    return projects
