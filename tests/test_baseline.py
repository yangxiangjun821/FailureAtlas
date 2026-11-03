from pathlib import Path

import torch

from failure_atlas.config import TrainConfig, load_train_config
from failure_atlas.model import FashionCNN
from failure_atlas.training import load_checkpoint, save_checkpoint


def test_model_output_shape() -> None:
    model = FashionCNN()
    outputs = model(torch.zeros(4, 1, 28, 28))
    assert outputs.shape == (4, 10)


def test_config_loads() -> None:
    config = load_train_config("configs/baseline.yaml")
    assert config.epochs == 8
    assert config.device == "auto"


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = FashionCNN()
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, TrainConfig(), {"accuracy": 0.5})
    restored, payload = load_checkpoint(path, torch.device("cpu"))
    assert payload["metrics"]["accuracy"] == 0.5
    assert torch.equal(
        model.state_dict()["classifier.4.weight"],
        restored.state_dict()["classifier.4.weight"],
    )
