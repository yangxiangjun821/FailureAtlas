import pytest
import torch

from failure_atlas.data import split_indices
from failure_atlas.model import FashionCNN
from failure_atlas.training import load_checkpoint, save_checkpoint
from failure_atlas.config import TrainConfig


def test_split_is_disjoint_deterministic_and_complete() -> None:
    train, val = split_indices(1000, 0.1, seed=7)
    assert len(val) == 100
    assert set(train).isdisjoint(val)
    assert sorted(train + val) == list(range(1000))
    assert (train, val) == split_indices(1000, 0.1, seed=7)


def test_checkpoint_rejects_unknown_model_class(tmp_path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"model_class": "Mystery", "model_state_dict": {}}, path)
    with pytest.raises(ValueError, match="Unknown model_class"):
        load_checkpoint(path, torch.device("cpu"))


def test_checkpoint_round_trip_with_restricted_loader(tmp_path) -> None:
    path = tmp_path / "ok.pt"
    save_checkpoint(path, FashionCNN(), TrainConfig(), {"accuracy": 0.5})
    model, payload = load_checkpoint(path, torch.device("cpu"))
    assert type(model).__name__ == "FashionCNN"
    assert payload["metrics"]["accuracy"] == 0.5


def test_holdout_and_repair_loaders_share_one_split(tmp_path, monkeypatch) -> None:
    from torch.utils.data import Subset

    from failure_atlas import data

    class Tiny:
        def __init__(self, root, train=True, transform=None, download=False):
            self.items = list(range(200))

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return torch.zeros(1, 4, 4), index % 10

    monkeypatch.setitem(data.DATASET_PROFILES["fashion_mnist"], "class", Tiny)
    base_train, base_val = data.build_holdout_loaders(tmp_path, 8, 0, 0.1, 42)
    fix_train, fix_val = data.build_repair_loaders(tmp_path, lambda x: x, 8, 0, 0.1, 42)
    assert isinstance(base_val.dataset, Subset)
    assert base_val.dataset.indices == fix_val.dataset.indices
    assert base_train.dataset.indices == fix_train.dataset.indices
    assert set(base_train.dataset.indices).isdisjoint(base_val.dataset.indices)
