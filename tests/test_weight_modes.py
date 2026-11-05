import pytest
import torch

from failure_atlas.config import (
    PRETRAINED_LR_SCALE,
    PRETRAINED_MAX_BATCH,
    check_weight_mode,
    effective_batch_size,
    effective_learning_rate,
)
from failure_atlas.model import build_model


def test_pretrained_fine_tuning_uses_a_smaller_learning_rate() -> None:
    assert effective_learning_rate(1e-3, "scratch") == 1e-3
    assert effective_learning_rate(1e-3, "pretrained") == pytest.approx(1e-3 * PRETRAINED_LR_SCALE)


def test_pretrained_batches_are_capped_unless_requested() -> None:
    assert effective_batch_size(512, "scratch") == 512
    assert effective_batch_size(512, "pretrained") == PRETRAINED_MAX_BATCH
    assert effective_batch_size(32, "pretrained") == 32
    assert effective_batch_size(512, "pretrained", explicit=True) == 512


def test_checkpoint_weight_mode_mismatch_is_rejected() -> None:
    scratch = build_model("mobilenet_v3", weight_mode="scratch")
    pretrained = build_model("mobilenet_v3", weight_mode="pretrained", initialize_weights=False)
    check_weight_mode(scratch, "scratch", "baseline.pt")
    check_weight_mode(pretrained, "pretrained", "baseline.pt")
    with pytest.raises(RuntimeError, match="holds a 'scratch' model"):
        check_weight_mode(scratch, "pretrained", "baseline.pt")
    # Models without the attribute are scratch models.
    check_weight_mode(torch.nn.Linear(1, 1), "scratch", "x.pt")


def test_listing_models_never_downloads_pretrained_weights(monkeypatch) -> None:
    """The catalog, its report and the dashboard only need shapes, not weights."""
    from torch.hub import load_state_dict_from_url

    def forbidden(*args, **kwargs):
        raise AssertionError("pretrained weights must not be downloaded for a listing")

    monkeypatch.setattr("torch.hub.load_state_dict_from_url", forbidden)
    monkeypatch.setattr("torchvision.models._api.load_state_dict_from_url", forbidden)
    assert load_state_dict_from_url is not forbidden

    from failure_atlas import web
    from failure_atlas.catalog_reporting import compatibility_markdown

    web.parameter_count.cache_clear()
    assert "ViT-B/16" in compatibility_markdown()
    models = {item["id"]: item for item in web.build_catalog_payload()["models"]}
    assert models["vit_b16"]["parameters"] > 80_000_000


def test_batch_halving_stops_at_the_first_size_that_fits() -> None:
    from failure_atlas.training import halve_until_fits

    assert halve_until_fits(64, lambda b: b <= 32) == 32
    assert halve_until_fits(512, lambda b: b <= 64) == 64
    assert halve_until_fits(64, lambda b: True) == 64
    assert halve_until_fits(64, lambda b: False) == 4
    assert halve_until_fits(2, lambda b: False) == 2


def test_batch_fitting_is_a_no_op_on_cpu() -> None:
    from failure_atlas.training import fit_batch_size

    model = build_model("lenet")
    assert fit_batch_size(model, 128, torch.device("cpu"), training=True) == 128


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_gpu_batch_probe_leaves_model_and_random_streams_untouched() -> None:
    from failure_atlas.training import fit_batch_size

    device = torch.device("cuda")
    model = build_model("mobilenet_v3", weight_mode="scratch").to(device)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    cpu_state, cuda_state = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
    fit_batch_size(model, 16, device, training=True)
    after = model.state_dict()
    assert all(torch.equal(before[key], after[key]) for key in before), "BatchNorm statistics changed"
    assert torch.equal(cpu_state, torch.get_rng_state())
    assert torch.equal(cuda_state, torch.cuda.get_rng_state(device))
