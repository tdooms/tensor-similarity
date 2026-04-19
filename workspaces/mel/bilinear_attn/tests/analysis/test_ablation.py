"""Tests for position-ablated loss computation."""
import pytest
import torch

from analysis.behaviour.ablation import compute_ablated_loss, compute_val_loss, ablate_rotary
from analysis.behaviour.tracker import TrackerConfig
from tests.analysis.conftest import V, save_fake_checkpoint


def test_ablate_rotary_is_identity(model):
    """``ablate_rotary`` must turn Rotary modules into pure identity for their
    duration, and restore the original behaviour on exit."""
    from models.attention_kernels.rotary import Rotary

    rotary_modules = [m for m in model.modules() if isinstance(m, Rotary)]
    assert rotary_modules, "Model should have Rotary modules"

    x_4d = torch.randn(1, 8, rotary_modules[0].cos_cached.shape[-1]).unsqueeze(2)
    normal_out = rotary_modules[0](x_4d)
    assert not torch.allclose(x_4d, normal_out)

    with ablate_rotary(model):
        assert torch.allclose(x_4d, rotary_modules[0](x_4d))

    assert torch.allclose(normal_out, rotary_modules[0](x_4d))


def test_forced_rotary_changes_logits(tiny_config):
    """Patching Rotary with reversed frequencies must change logits, then
    restore cleanly when the patch is reverted."""
    from models import AttentionLM
    from models.attention_kernels.rotary import Rotary

    cfg = {
        **tiny_config,
        "model": {**tiny_config["model"], "attn_type": "softmax"},
        "init": {"std_embed": 0.5, "std_qkv": 0.5, "std_o": 0.5},
    }
    torch.manual_seed(42)
    sm_model = AttentionLM.from_config(cfg).eval()
    x = torch.randint(0, V, (1, 8))

    with torch.no_grad():
        normal_logits = sm_model(x).clone()

    originals = {}
    for name, m in sm_model.named_modules():
        if isinstance(m, Rotary):
            originals[name] = m.forward
            flipped_cos = m.cos_cached.flip(-1)
            flipped_sin = m.sin_cached.flip(-1)

            def _reversed_rotary(x, _cos=flipped_cos, _sin=flipped_sin):
                seq_len = x.size(1)
                a, b = x.chunk(2, dim=-1)
                y = torch.cat((-b, a), dim=-1)
                return (x * _cos[:, :seq_len]) + (y * _sin[:, :seq_len])

            m.forward = _reversed_rotary

    with torch.no_grad():
        altered_logits = sm_model(x)
    assert (normal_logits - altered_logits).abs().max().item() > 1e-2

    for name, m in sm_model.named_modules():
        if name in originals:
            m.forward = originals[name]

    with torch.no_grad():
        restored_logits = sm_model(x)
    assert torch.allclose(normal_logits, restored_logits, atol=1e-6)


@pytest.mark.parametrize("fn", [compute_val_loss, compute_ablated_loss])
def test_loss_fn_returns_finite_float(fn, model, dummy_dataloader):
    loss = fn(model, dummy_dataloader, device="cpu", max_batches=5)
    assert isinstance(loss, float) and loss > 0
    assert not torch.isnan(torch.tensor(loss))


def _ablation_config():
    return TrackerConfig(
        bigram_enabled=False, ngram_enabled=False,
        ablation_enabled=True, ablation_compute_every=10,
        ablation_max_val_batches=5,
    )


def test_tracker_compute_metrics_includes_ablation(make_tracker):
    t = make_tracker(config=_ablation_config())
    t._is_fitted = True  # no bigram/ngram to fit

    metrics = t.compute_metrics(step=10)
    assert {"val_loss", "ablated_loss", "ablation_gap"} <= metrics.keys()
    assert isinstance(metrics["ablated_loss"], float)
    assert metrics["ablation_gap"] == pytest.approx(
        metrics["ablated_loss"] - metrics["val_loss"]
    )


def test_tracker_should_compute_ablation(make_tracker):
    t = make_tracker(config=TrackerConfig(
        bigram_enabled=False, ngram_enabled=False,
        ablation_enabled=True, ablation_compute_every=200,
    ))
    assert t.should_compute(200)
    assert t.should_compute(400)
    assert not t.should_compute(50)


def test_evaluate_checkpoint_includes_ablation(make_tracker, model, tmp_path):
    ckpt = tmp_path / "step_100.pt"
    save_fake_checkpoint(model, ckpt, step=100)
    t = make_tracker(config=_ablation_config())
    t._is_fitted = True

    metrics = t.evaluate_checkpoint(ckpt)
    assert metrics["step"] == 100
    assert {"val_loss", "ablated_loss", "ablation_gap"} <= metrics.keys()
