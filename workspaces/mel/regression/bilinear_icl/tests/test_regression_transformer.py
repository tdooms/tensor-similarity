import torch
import pytest

from bilinear_icl.data import sample_episodes, to_sequence
from bilinear_icl.models import RegressionTransformer


@pytest.mark.parametrize("attn_type", ["bilinear", "softmax"])
def test_regression_transformer_shape(small_cfg, attn_type):
    model = RegressionTransformer(**small_cfg, attn_type=attn_type)
    xs, ys, _ = sample_episodes(4, small_cfg["K"], small_cfg["D"], 0.05)
    y_hat = model(to_sequence(xs, ys))
    assert y_hat.shape == (4, small_cfg["K"])


def test_prediction_positions_match_readout(small_cfg):
    model = RegressionTransformer(**small_cfg)
    xs, ys, _ = sample_episodes(2, small_cfg["K"], small_cfg["D"], 0.05)
    raw = to_sequence(xs, ys)
    y_hat, dbg = model(raw, return_debug=True)
    manual = model.w_out(dbg["hidden_pre_readout"]).squeeze(-1)[:, model.x_idx]
    assert torch.allclose(y_hat, manual)


def test_causal_pattern_is_upper_zero(small_cfg):
    model = RegressionTransformer(**small_cfg)
    x = torch.randn(2, model.n_ctx, model.d_model)
    _, d = model.layers[0].attn(x, return_debug=True)
    p = d["pattern"]
    upper = torch.triu(torch.ones(model.n_ctx, model.n_ctx), diagonal=1).bool()
    assert torch.allclose(p[..., upper], torch.zeros_like(p[..., upper]))


def test_embed_propagates_grad_to_bos(small_cfg):
    model = RegressionTransformer(**small_cfg)
    xs, ys, _ = sample_episodes(2, small_cfg["K"], small_cfg["D"], 0.125)
    y_hat = model(to_sequence(xs, ys))
    loss = y_hat.sum()
    loss.backward()
    assert model.bos.grad is not None
    assert model.bos.grad.abs().sum().item() > 0.0
