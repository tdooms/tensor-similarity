import math

import pytest
import torch

from bilinear_icl.models import RegressionTransformer


def test_mup_init_scales_match_expected():
    cfg = {
        "D": 4,
        "K": 8,
        "d_model": 256,
        "n_head": 8,
        "n_layers": 4,
        "d_mlp": 512,
    }
    torch.manual_seed(0)
    model = RegressionTransformer(**cfg, init_type="mup")

    d_head = cfg["d_model"] // cfg["n_head"]
    target_qk = 1.0 / math.sqrt(d_head)
    target_mlp_d = 1.0 / math.sqrt(cfg["d_mlp"] * cfg["n_layers"])

    qk_std = float(model.layers[0].attn.q1.weight.std().item())
    mlp_d_std = float(model.layers[0].mlp.d.weight.std().item())

    assert 0.5 * target_qk <= qk_std <= 2.0 * target_qk
    assert 0.5 * target_mlp_d <= mlp_d_std <= 2.0 * target_mlp_d


def test_normal_init_scales_match_legacy_defaults():
    cfg = {
        "D": 4,
        "K": 8,
        "d_model": 256,
        "n_head": 8,
        "n_layers": 2,
        "d_mlp": 256,
    }
    torch.manual_seed(0)
    model = RegressionTransformer(**cfg, init_type="normal")

    assert model.layers[0].attn.q1.weight.std().item() == pytest.approx(0.02, rel=0.35)
    assert model.layers[0].attn.o.weight.std().item() == pytest.approx(0.01, rel=0.35)
    assert model.layers[0].mlp.l.weight.std().item() == pytest.approx(0.02, rel=0.35)
    assert model.layers[0].mlp.d.weight.std().item() == pytest.approx(0.01, rel=0.35)
