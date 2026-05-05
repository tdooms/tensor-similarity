import itertools

import torch

from bilinear_icl.data import sample_episodes, to_sequence
from bilinear_icl.models import RegressionTransformer


def test_norm_places_combinations_forward_finite(small_cfg):
    places = ["pre_attn", "pre_mlp", "pre_unembed"]
    xs, ys, _ = sample_episodes(2, small_cfg["K"], small_cfg["D"], 0.05)
    raw = to_sequence(xs, ys)

    for r in range(len(places) + 1):
        for combo in itertools.combinations(places, r):
            model = RegressionTransformer(
                **small_cfg,
                norm_type="tok0",
                norm_places=list(combo),
            )
            out = model(raw)
            assert out.shape == (2, small_cfg["K"])
            assert torch.isfinite(out).all()


def test_none_norm_with_empty_places_matches_identity_unembed(small_cfg):
    torch.manual_seed(0)
    m0 = RegressionTransformer(**small_cfg, norm_type="none", norm_places=[])
    torch.manual_seed(0)
    m1 = RegressionTransformer(**small_cfg, norm_type="none", norm_places=["pre_unembed"])

    xs, ys, _ = sample_episodes(2, small_cfg["K"], small_cfg["D"], 0.05)
    raw = to_sequence(xs, ys)

    out0 = m0(raw)
    out1 = m1(raw)
    assert torch.allclose(out0, out1)


def test_default_norm_behavior_matches_explicit_legacy(small_cfg):
    torch.manual_seed(0)
    default_model = RegressionTransformer(**small_cfg)
    torch.manual_seed(0)
    explicit_model = RegressionTransformer(**small_cfg, norm_type="tok0", norm_places=["pre_unembed"])

    xs, ys, _ = sample_episodes(2, small_cfg["K"], small_cfg["D"], 0.05)
    raw = to_sequence(xs, ys)

    out_default = default_model(raw)
    out_explicit = explicit_model(raw)
    assert torch.allclose(out_default, out_explicit)
