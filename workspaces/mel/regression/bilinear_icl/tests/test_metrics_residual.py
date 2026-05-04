import torch

from bilinear_icl.eval import residual
from bilinear_icl.models import RegressionTransformer


class FakeModel(RegressionTransformer):
    pass


def test_residual_metrics_ranges(small_cfg):
    model = RegressionTransformer(**small_cfg)
    B = 64
    xs = torch.randn(B, small_cfg["K"], small_cfg["D"])
    ys = torch.randn(B, small_cfg["K"])
    t = torch.randn(B, small_cfg["D"])
    out = residual.compute(model, (xs, ys, t))
    assert 0.0 <= out["rav"] <= 1.0 + 1e-5
    assert out["erank"] > 0.0
