import torch

from bilinear_icl.models.norm import BOSScalarNorm


def test_bos_scalar_norm_basic():
    n = BOSScalarNorm(eps=1e-6)
    x = torch.randn(2, 5, 8)
    out = n(x)
    assert out.shape == x.shape


def test_bos_scalar_norm_no_nan_on_zero_bos():
    n = BOSScalarNorm(eps=1e-6)
    x = torch.randn(2, 5, 8)
    x[:, 0, :] = 0.0
    out = n(x)
    assert torch.isfinite(out).all()
