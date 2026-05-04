import torch

from bilinear_icl.models.bilinear_mlp import BilinearMLP


def test_bilinear_mlp_residual_and_shape():
    x = torch.randn(3, 5, 16)
    m0 = BilinearMLP(16, 16, scale=0.0)
    out0 = m0(x)
    assert out0.shape == x.shape
    assert torch.allclose(out0, x)

    m1 = BilinearMLP(16, 16, scale=1.0)
    out1 = m1(x)
    direct = m1.d(m1.l(x) * m1.r(x))
    assert torch.allclose(out1, direct)
    assert m1.l.bias is None and m1.r.bias is None and m1.d.bias is None
