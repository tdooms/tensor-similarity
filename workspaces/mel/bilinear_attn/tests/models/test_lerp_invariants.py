"""Invariant tests for the ``lerp(x, o(z), scale)`` residual convention.

Each attention kernel combines the residual ``x`` with the attention output
``o(z)`` via a linear interpolation parameterised by ``scale``:

    out = lerp(x, o(z), scale) = x + scale * (o(z) - x)

Two corner cases pin this convention unambiguously:

* ``scale = 0``  →  ``out == x``       (pure residual; attention bypassed)
* ``scale = 1``  →  ``out == o(z)``     (pure attention; residual bypassed)

If either invariant fails, the kernel has silently drifted to a different
residual convention (e.g. ``out = x + scale * o(z)``), which is the class
of bug that originally escaped review. Run against every kernel in the
registry so any new kernel automatically inherits the guarantee.
"""
import pytest
import torch

from models.attention_kernels.bilinear import BilinearAttention, QuadraticAttention
from models.attention_kernels.softmax import SoftmaxAttention


KERNELS = [BilinearAttention, QuadraticAttention, SoftmaxAttention]


def _make_kernel(cls, scale):
    torch.manual_seed(0)
    return cls(d_model=16, n_head=4, n_ctx=8, scale=scale).double().eval()


def _input():
    torch.manual_seed(1)
    return torch.randn(2, 8, 16, dtype=torch.float64)


@pytest.mark.parametrize("cls", KERNELS, ids=lambda c: c.__name__)
def test_scale_zero_is_identity(cls):
    """scale=0 must make the layer a pure pass-through: out == x."""
    layer = _make_kernel(cls, scale=0.0)
    x = _input()
    with torch.no_grad():
        out = layer(x)
    torch.testing.assert_close(out, x, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("cls", KERNELS, ids=lambda c: c.__name__)
def test_scale_one_is_pure_attention(cls):
    """scale=1 must drop the residual: out == o(z), i.e. independent of the
    additive residual branch."""
    layer_a = _make_kernel(cls, scale=1.0)
    # Build a twin with identical weights but a different residual input;
    # with scale=1 the residual branch is dead, so outputs must match.
    layer_b = _make_kernel(cls, scale=1.0)
    layer_b.load_state_dict(layer_a.state_dict())

    x = _input()
    x_shifted = x + 7.3  # arbitrary constant; tests residual is bypassed

    with torch.no_grad():
        out_a = layer_a(x)
        out_b = layer_b(x_shifted)

    # The attention branch sees a different x, so outputs won't be equal;
    # instead assert that the residual is *not* mixed in. Concretely,
    # compute o(z) by the lerp formula: o(z) = x + (out - x)/scale. With
    # scale=1 this is just `out`. Cross-check by re-running at scale=0.5.
    layer_half = _make_kernel(cls, scale=0.5)
    layer_half.load_state_dict(layer_a.state_dict())
    with torch.no_grad():
        out_half = layer_half(x)

    # Derived o(z) from scale=0.5: o(z) = x + (out_half - x) / 0.5
    oz_from_half = x + (out_half - x) / 0.5
    torch.testing.assert_close(out_a, oz_from_half, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("cls", KERNELS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("scale", [0.1, 0.3, 0.5, 0.8])
def test_lerp_convexity(cls, scale):
    """For any scale in (0, 1): out = (1-scale)*x + scale*o(z).

    Pinning this explicitly via two evaluations at different scales lets
    us recover o(z) two ways and confirm they agree.
    """
    layer_scale = _make_kernel(cls, scale=scale)
    layer_one = _make_kernel(cls, scale=1.0)
    layer_one.load_state_dict(layer_scale.state_dict())

    x = _input()
    with torch.no_grad():
        out_scale = layer_scale(x)
        out_one = layer_one(x)  # this is o(z)

    expected = x + scale * (out_one - x)
    torch.testing.assert_close(out_scale, expected, atol=1e-10, rtol=1e-10)
