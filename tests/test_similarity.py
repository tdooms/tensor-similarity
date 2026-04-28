import pytest
import torch

from src.components.linear import Linear
from src.components.mlp import MLP
from src.components.attention import Attention
from src.components.similarity import similarity
from src.models.transformer import Transformer


def cosine(s):
    """Cosine from the 2x2 stacked similarity output. `_propagate` returns
    Σ normalized per-layer; the prefactors cancel in the ratio."""
    tr = lambda x: torch.einsum('ijij->', x[:, 1:, :, 1:])
    return (tr(s[0, 1]) / (tr(s[0, 0]).clamp_min(0).sqrt()
                           * tr(s[1, 1]).clamp_min(0).sqrt() + 1e-30)).item()


def mc_cosine(model_a, model_b, d_input, n_samples=1_000_000, n_ctx=None):
    """MC cosine: scale-invariant, matches `cosine(s)` after the un-scale was removed.
    `n_ctx=None` for non-sequential models, integer for sequential. Sampling
    inherits the model's dtype/device so fp64 toy models don't fail at .randn()."""
    p = next(model_a.parameters())
    shape = (n_samples, d_input) if n_ctx is None else (n_samples, n_ctx, d_input)
    dims = (-1,) if n_ctx is None else (-1, -2)
    x = torch.randn(*shape, device=p.device, dtype=p.dtype)
    with torch.no_grad():
        a, b = model_a(x), model_b(x)
        ip = (a * b).sum(dim=dims).mean().item()
        norm_a = (a * a).sum(dim=dims).mean().sqrt().item()
        norm_b = (b * b).sum(dim=dims).mean().sqrt().item()
    return ip / (norm_a * norm_b + 1e-30)


# --- Test fixtures: minimal models for specific component tests ---

class MLPModel(torch.nn.Module):
    """Embed -> MLP -> head. Tests bilinear MLP similarity in isolation."""
    n_ctx = 1
    def __init__(self, d_in, d, d_h, d_out, **kwargs):
        super().__init__()
        self.embed = Linear(d_in, d, bias=False)
        self.body = MLP(d, d_h, **kwargs)
        self.head = Linear(d, d_out, bias=False)

    def components(self):
        return [self.embed, self.body, self.head]

    def forward(self, x):
        return self.head(self.body(self.embed(x)))


class TwoLayerMLPModel(torch.nn.Module):
    """Embed -> MLP -> MLP -> head. Tests Gaussian propagation through 2 layers."""
    n_ctx = 1
    def __init__(self, d_in, d, d_h, d_out, **kwargs):
        super().__init__()
        self.embed = Linear(d_in, d, bias=False)
        self.mlp0 = MLP(d, d_h, **kwargs)
        self.mlp1 = MLP(d, d_h, **kwargs)
        self.head = Linear(d, d_out, bias=False)

    def components(self):
        return [self.embed, self.mlp0, self.mlp1, self.head]

    def forward(self, x):
        return self.head(self.mlp1(self.mlp0(self.embed(x))))


LIKE = dict(device="cpu", dtype=torch.float32)


def assert_exact(state, model_a, model_b, d_input, tol=0.02, n_ctx=None, n_samples=1_000_000):
    """Assert exact cosine matches MC cosine within tolerance."""
    exact = cosine(state)
    mc = mc_cosine(model_a, model_b, d_input, n_samples=n_samples, n_ctx=n_ctx)
    rel_err = abs(exact - mc) / max(abs(mc), 1e-8)
    assert rel_err < tol, f"exact={exact:.6f}, mc={mc:.6f}, rel_err={rel_err:.4%}"


def assert_self_cosine(state, tol=1e-10):
    assert abs(cosine(state) - 1.0) < tol


# --- Linear ---

class TestLinear:
    def test_self(self):
        torch.manual_seed(0)
        m = MLPModel(6, 8, 16, 4, scale=0.0).double()  # scale=0 => pure linear
        s = similarity(m, m)
        assert_exact(s, m, m, 6)
        assert_self_cosine(s)

    def test_cross(self):
        torch.manual_seed(0); a = MLPModel(6, 8, 16, 4, scale=0.0).double()
        torch.manual_seed(1); b = MLPModel(6, 8, 16, 4, scale=0.0).double()
        assert_exact(similarity(a, b), a, b, 6)


# --- Bilinear MLP ---

class TestBilinear:
    def test_no_residual(self):
        torch.manual_seed(42)
        m = MLPModel(4, 8, 16, 3, scale=1.0).double()
        s = similarity(m, m)
        assert_exact(s, m, m, 4)
        assert_self_cosine(s)

    def test_with_residual(self):
        torch.manual_seed(42)
        m = MLPModel(4, 8, 16, 3, scale=0.5).double()
        s = similarity(m, m)
        assert_exact(s, m, m, 4)
        assert_self_cosine(s)

    def test_with_bias(self):
        torch.manual_seed(42)
        m = MLPModel(4, 8, 16, 3, scale=0.7, bias=True).double()
        s = similarity(m, m)
        assert_exact(s, m, m, 4)
        assert_self_cosine(s)

    def test_cross(self):
        torch.manual_seed(42); a = MLPModel(4, 8, 16, 3, scale=0.5).double()
        torch.manual_seed(99); b = MLPModel(4, 8, 16, 3, scale=0.5).double()
        assert_exact(similarity(a, b), a, b, 4)

    def test_cross_with_bias(self):
        torch.manual_seed(42); a = MLPModel(4, 8, 16, 3, scale=0.7, bias=True).double()
        torch.manual_seed(99); b = MLPModel(4, 8, 16, 3, scale=0.7, bias=True).double()
        assert_exact(similarity(a, b), a, b, 4)


# --- Two-layer MLP (Gaussian approximation) ---

class TestTwoLayerMLP:
    def test_self(self):
        torch.manual_seed(42)
        m = TwoLayerMLPModel(4, 8, 16, 3, scale=0.5).double()
        s = similarity(m, m)
        assert_exact(s, m, m, 4, tol=0.1)

    def test_cross(self):
        torch.manual_seed(42); a = TwoLayerMLPModel(4, 8, 16, 3, scale=0.5).double()
        torch.manual_seed(99); b = TwoLayerMLPModel(4, 8, 16, 3, scale=0.5).double()
        assert_exact(similarity(a, b), a, b, 4, tol=0.2)


# --- Attention (using Transformer with n_layer=1, no MLP via scale trick) ---

class TestAttention:
    def test_no_residual(self):
        torch.manual_seed(42)
        m = Transformer(4, 8, 2, 3, 16, 3, n_layer=1, mask='none', scale=1.0).double()
        # Remove MLP effect by setting its scale to 0 (pure passthrough)
        m.body[0].mlp = MLP(8, 16, scale=0.0).double()
        s = similarity(m, m)
        assert_exact(s, m, m, 4, tol=0.1, n_ctx=3)

    def test_with_residual(self):
        torch.manual_seed(42)
        m = Transformer(4, 8, 2, 3, 16, 3, n_layer=1, mask='none', scale=0.5).double()
        m.body[0].mlp = MLP(8, 16, scale=0.0).double()
        s = similarity(m, m)
        assert_exact(s, m, m, 4, tol=0.1, n_ctx=3)

    def test_cross(self):
        torch.manual_seed(42)
        a = Transformer(4, 8, 2, 3, 16, 3, n_layer=1, mask='none', scale=1.0).double()
        a.body[0].mlp = MLP(8, 16, scale=0.0).double()
        torch.manual_seed(99)
        b = Transformer(4, 8, 2, 3, 16, 3, n_layer=1, mask='none', scale=1.0).double()
        b.body[0].mlp = MLP(8, 16, scale=0.0).double()
        assert_exact(similarity(a, b), a, b, 4, tol=0.3, n_ctx=3, n_samples=500_000)

    def test_causal_mask(self):
        torch.manual_seed(42)
        m = Transformer(4, 8, 2, 3, 16, 3, n_layer=1, mask='causal', scale=0.5).double()
        m.body[0].mlp = MLP(8, 16, scale=0.0).double()
        s = similarity(m, m)
        assert_exact(s, m, m, 4, tol=0.1, n_ctx=3)
        assert_self_cosine(s, tol=1e-6)

    def test_with_bias(self):
        torch.manual_seed(42)
        m = Transformer(4, 8, 2, 3, 16, 3, n_layer=1, mask='none', bias=True, scale=0.5).double()
        m.body[0].mlp = MLP(8, 16, scale=0.0, bias=True).double()
        s = similarity(m, m)
        assert_exact(s, m, m, 4, tol=0.1, n_ctx=3)
        assert_self_cosine(s, tol=1e-6)


# --- Full transformer (attention + MLP) ---

class TestTransformer:
    def test_single_layer(self):
        torch.manual_seed(42)
        m = Transformer(4, 8, 2, 2, 16, 3, n_layer=1, mask='none', scale=0.5).double()
        s = similarity(m, m)
        assert_exact(s, m, m, 4, tol=0.2, n_ctx=2)
        assert_self_cosine(s, tol=1e-6)

    def test_single_layer_cross(self):
        torch.manual_seed(42)
        a = Transformer(4, 8, 2, 2, 16, 3, n_layer=1, mask='none', scale=0.5).double()
        torch.manual_seed(99)
        b = Transformer(4, 8, 2, 2, 16, 3, n_layer=1, mask='none', scale=0.5).double()
        assert_exact(similarity(a, b), a, b, 4, tol=0.3, n_ctx=2, n_samples=500_000)

    def test_two_layer(self):
        torch.manual_seed(42)
        m = Transformer(4, 8, 2, 2, 16, 3, n_layer=2, mask='none', scale=0.5).double()
        s = similarity(m, m)
        assert_self_cosine(s, tol=1e-4)


# --- Scale (regression) ---

@pytest.mark.slow
class TestScale:
    def test_two_layer_d128_ctx4_big_input(self):
        # Melwina's case. Without @torch.no_grad on _run, the autograd graph
        # retains every intermediate across all 266 dedup'd attn act×act
        # configs and OOMs at >10 GB peak; with it, ~15 GB cold and stable.
        torch.manual_seed(42)
        m = Transformer(4096, 128, 8, 4, 128, 4096, n_layer=2,
                        mask='none', scale=0.5).double()
        assert_self_cosine(similarity(m, m), tol=1e-4)


# --- Numerical robustness (regression for the fp32 overflow bug) ---

class TestCosineFromPartsRobustness:
    """`cosine_from_parts` consumes the (aa, ab, bb) triple from `_propagate`.
    Before commit X, it computed `(aa*bb).sqrt()`, which overflowed fp32 at
    vocab scale (trace > ~1.8e19 ⇒ trace² > 3.4e38 = fp32 max ⇒ +inf ⇒ cos=0).
    The fix is `aa.sqrt() * bb.sqrt()`. These tests pin both halves: the math
    invariant (self-cosine = 1) and the overflow-resilience invariant (works
    with traces past the old ceiling)."""

    @staticmethod
    def _self_pair(trace_target, dtype):
        """Build aa = ab = bb with the einsum trace ≈ trace_target.
        Shape (n_ctx=2, d=3, n_ctx=2, d=3); only the (i, j>0, i, j>0) diagonals
        contribute via the [:, 1:, :, 1:] slice, so 4 cells total."""
        n_ctx, d = 2, 3
        diag = trace_target / (n_ctx * (d - 1))  # 4 cells
        x = torch.zeros(n_ctx, d, n_ctx, d, dtype=dtype)
        for i in range(n_ctx):
            for j in range(1, d):
                x[i, j, i, j] = diag
        return x

    @pytest.mark.parametrize("trace", [1.0, 1e6, 1e15, 1e19, 1e25])
    def test_self_cosine_is_one_across_dynamic_range_fp32(self, trace):
        """Regression: the old code returned 0 for trace ≥ ~1.8e19 in fp32."""
        from src.figures import cosine_from_parts
        x = self._self_pair(trace, torch.float32)
        assert cosine_from_parts(x, x, x) == pytest.approx(1.0, abs=1e-5)

    def test_self_cosine_is_one_at_fp32_max(self):
        """Past where `(aa*bb).sqrt()` would overflow but `aa.sqrt()*bb.sqrt()` doesn't.
        fp32 max ~3.4e38; trace ≈ 1e19 ⇒ trace² ≈ 1e38 (boundary)."""
        from src.figures import cosine_from_parts
        x = self._self_pair(1e19, torch.float32)
        assert cosine_from_parts(x, x, x) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_inputs_give_zero(self):
        """Sanity: two structurally orthogonal Σ should give cos = 0."""
        from src.figures import cosine_from_parts
        n_ctx, d = 2, 4
        a = torch.zeros(n_ctx, d, n_ctx, d)
        b = torch.zeros(n_ctx, d, n_ctx, d)
        a[0, 1, 0, 1] = 1.0  # mass on dim 1
        b[0, 2, 0, 2] = 1.0  # mass on dim 2 — disjoint
        ab = torch.zeros(n_ctx, d, n_ctx, d)  # zero cross
        assert cosine_from_parts(a, ab, b) == pytest.approx(0.0, abs=1e-6)


class TestCrossPairSymmetry:
    """`cos(a, b)` must equal `cos(b, a)` — trace is invariant to transpose.
    fp64 is bit-exact; fp32 holds to ULP-scale on toy models. A regression of
    these would mean a real symmetry break in `_update`/`_propagate`."""

    def test_fp64_bit_exact(self):
        torch.manual_seed(42); a = MLPModel(4, 8, 16, 3, scale=0.5).double()
        torch.manual_seed(99); b = MLPModel(4, 8, 16, 3, scale=0.5).double()
        assert cosine(similarity(a, b)) == cosine(similarity(b, a))

    def test_fp32_within_ulp(self):
        torch.manual_seed(42); a = MLPModel(4, 8, 16, 3, scale=1.0).float()
        torch.manual_seed(99); b = MLPModel(4, 8, 16, 3, scale=1.0).float()
        diff = abs(cosine(similarity(a, b)) - cosine(similarity(b, a)))
        assert diff < 1e-5, f"fp32 asymmetry {diff:.2e} exceeds ULP-scale tolerance"

    def test_repeat_call_is_deterministic(self):
        """Same call ordering, repeated, must give bit-identical results."""
        torch.manual_seed(42); a = MLPModel(4, 8, 16, 3, scale=1.0).float()
        torch.manual_seed(99); b = MLPModel(4, 8, 16, 3, scale=1.0).float()
        c1 = cosine(similarity(a, b))
        c2 = cosine(similarity(a, b))
        c3 = cosine(similarity(a, b))
        assert c1 == c2 == c3


class TestPropagationNormalisation:
    """`_propagate` returns *normalized* Σ (no un-scale). For self-pairs this means
    the three returned tensors are the same object, with max|·| ≤ 1 per element.
    For cross-pairs they're independently normalized."""

    def test_self_pair_returns_aliased_tensors(self):
        """Self-pair: aa, ab, bb must alias the same tensor."""
        from src.components.similarity import _propagate, _precompile_mode
        torch.manual_seed(0)
        m = MLPModel(4, 8, 16, 3, scale=0.5).double()
        with torch.no_grad(), _precompile_mode():
            aa, ab, bb = _propagate(m, m)
        assert aa is ab and ab is bb, "self-pair should alias"

    def test_self_pair_normalized_max_abs_is_one(self):
        """Per-layer normalization: returned Σ should have max|·| ≤ 1."""
        from src.components.similarity import _propagate, _precompile_mode
        torch.manual_seed(0)
        m = MLPModel(4, 8, 16, 3, scale=0.5).double()
        with torch.no_grad(), _precompile_mode():
            aa, _, _ = _propagate(m, m)
        assert aa.abs().max().item() <= 1.0 + 1e-6
