import pytest
import torch

from src.components.linear import Linear
from src.components.mlp import MLP
from src.components.attention import Attention
from src.components.similarity import State, similarity
from src.models.transformer import Transformer


def inner_product(state):
    return torch.einsum('ijij->', state.S_ab[:, 1:, :, 1:]).item()


def cosine(state):
    tr = lambda S: torch.einsum('ijij->', S[:, 1:, :, 1:])
    return (tr(state.S_ab) / (tr(state.S_aa) * tr(state.S_bb)) ** 0.5).item()


def mc_inner_product(model_a, model_b, d_input, n_samples=1_000_000, **like):
    """Monte Carlo E[f_a(x)^T f_b(x)] for non-sequential models."""
    x = torch.randn(n_samples, d_input, **like)
    with torch.no_grad():
        return (model_a(x) * model_b(x)).sum(dim=-1).mean().item()


def mc_inner_product_seq(model_a, model_b, d_input, n_ctx, n_samples=200_000, **like):
    """Monte Carlo E[Σ_s f_a(x)_s · f_b(x)_s] for sequential models."""
    x = torch.randn(n_samples, n_ctx, d_input, **like)
    with torch.no_grad():
        return (model_a(x) * model_b(x)).sum(dim=(-1, -2)).mean().item()


# --- Test fixtures: minimal models for specific component tests ---

class MLPModel(torch.nn.Module):
    """Embed -> MLP -> head. Tests bilinear MLP similarity in isolation."""
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


LIKE = dict(device="cpu", dtype=torch.float64)


def assert_exact(state, model_a, model_b, d_input, tol=0.02, n_ctx=None):
    """Assert exact inner product matches MC within tolerance."""
    exact = inner_product(state)
    if n_ctx:
        mc = mc_inner_product_seq(model_a, model_b, d_input, n_ctx, **LIKE)
    else:
        mc = mc_inner_product(model_a, model_b, d_input, **LIKE)
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
        s = similarity(a, b)
        exact = inner_product(s)
        mc = mc_inner_product_seq(a, b, 4, 3, n_samples=500_000, **LIKE)
        assert abs(exact - mc) < max(abs(mc) * 0.3, 5e-6), f"exact={exact:.8f}, mc={mc:.8f}"

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
        s = similarity(a, b)
        exact = inner_product(s)
        mc = mc_inner_product_seq(a, b, 4, 2, n_samples=500_000, **LIKE)
        assert abs(exact - mc) < max(abs(mc) * 0.3, 1e-5), f"exact={exact:.8f}, mc={mc:.8f}"

    def test_two_layer(self):
        torch.manual_seed(42)
        m = Transformer(4, 8, 2, 2, 16, 3, n_layer=2, mask='none', scale=0.5).double()
        s = similarity(m, m)
        assert_self_cosine(s, tol=1e-4)
