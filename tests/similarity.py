import pytest
import torch

from src.components.linear import Linear
from src.components.mlp import MLP
from src.components.attention import Attention
from src.components.similarity import State, similarity


def inner_product(state):
    return torch.einsum('ijij->', state.S_ab[:, 1:, :, 1:]).item()


def cosine(state):
    tr = lambda S: torch.einsum('ijij->', S[:, 1:, :, 1:])
    return (tr(state.S_ab) / (tr(state.S_aa) * tr(state.S_bb)) ** 0.5).item()


def mc_inner_product(model_a, model_b, d_input, n_samples=1_000_000, **like):
    """Monte Carlo estimate of E[f_a(x)^T f_b(x)] for x ~ N(0, I)."""
    x = torch.randn(n_samples, d_input, **like)
    with torch.no_grad():
        y_a = model_a(x)
        y_b = model_b(x)
    return (y_a * y_b).sum(dim=-1).mean().item()


class SimpleModel(torch.nn.Module):
    """Embed -> head (linear only, no MLP)."""
    def __init__(self, d_input, d_output):
        super().__init__()
        self.embed = Linear(d_input, d_output, bias=False)

    def components(self):
        return [self.embed]

    def forward(self, x):
        return self.embed(x)


class ToyModel(torch.nn.Module):
    """Embed -> MLP -> head."""
    def __init__(self, d_input, d_model, d_hidden, d_output, scale=1.0, bias=False):
        super().__init__()
        self.embed = Linear(d_input, d_model, bias=False)
        self.body = MLP(d_model, d_hidden, bias=bias, scale=scale)
        self.head = Linear(d_model, d_output, bias=False)

    def components(self):
        return [self.embed, self.body, self.head]

    def forward(self, x):
        return self.head(self.body(self.embed(x)))


class TwoLayerModel(torch.nn.Module):
    """Embed -> MLP -> MLP -> head."""
    def __init__(self, d_input, d_model, d_hidden, d_output, scale=1.0, bias=False):
        super().__init__()
        self.embed = Linear(d_input, d_model, bias=False)
        self.body0 = MLP(d_model, d_hidden, bias=bias, scale=scale)
        self.body1 = MLP(d_model, d_hidden, bias=bias, scale=scale)
        self.head = Linear(d_model, d_output, bias=False)

    def components(self):
        return [self.embed, self.body0, self.body1, self.head]

    def forward(self, x):
        return self.head(self.body1(self.body0(self.embed(x))))


LIKE = dict(device="cpu", dtype=torch.float64)


class TestLinearOnly:
    def test_self_similarity(self):
        torch.manual_seed(0)
        model = SimpleModel(6, 4).double()
        state = similarity(model, model)

        exact = inner_product(state)
        mc = mc_inner_product(model, model, 6, **LIKE)

        assert abs(exact - mc) / abs(mc) < 0.02, f"exact={exact:.6f}, mc={mc:.6f}"
        assert abs(cosine(state) - 1.0) < 1e-10

    def test_cross_similarity(self):
        torch.manual_seed(0)
        model_a = SimpleModel(6, 4).double()
        torch.manual_seed(1)
        model_b = SimpleModel(6, 4).double()

        state = similarity(model_a, model_b)
        exact = inner_product(state)
        mc = mc_inner_product(model_a, model_b, 6, **LIKE)

        assert abs(exact - mc) / max(abs(mc), 1e-8) < 0.02, f"exact={exact:.6f}, mc={mc:.6f}"


class TestBilinearNoResidual:
    """MLP with scale=1 (no residual)."""

    def test_self_similarity(self):
        torch.manual_seed(42)
        model = ToyModel(4, 8, 16, 3, scale=1.0).double()
        state = similarity(model, model)

        exact = inner_product(state)
        mc = mc_inner_product(model, model, 4, **LIKE)

        assert abs(exact - mc) / abs(mc) < 0.02, f"exact={exact:.6f}, mc={mc:.6f}"
        assert abs(cosine(state) - 1.0) < 1e-10

    def test_cross_similarity(self):
        torch.manual_seed(42)
        model_a = ToyModel(4, 8, 16, 3, scale=1.0).double()
        torch.manual_seed(99)
        model_b = ToyModel(4, 8, 16, 3, scale=1.0).double()

        state = similarity(model_a, model_b)
        exact = inner_product(state)
        mc = mc_inner_product(model_a, model_b, 4, **LIKE)

        assert abs(exact - mc) / max(abs(mc), 1e-8) < 0.02, f"exact={exact:.6f}, mc={mc:.6f}"


class TestBilinearWithResidual:
    """MLP with scale=0.5 (residual blending)."""

    def test_self_similarity(self):
        torch.manual_seed(42)
        model = ToyModel(4, 8, 16, 3, scale=0.5).double()
        state = similarity(model, model)

        exact = inner_product(state)
        mc = mc_inner_product(model, model, 4, **LIKE)

        assert abs(exact - mc) / abs(mc) < 0.02, f"exact={exact:.6f}, mc={mc:.6f}"
        assert abs(cosine(state) - 1.0) < 1e-10

    def test_cross_similarity(self):
        torch.manual_seed(42)
        model_a = ToyModel(4, 8, 16, 3, scale=0.5).double()
        torch.manual_seed(99)
        model_b = ToyModel(4, 8, 16, 3, scale=0.5).double()

        state = similarity(model_a, model_b)
        exact = inner_product(state)
        mc = mc_inner_product(model_a, model_b, 4, **LIKE)

        assert abs(exact - mc) / max(abs(mc), 1e-8) < 0.02, f"exact={exact:.6f}, mc={mc:.6f}"


class TestPureResidual:
    """MLP with scale=0 (pure residual, should be equivalent to identity)."""

    def test_self_similarity(self):
        torch.manual_seed(42)
        model = ToyModel(4, 8, 16, 3, scale=0.0).double()
        state = similarity(model, model)

        exact = inner_product(state)
        mc = mc_inner_product(model, model, 4, **LIKE)

        assert abs(exact - mc) / abs(mc) < 0.02, f"exact={exact:.6f}, mc={mc:.6f}"


class TestWithBias:
    """MLP with bias=True."""

    def test_self_similarity(self):
        torch.manual_seed(42)
        model = ToyModel(4, 8, 16, 3, scale=0.7, bias=True).double()
        state = similarity(model, model)

        exact = inner_product(state)
        mc = mc_inner_product(model, model, 4, **LIKE)

        assert abs(exact - mc) / abs(mc) < 0.02, f"exact={exact:.6f}, mc={mc:.6f}"
        assert abs(cosine(state) - 1.0) < 1e-10

    def test_cross_similarity(self):
        torch.manual_seed(42)
        model_a = ToyModel(4, 8, 16, 3, scale=0.7, bias=True).double()
        torch.manual_seed(99)
        model_b = ToyModel(4, 8, 16, 3, scale=0.7, bias=True).double()

        state = similarity(model_a, model_b)
        exact = inner_product(state)
        mc = mc_inner_product(model_a, model_b, 4, **LIKE)

        assert abs(exact - mc) / max(abs(mc), 1e-8) < 0.02, f"exact={exact:.6f}, mc={mc:.6f}"


class TestTwoLayer:
    """Embed -> MLP -> MLP -> head. Gaussian assumption is approximate after layer 1."""

    def test_self_similarity(self):
        torch.manual_seed(42)
        model = TwoLayerModel(4, 8, 16, 3, scale=0.5).double()
        state = similarity(model, model)

        exact = inner_product(state)
        mc = mc_inner_product(model, model, 4, **LIKE)

        rel_err = abs(exact - mc) / abs(mc)
        assert rel_err < 0.1, f"exact={exact:.6f}, mc={mc:.6f}, rel_err={rel_err:.4%}"

    def test_cross_similarity(self):
        torch.manual_seed(42)
        model_a = TwoLayerModel(4, 8, 16, 3, scale=0.5).double()
        torch.manual_seed(99)
        model_b = TwoLayerModel(4, 8, 16, 3, scale=0.5).double()

        state = similarity(model_a, model_b)
        exact = inner_product(state)
        mc = mc_inner_product(model_a, model_b, 4, **LIKE)

        rel_err = abs(exact - mc) / max(abs(mc), 1e-8)
        assert rel_err < 0.2, f"exact={exact:.6f}, mc={mc:.6f}, rel_err={rel_err:.4%}"


class AttnModel(torch.nn.Module):
    """Embed -> Attention -> head."""
    def __init__(self, d_input, d_model, n_head, n_ctx, d_output, scale=1.0):
        super().__init__()
        self.embed = Linear(d_input, d_model, bias=False)
        self.attn = Attention(d_model, n_head, n_ctx, mask='none', bias=False, scale=scale)
        self.head = Linear(d_model, d_output, bias=False)

    def components(self):
        return [self.embed, self.attn, self.head]

    def forward(self, x):
        x = self.embed(x)
        x = self.attn(x)
        return self.head(x)


def mc_inner_product_seq(model_a, model_b, d_input, n_ctx, n_samples=200_000, **like):
    """MC estimate of E[Σ_s f_a(x)_s · f_b(x)_s] for x ~ N(0,I) i.i.d. per position."""
    x = torch.randn(n_samples, n_ctx, d_input, **like)
    with torch.no_grad():
        y_a = model_a(x)  # (batch, n_ctx, d_output)
        y_b = model_b(x)
    return (y_a * y_b).sum(dim=(-1, -2)).mean().item()


class TransformerModel(torch.nn.Module):
    """Embed -> Attention -> MLP -> head (single transformer layer)."""
    def __init__(self, d_input, d_model, n_head, n_ctx, d_hidden, d_output, scale=0.5):
        super().__init__()
        self.embed = Linear(d_input, d_model, bias=False)
        self.attn = Attention(d_model, n_head, n_ctx, mask='none', bias=False, scale=scale)
        self.mlp = MLP(d_model, d_hidden, bias=False, scale=scale)
        self.head = Linear(d_model, d_output, bias=False)

    def components(self):
        return [self.embed, self.attn, self.mlp, self.head]

    def forward(self, x):
        x = self.embed(x)
        x = self.attn(x)
        x = self.mlp(x)
        return self.head(x)


class TestTransformerLayer:
    """Embed -> Attention -> MLP -> head (full single transformer layer)."""

    def test_self_similarity(self):
        torch.manual_seed(42)
        model = TransformerModel(4, 8, 2, 2, 16, 3, scale=0.5).double()
        state = similarity(model, model)

        exact = inner_product(state)
        mc = mc_inner_product_seq(model, model, 4, 2, **LIKE)

        rel_err = abs(exact - mc) / abs(mc)
        # Gaussian approximation after attention, so wider tolerance
        assert rel_err < 0.2, f"exact={exact:.6f}, mc={mc:.6f}, rel_err={rel_err:.4%}"
        assert abs(cosine(state) - 1.0) < 1e-6

    def test_cross_similarity(self):
        torch.manual_seed(42)
        model_a = TransformerModel(4, 8, 2, 2, 16, 3, scale=0.5).double()
        torch.manual_seed(99)
        model_b = TransformerModel(4, 8, 2, 2, 16, 3, scale=0.5).double()

        state = similarity(model_a, model_b)
        exact = inner_product(state)
        mc = mc_inner_product_seq(model_a, model_b, 4, 2, n_samples=500_000, **LIKE)

        # Cross-similarity is small; use absolute tolerance
        assert abs(exact - mc) < max(abs(mc) * 0.3, 1e-5), f"exact={exact:.8f}, mc={mc:.8f}"


class TestAttentionNoResidual:
    """Embed -> Attention(scale=1) -> head, no rotary effect at small d."""

    def test_self_similarity(self):
        torch.manual_seed(42)
        model = AttnModel(4, 8, 2, 3, 3, scale=1.0).double()
        state = similarity(model, model)

        exact = inner_product(state)
        mc = mc_inner_product_seq(model, model, 4, 3, **LIKE)

        rel_err = abs(exact - mc) / abs(mc)
        assert rel_err < 0.1, f"exact={exact:.6f}, mc={mc:.6f}, rel_err={rel_err:.4%}"

    def test_cross_similarity(self):
        torch.manual_seed(42)
        model_a = AttnModel(4, 8, 2, 3, 3, scale=1.0).double()
        torch.manual_seed(99)
        model_b = AttnModel(4, 8, 2, 3, 3, scale=1.0).double()

        state = similarity(model_a, model_b)
        exact = inner_product(state)
        mc = mc_inner_product_seq(model_a, model_b, 4, 3, n_samples=500_000, **LIKE)

        # Cross-similarity of random attention is near zero; use absolute tolerance
        assert abs(exact - mc) < 5e-6, f"exact={exact:.8f}, mc={mc:.8f}"


class TestAttentionWithResidual:
    """Embed -> Attention(scale=0.5) -> head."""

    def test_self_similarity(self):
        torch.manual_seed(42)
        model = AttnModel(4, 8, 2, 3, 3, scale=0.5).double()
        state = similarity(model, model)

        exact = inner_product(state)
        mc = mc_inner_product_seq(model, model, 4, 3, **LIKE)

        rel_err = abs(exact - mc) / abs(mc)
        assert rel_err < 0.1, f"exact={exact:.6f}, mc={mc:.6f}, rel_err={rel_err:.4%}"
