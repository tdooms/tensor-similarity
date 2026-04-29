"""Tests for the 2-layer path decomposition.

Three checks:
  1. Forward decomposition  : sum_rho F_rho(X) approx F(X).
  2. Family aggregation     : sum of fine-grained per-head terms within a
                              family approx the family value.
  3. TN similarity additivity: sum_{rho,sigma} <F_rho, F_sigma> approx <F, F>
                              via per-source second-moment propagation.

Test 3 reference: a layered _step propagation with all term-level symmetries
stripped (see NOTE.md). The existing `similarity()` uses orbit-dedup with the
active term's joint swap symmetry, which is an algebraic identity of the
forward pass but not a per-Wick-matching invariance — so it gives a slightly
different scalar than the stripped-symmetry version. The path decomposition
inherits no such symmetry assumption and matches the stripped version
exactly.

Run from repo root:
    pytest workspaces/mel/bilinear_attn/experiments/path_decomp/
"""
import pytest
import torch

from src.components.base import Term
from src.components.mlp import MLP
from src.components.similarity import (
    State, _initial_state, _moment, _step, similarity,
)
from src.models.transformer import Transformer

from workspaces.mel.bilinear_attn.experiments.path_decomp.forward import (
    enumerate_families, family_outputs, family_outputs_fine,
    forward_via_decomposition,
)
from workspaces.mel.bilinear_attn.experiments.path_decomp.moments import (
    family_pair_inner_products,
)


LIKE = dict(device='cpu', dtype=torch.float64)


def _make_attn_only(seed=42, d_in=2, d_model=4, n_head=1, n_ctx=2, d_h=8,
                    d_out=2, scale=0.5, mask='none'):
    """2-layer Transformer with MLPs replaced by passthrough (scale=0)."""
    torch.manual_seed(seed)
    m = Transformer(d_in, d_model, n_head, n_ctx, d_h, d_out,
                    n_layer=2, mask=mask, scale=scale).double()
    for layer in m.body:
        layer.mlp = MLP(d_model, d_h, scale=0.0).double()
    return m


def _strip_sym(terms):
    return [Term(t.tn, t.legs, symmetries=()) for t in terms]


def _similarity_no_sym(model_a, model_b):
    """Layered _step propagation with all term-level symmetries stripped, and
    MLP components skipped (treated as exact passthrough; see NOTE.md for the
    MLP scale=0 issue in the existing similarity).

    The result matches Monte-Carlo ground truth and is what the path
    decomposition's family-pair sum agrees with.
    """
    state = _initial_state(model_a)
    comps_a = model_a.components()
    comps_b = model_b.components()
    for ca, cb in zip(comps_a, comps_b):
        if isinstance(ca, MLP):
            continue  # passthrough at scale=0
        n = state.s_aa.shape[0]
        like = dict(device=state.s_aa.device, dtype=state.s_aa.dtype)
        ta = _strip_sym(ca.terms(n, **like))
        tb = _strip_sym(cb.terms(n, **like))
        state = _step(state, ta, tb)
    return state


# --- Test 1: forward decomposition ----------------------------------------

class TestForwardDecomposition:
    @pytest.mark.parametrize("scale", [0.5, 0.3, 1.0])
    def test_sum_equals_forward(self, scale):
        m = _make_attn_only(scale=scale)
        x = torch.randn(7, 2, 2, **LIKE)
        F_actual = m(x)
        F_decomp = forward_via_decomposition(m, x)
        diff = (F_actual - F_decomp).abs().max().item()
        assert diff < 1e-10, f"max abs diff = {diff:.2e}"

    def test_with_causal_mask(self):
        m = _make_attn_only(mask='causal')
        x = torch.randn(4, 2, 2, **LIKE)
        F_actual = m(x)
        F_decomp = forward_via_decomposition(m, x)
        assert torch.allclose(F_actual, F_decomp, atol=1e-10)


# --- Test 2: family aggregation -------------------------------------------

class TestFamilyAggregation:
    def test_fine_matches_family(self):
        m = _make_attn_only()
        x = torch.randn(5, 2, 2, **LIKE)
        coarse = family_outputs(m, x)
        fine = family_outputs_fine(m, x)

        assert set(coarse.keys()) == set(fine.keys())
        for k in coarse:
            diff = (coarse[k] - fine[k]).abs().max().item()
            assert diff < 1e-10, f"family {k}: max diff = {diff:.2e}"

    def test_family_count(self):
        fams = list(enumerate_families())
        assert len(fams) == 34
        assert fams[0] == 'direct'
        assert fams[1] == 'layer1'
        for i, f in enumerate(fams[2:]):
            assert f == ('layer2', i)


# --- Test 3: TN similarity additivity -------------------------------------

class TestTNSimilarityAdditivity:
    def test_self_similarity(self):
        m = _make_attn_only(seed=42)
        ref_state = _similarity_no_sym(m, m)
        ref = torch.einsum('ijij->', ref_state.s_ab[:, 1:, :, 1:]).item()
        matrix, total = family_pair_inner_products(m, m)
        assert len(matrix) == 34 * 34
        rel = abs(total - ref) / max(abs(ref), 1e-12)
        assert rel < 1e-8, f"total={total:.6e} ref={ref:.6e} rel={rel:.2e}"

    def test_cross_similarity(self):
        a = _make_attn_only(seed=42)
        b = _make_attn_only(seed=7)
        ref_state = _similarity_no_sym(a, b)
        ref = torch.einsum('ijij->', ref_state.s_ab[:, 1:, :, 1:]).item()
        matrix, total = family_pair_inner_products(a, b)
        rel = abs(total - ref) / max(abs(ref), 1e-12)
        assert rel < 1e-8, f"total={total:.6e} ref={ref:.6e} rel={rel:.2e}"

    def test_self_diagonal_nonneg(self):
        """For self-similarity, every diagonal entry <F_rho, F_rho> >= 0."""
        m = _make_attn_only(seed=42)
        matrix, _ = family_pair_inner_products(m, m)
        for fam in enumerate_families():
            v = matrix[(fam, fam)]
            assert v >= -1e-10, f"{fam}: diagonal entry {v:.3e} is negative"

    def test_matches_monte_carlo(self):
        """Path-decomp total should match MC ground truth within stochastic tol."""
        m = _make_attn_only(seed=42)
        torch.manual_seed(0)
        x = torch.randn(200_000, 2, 2, **LIKE)
        with torch.no_grad():
            y = m(x)
        mc = (y * y).sum(dim=(-1, -2)).mean().item()
        _, total = family_pair_inner_products(m, m)
        rel = abs(total - mc) / max(abs(mc), 1e-12)
        assert rel < 0.02, f"total={total:.4e} mc={mc:.4e} rel={rel:.3%}"

    def test_self_symmetric(self):
        """For self-similarity, the matrix is symmetric in (fam_a, fam_b)."""
        m = _make_attn_only(seed=42)
        matrix, _ = family_pair_inner_products(m, m)
        for fa in enumerate_families():
            for fb in enumerate_families():
                d = abs(matrix[(fa, fb)] - matrix[(fb, fa)])
                # Tolerant scaling by magnitude; numerical fp.
                scale = max(abs(matrix[(fa, fb)]), abs(matrix[(fb, fa)]), 1e-12)
                assert d / scale < 1e-7, f"asym {fa},{fb}: {d:.2e}"
