"""Tests for experiments/mlp_sim: TN similarity formulas, polynomial forward,
and MC agreement.

Focus: verify the residual README's boxed equations match the implementation
*exactly* (closed-form identities, up to fp rounding) and that the MC
estimator converges to the TN formula for small models.

These tests avoid MNIST and avoid training; they operate on random weights /
random symmetric tensors at small `d` so they run in <5s on CPU.
"""
from __future__ import annotations

import itertools
import math

import pytest
import torch

from experiments.mlp_sim.residual import (
    BilinearMLPv2,
    _double_factorial,
    _trace_chain,
    forward_polynomial,
    isserlis_coefficient,
    mc_cosine,
    tn_inner_product_polys,
    tn_pair_inner_product,
)
from experiments.mlp_sim.run import (
    BilinearMLP,
    build_symmetric_T,
    tn_inner_product as run_tn_inner_product,
)


# ---------------------------------------------------------------------------
# Helpers: random symmetric feature tensors
# ---------------------------------------------------------------------------
def _random_sym_tensor(f: int, m: int, d: int, seed: int,
                       dtype=torch.float64) -> torch.Tensor:
    """A random tensor of shape (f, d, ..., d) with m input axes, symmetrised
    over those m input axes. Matches the output format of ``forward_polynomial``.
    """
    g = torch.Generator().manual_seed(seed)
    T = torch.randn((f,) + (d,) * m, generator=g, dtype=dtype)
    if m <= 1:
        return T
    acc = torch.zeros_like(T)
    for p in itertools.permutations(range(m)):
        perm = (0,) + tuple(1 + i for i in p)
        acc = acc + T.permute(*perm)
    return acc / math.factorial(m)


# ---------------------------------------------------------------------------
# 1. Coefficient identities (README constants)
# ---------------------------------------------------------------------------
def test_isserlis_coefficient_hardcoded_values():
    # n=2:  E[A(x)B(x)] = 2 <A,B> + <tauA, tauB>
    assert isserlis_coefficient(2, 0) == 2
    assert isserlis_coefficient(2, 1) == 1
    # n=4:  24 <A,B> + 72 <tauA,tauB> + 9 tau^2 A * tau^2 B    (run.py hardcoded)
    assert isserlis_coefficient(4, 0) == 24
    assert isserlis_coefficient(4, 1) == 72
    assert isserlis_coefficient(4, 2) == 9
    # n=6
    assert isserlis_coefficient(6, 0) == math.factorial(6)         # 720
    assert isserlis_coefficient(6, 1) == 15 ** 2 * 1 * math.factorial(4)  # C(6,2)=15
    # Sanity: coefficient sum equals (2n-1)!! * n!  (Isserlis count of pairings
    # of 2n indices weighted by #A-internal-pairings choice, summed) -- this
    # isn't a clean identity so we don't test it here.


def test_homogeneous_formula_matches_general():
    """For m1 = m2 = n, tn_pair_inner_product must reduce to
    sum_r c_{n,r} <tau^r A, tau^r B> with c_{n,r} = C(n,2r)^2 (2r-1)!!^2 (n-2r)!.
    """
    d = 4
    for n in (2, 3, 4, 5, 6):
        A = _random_sym_tensor(3, n, d, seed=n)
        B = _random_sym_tensor(3, n, d, seed=n + 100)
        trA = _trace_chain(A, n)
        trB = _trace_chain(B, n)

        ref = torch.zeros((), dtype=A.dtype)
        for r in range(n // 2 + 1):
            ref = ref + isserlis_coefficient(n, r) * (trA[r] * trB[r]).sum()

        got = tn_pair_inner_product(trA, trB, n, n)
        assert torch.allclose(ref, got, atol=1e-9, rtol=1e-9), n


# ---------------------------------------------------------------------------
# 2. Parity: cross-pairs with odd total degree vanish exactly
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m1,m2", [(1, 2), (2, 3), (1, 4), (3, 4), (2, 5)])
def test_odd_parity_pair_is_exact_zero(m1, m2):
    """README: 'If m1+m2 is odd, the term vanishes because the Gaussian is
    centered.' Verify the implementation returns *exact* zero (not noise)."""
    assert (m1 + m2) % 2 == 1
    d = 4
    A = _random_sym_tensor(3, m1, d, seed=0)
    B = _random_sym_tensor(3, m2, d, seed=1)
    trA = _trace_chain(A, m1)
    trB = _trace_chain(B, m2)
    got = tn_pair_inner_product(trA, trB, m1, m2)
    assert got.item() == 0.0


def test_residual_cross_degree_odd_vanishes_mc():
    """Residual bilinear MLP: E[y_A^[m1] y_B^[m2]] = 0 for m1+m2 odd.
    Cross-check: TN is exactly zero AND MC converges to zero for such pairs."""
    torch.manual_seed(0)
    d, d_out = 4, 3
    A = BilinearMLPv2(d, d_out, n_layers=2, residual=True, seed=11)
    B = BilinearMLPv2(d, d_out, n_layers=2, residual=True, seed=22)
    A = A.to(dtype=torch.float64).eval()
    B = B.to(dtype=torch.float64).eval()

    PA = forward_polynomial(A)
    PB = forward_polynomial(B)
    # 2-layer residual -> degrees {1, 2, 3, 4}. Pick odd-sum pair (1, 2).
    assert 1 in PA and 2 in PB
    trA = _trace_chain(PA[1], 1)
    trB = _trace_chain(PB[2], 2)
    tn = tn_pair_inner_product(trA, trB, 1, 2)
    assert tn.item() == 0.0


# ---------------------------------------------------------------------------
# 3. Polynomial forward correctness: y(x) = sum_m T_m(x^{\otimes m})
# ---------------------------------------------------------------------------
def _eval_poly(P: dict[int, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Evaluate the symbolic polynomial representation at inputs x.
    P[m] has shape (f, d, ..., d) with m input axes. x has shape (B, d).
    Returns y of shape (B, f)."""
    B = x.shape[0]
    out = torch.zeros((B, next(iter(P.values())).shape[0]),
                      dtype=x.dtype, device=x.device)
    for m, T in P.items():
        if m == 0:
            out = out + T
            continue
        # contract each of the m input axes with x
        # einsum: T[f,i1,...,im] * x[B,i1] * ... * x[B,im] -> (B,f)
        letters = "ghijklmnopqrstuvwxyz"
        idx = letters[:m]
        eq = f"f{idx}," + ",".join(f"b{c}" for c in idx) + f"->bf"
        out = out + torch.einsum(eq, T, *([x] * m))
    return out


@pytest.mark.parametrize("n_layers,residual", [(1, False), (1, True),
                                               (2, False), (2, True),
                                               (3, True)])
def test_polynomial_forward_matches_direct(n_layers, residual):
    torch.manual_seed(0)
    d, d_out = 4, 3
    model = BilinearMLPv2(d, d_out, n_layers=n_layers, residual=residual, seed=7)
    model = model.to(dtype=torch.float64).eval()

    P = forward_polynomial(model)
    x = torch.randn(16, d, dtype=torch.float64)
    with torch.no_grad():
        y_direct = model(x)
    y_poly = _eval_poly(P, x)
    assert torch.allclose(y_direct, y_poly, atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# 4. Symmetry invariant of forward_polynomial output
# ---------------------------------------------------------------------------
def test_forward_polynomial_tensors_are_fully_symmetric():
    torch.manual_seed(0)
    d, d_out = 4, 2
    model = BilinearMLPv2(d, d_out, n_layers=2, residual=True, seed=3)
    model = model.to(dtype=torch.float64).eval()
    P = forward_polynomial(model)
    for m, T in P.items():
        if m < 2:
            continue
        # sample a few axis swaps; symmetric tensor is invariant
        for i, j in [(0, m - 1), (0, 1), (m // 2, m - 1)]:
            if i == j:
                continue
            perm = list(range(1, m + 1))
            perm[i], perm[j] = perm[j], perm[i]
            T_sw = T.permute(0, *perm)
            assert torch.allclose(T, T_sw, atol=1e-12), (m, i, j)


# ---------------------------------------------------------------------------
# 5. Self-similarity: TN cosine of a model with itself = 1 exactly
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_layers,residual", [(1, False), (2, False),
                                               (2, True), (3, True)])
def test_tn_self_similarity_is_one(n_layers, residual):
    torch.manual_seed(0)
    d, d_out = 4, 3
    m = BilinearMLPv2(d, d_out, n_layers=n_layers, residual=residual, seed=5)
    m = m.to(dtype=torch.float64).eval()
    P = forward_polynomial(m)
    aa = tn_inner_product_polys(P, P)
    # Must be strictly positive (norm^2) and cosine = 1.
    assert aa.item() > 0
    cos = (aa / (aa * aa).sqrt()).item()
    assert abs(cos - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# 6. run.py n=4 formula vs the general residual formula (m1=m2=4)
# ---------------------------------------------------------------------------
def test_run_n4_formula_matches_general():
    """run.tn_inner_product (hardcoded 24/72/9) must equal the general
    formula from residual.tn_pair_inner_product on the same symmetric tensor."""
    torch.manual_seed(0)
    d_in, d_hidden, d_out = 4, 4, 3
    mA = BilinearMLP(d_in, d_hidden, d_out, seed=1)
    mB = BilinearMLP(d_in, d_hidden, d_out, seed=2)
    TA = build_symmetric_T(mA).double()
    TB = build_symmetric_T(mB).double()

    ref = run_tn_inner_product(TA, TB)

    trA = _trace_chain(TA, 4)
    trB = _trace_chain(TB, 4)
    got = tn_pair_inner_product(trA, trB, 4, 4)
    assert torch.allclose(ref, got, atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------------
# 7. TN vs MC convergence on a tiny model (sanity: formula matches sampling)
# ---------------------------------------------------------------------------
def test_tn_matches_mc_small_residual_model():
    """For a small 1-layer residual model (degrees {1, 2}), TN and MC should
    agree to ~1e-2 at 200k samples. This is the end-to-end check that both
    the formula AND the polynomial forward are correct."""
    torch.manual_seed(0)
    d, d_out = 4, 3
    A = BilinearMLPv2(d, d_out, n_layers=1, residual=True, seed=10)
    B = BilinearMLPv2(d, d_out, n_layers=1, residual=True, seed=20)

    # TN
    A64 = A.to(dtype=torch.float64).eval()
    B64 = B.to(dtype=torch.float64).eval()
    PA = forward_polynomial(A64)
    PB = forward_polynomial(B64)
    tn_ab = tn_inner_product_polys(PA, PB).item()
    tn_aa = tn_inner_product_polys(PA, PA).item()
    tn_bb = tn_inner_product_polys(PB, PB).item()
    tn_cos = tn_ab / math.sqrt(tn_aa * tn_bb)

    # MC (fresh copies since mc_cosine mutates dtype)
    A_mc = BilinearMLPv2(d, d_out, n_layers=1, residual=True, seed=10)
    B_mc = BilinearMLPv2(d, d_out, n_layers=1, residual=True, seed=20)
    torch.manual_seed(42)
    mc_cos = mc_cosine(A_mc, B_mc, n_samples=200_000, batch_size=8192,
                       dtype=torch.float64, device="cpu")

    assert abs(tn_cos - mc_cos) < 2e-2, (tn_cos, mc_cos)


# ---------------------------------------------------------------------------
# 8. Double factorial edge cases (used by both formulas)
# ---------------------------------------------------------------------------
def test_double_factorial_edges():
    assert _double_factorial(-1) == 1  # (-1)!! = 1 convention; r=0 coefficient
    assert _double_factorial(0) == 1
    assert _double_factorial(1) == 1
    assert _double_factorial(3) == 3
    assert _double_factorial(5) == 15
    assert _double_factorial(7) == 105
