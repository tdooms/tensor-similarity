"""Per-source TN similarity (vectorized) for the 2-layer path decomposition.

Builds 4 'master' tensors (one per layer-2 term-pair: residual/active x
residual/active) in which each input leg keeps an OPEN size-2 source axis.
The 34*34 family-pair output moments are then read off as plain slices of
the master tensors. This collapses ~1156 individual Wick contractions into
just 4 (one per term-pair), each of cost comparable to the existing
similarity()'s per-layer cost.

The match between the path decomposition and the existing layered-covariance
similarity is exact (both perform Gaussian propagation across attention).
"""
from functools import cache
from math import prod

import torch

from src.components.base import Term
from src.components.similarity import State, _initial_state, _moment, _step, _join, _OUT
from src.components.utils import bridged_contract, matchings, orbits

from workspaces.mel.bilinear_attn.experiments.path_decomp.forward import (
    N_LAYER2_FAMILIES, enumerate_families,
)


# ---------------------------------------------------------------------------
# Family -> term type and per-leg source bits
# ---------------------------------------------------------------------------

# Term-type code: 0 = residual term (1 leg), 1 = active term (5 legs).

def _family_to_tt_and_src(family):
    """(term_type, src_bits_per_leg) for one family.

    Source bit per leg:
      0 -> leg reads from layer-1 RESIDUAL output  (r0 = (1-s1)*x).
      1 -> leg reads from layer-1 ACTIVE   output  (rs = s1*sum_h head_h_l1).

    For l2-residual (1 leg): src = 0 ('direct') or 1 ('layer1').
    For l2-active (5 legs in sorted-key order in:d0..in:d4 = V, K1, K2, Q1, Q2):
        rho bits = (alpha=Q1, beta=K1, gamma=Q2, delta=K2, eta=V)
        src      = (eta,      beta,    delta,    alpha,    gamma) for legs (V, K1, K2, Q1, Q2).
    """
    if family == 'direct':
        return 0, (0,)
    if family == 'layer1':
        return 0, (1,)
    fam, rho = family
    assert fam == 'layer2'
    bits = [(rho >> i) & 1 for i in range(5)]
    return 1, (bits[4], bits[1], bits[3], bits[0], bits[2])


# ---------------------------------------------------------------------------
# Wick plan (topology-only, no symmetry to keep src axes per-leg consistent)
# ---------------------------------------------------------------------------

@cache
def _isserlis_plan_no_sym(legs_basic, device, dtype):
    """All matchings + correction term, with weight 1 each (no symmetry dedup).

    Symmetry dedup is unsafe here: orbit-related matchings produce result
    tensors that differ by a permutation of the per-leg src axes (each leg
    owns its own src axis). We therefore enumerate every matching.
    """
    n = len(legs_basic)
    all_m = matchings(tuple(range(n)))
    configs = tuple(all_m) + (tuple((i, i) for i in range(n)),)
    weights = (1.0,) * len(all_m) + (-(prod(range(1, n, 2)) - 1.0),)
    return configs, torch.tensor(weights, device=device, dtype=dtype)


def _master_moment(tl, tr, ml, mr, S):
    """One vectorized Wick contraction with per-leg src axes left open.

    Returns a tensor of shape `_OUT_shape + (2,) * n_legs` where the trailing
    axes index per-leg src bits in leg order (a-side legs first, then b-side).
    """
    # Strip any term-level symmetries: src-axis bookkeeping makes sym-dedup unsafe.
    tl = Term(tl.tn, tl.legs, symmetries=())
    tr = Term(tr.tn, tr.legs, symmetries=())
    tn, legs_basic, _syms = _join(tl, tr, ml, mr)
    n_legs = len(legs_basic)
    src_names = tuple(f'src:leg{i}' for i in range(n_legs))

    configs, weights = _isserlis_plan_no_sym(legs_basic, S.device, S.dtype)

    master = None
    out_inds = _OUT + src_names
    for cfg, w in zip(configs, weights.tolist()):
        bridges = []
        for i, j in cfg:
            a = legs_basic[i]
            b = legs_basic[j]
            if i == j:
                # mu-bridge: keep src axis open. shape (2, n_ctx, d+1).
                m = a[2]
                data = torch.stack([S[m, m, s, s, :, :, 0, 0] for s in range(2)])
                inds = (src_names[i],) + a[:2]
            else:
                # Cross bridge: keep (sl, sr) axes open. shape (2, 2, n_ctx, d+1, n_ctx, d+1).
                data = S[a[2], b[2]]
                inds = (src_names[i], src_names[j]) + a[:2] + b[:2]
            bridges.append((data, inds))
        contrib = bridged_contract(tn, bridges, out_inds)
        master = w * contrib if master is None else master + w * contrib
    return master


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def _check_attention_only(model):
    assert len(model.body) == 2, "This implementation handles 2-layer models only."
    for layer in model.body:
        assert layer.mlp.scale == 0.0, (
            "Path decomposition requires MLP scale=0 (passthrough)."
        )


def _stack_s_split(s_split, n_ctx, d_padded, like):
    """Pack {(ml, mr, sl, sr): tensor} into a single (2, 2, 2, 2, ...) tensor."""
    S = torch.empty(2, 2, 2, 2, n_ctx, d_padded, n_ctx, d_padded, **like)
    for (ml, mr, sl, sr), v in s_split.items():
        S[ml, mr, sl, sr] = v
    return S


@torch.no_grad()
def family_pair_inner_products(model_a, model_b):
    """Compute the 34*34 family-pair inner-product matrix.

    Returns:
      matrix : dict {(family_a, family_b): float}
      total  : float; sum of matrix entries.
    """
    _check_attention_only(model_a)
    _check_attention_only(model_b)

    state = _initial_state(model_a)
    n_ctx = state.s_aa.shape[0]
    like = dict(device=state.s_aa.device, dtype=state.s_aa.dtype)

    # --- Embed --- (this changes the trailing dim from d_in+1 to d_model+1)
    state = _step(state,
                  model_a.embed.terms(n_ctx, **like),
                  model_b.embed.terms(n_ctx, **like))
    d_padded = state.s_aa.shape[1]

    # --- Layer 1: split into 4 per-source sub-moments per pair ---
    # Strip term-level symmetries so the sub-moments match the no-symmetry
    # layered Wick (see NOTE.md: orbit dedup with the active term's joint
    # swap is not a per-matching invariance with general bridges).
    _ns = lambda ts: [Term(t.tn, t.legs, symmetries=()) for t in ts]
    ta1 = _ns(model_a.body[0].attn.terms(n_ctx, **like))
    tb1 = _ns(model_b.body[0].attn.terms(n_ctx, **like))
    assert len(ta1) == 2 and len(tb1) == 2

    sides = {0: ta1, 1: tb1}
    s_split = {}
    for ml in (0, 1):
        for mr in (0, 1):
            for sl in (0, 1):
                for sr in (0, 1):
                    s_split[(ml, mr, sl, sr)] = _moment(
                        sides[ml][sl], sides[mr][sr], ml, mr, state
                    )
    S = _stack_s_split(s_split, n_ctx, d_padded, like)
    # MLPs (scale=0) are passthrough.

    # --- Layer 2: master tensors per (term_type_a, term_type_b, ml, mr) ---
    ta2 = model_a.body[1].attn.terms(n_ctx, **like)
    tb2 = model_b.body[1].attn.terms(n_ctx, **like)
    assert len(ta2) == 2 and len(tb2) == 2

    # We'll only need (ml=0, mr=1) for s_ab.
    masters = {}
    for tta in (0, 1):
        for ttb in (0, 1):
            masters[(tta, ttb)] = _master_moment(ta2[tta], tb2[ttb], 0, 1, S)

    # --- Slice masters per family pair, then propagate through head + trace ---
    th_a = model_a.head.terms(n_ctx, **like)
    th_b = model_b.head.terms(n_ctx, **like)
    assert len(th_a) == 1 and len(th_b) == 1

    fams = list(enumerate_families())
    matrix = {}
    for fa in fams:
        tta, src_a = _family_to_tt_and_src(fa)
        for fb in fams:
            ttb, src_b = _family_to_tt_and_src(fb)
            m = masters[(tta, ttb)]
            # Slice trailing src axes at the family-specified bits.
            idx = (slice(None),) * 4 + tuple(src_a) + tuple(src_b)
            s_ab_l2 = m[idx]
            # Head + trace.
            proxy = State(s_ab_l2, s_ab_l2, s_ab_l2)
            s_ab_out = _moment(th_a[0], th_b[0], 0, 1, proxy)
            matrix[(fa, fb)] = torch.einsum('ijij->', s_ab_out[:, 1:, :, 1:]).item()
    return matrix, sum(matrix.values())
