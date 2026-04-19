"""Exact Gaussian functional similarity via second-moment propagation.

Three sentences:
  (1) Σ = I_n ⊗ I_d — the Gaussian input covariance in padded representation.
  (2) For each component: Σ ← Σ_{l, r ∈ terms} E[l · rᵀ | x ~ N(·, Σ)]
      via Isserlis (deduped Wick matchings + μ-correction).
  (3) `cross(a, b)` is the primitive (rank-4); `gauge(a) = cross(a, a)` is a
      fast self-path; `similarity(a, b)` assembles the 2×2 block.

`cross(a, b)` for a ≠ b jointly propagates the triple (Σ_aa, Σ_ab, Σ_bb) —
the layer update for Σ_ab routes its bridges through Σ_aa/Σ_ab/Σ_bb based on
which side each leg came from. See SIMILARITY.md for the math.
"""
from functools import cache, partial
from math import prod

import torch
from quimb.tensor import Tensor

from src.components.utils import (
    bridged_contract, capture_cuda_graph, direct_product, matchings, orbits,
)

_OUT = ('l:out:s', 'l:out:d', 'r:out:s', 'r:out:d')
_GRAPHS: dict = {}


# ── Isserlis combinatorics ─────────────────────────────────────────────────

def _canon(matching, perm, legs):
    """Canonical form of a matching under a positional permutation."""
    return tuple(sorted(tuple(sorted((legs[perm[i]], legs[perm[j]])))
                         for i, j in matching))


@cache
def _isserlis_plan(legs, syms, device, dtype):
    """Deduped Wick matchings + all-μ correction, weights as a ready tensor."""
    wick = orbits(matchings(tuple(range(len(legs)))), syms, partial(_canon, legs=legs))
    configs = tuple(m for m, _ in wick) + (tuple((i, i) for i in range(len(legs))),)
    weights = tuple(w for _, w in wick) + (-(prod(range(1, len(legs), 2)) - 1),)
    return configs, torch.tensor(weights, device=device, dtype=dtype)


# ── joint TN for E[t_l · t_rᵀ] ────────────────────────────────────────────

def _sided(term, side):
    """Prefix term's indices with `side:` (l = left half of outer product, r = right)."""
    ren = lambda n: f'{side}:{n}'
    tn = term.tn.reindex({i: ren(i) for i in term.tn.ind_map})
    legs = tuple((ren(pos), ren(dat)) for dat, pos in sorted(term.legs.items()))
    return tn, legs, term.symmetries


def _join(term_l, term_r):
    """Joint TN, legs, combined symmetries."""
    tn_l, legs_l, sl = _sided(term_l, 'l')
    tn_r, legs_r, sr = _sided(term_r, 'r')
    return tn_l | tn_r, legs_l + legs_r, direct_product(sl, len(legs_l), sr, len(legs_r))


def _bridge(sigmas, leg_i, leg_j):
    """Attach the appropriate Σ between two legs.

    `sigmas` has keys ('l','l') → Σ_aa, ('l','r') → Σ_ab, ('r','r') → Σ_bb.
    Self-pair slices Σ at (pos_R=0, dat_R=0), yielding μ. The (r,l) combo
    reuses Σ_ab with swapped attach order (= Σ_ba transpose)."""
    side = lambda leg: 'l' if leg[0].startswith('l:') else 'r'
    si, sj = side(leg_i), side(leg_j)
    if leg_i == leg_j:
        return sigmas[(si, si)][..., 0, 0], leg_i
    if (si, sj) == ('r', 'l'):
        return sigmas[('l', 'r')], (*leg_j, *leg_i)
    return sigmas[(si, sj)], (*leg_i, *leg_j)


def _moment(term_l, term_r, sigmas):
    """E[t_l · t_rᵀ] via Isserlis + μ correction."""
    tn, legs, syms = _join(term_l, term_r)
    s = sigmas[('l', 'l')]
    configs, weights = _isserlis_plan(legs, syms, s.device, s.dtype)
    return sum(w * bridged_contract(
        tn, [_bridge(sigmas, legs[i], legs[j]) for i, j in c], _OUT)
        for c, w in zip(configs, weights))


def _step(sigmas, terms_l, terms_r):
    """Σ ← Σ_{l, r} E[l · rᵀ] over term pairs."""
    return sum(_moment(l, r, sigmas) for l in terms_l for r in terms_r)


def _self(sigma):
    """The three-slot sigmas dict where all slots are the same self-moment."""
    return {('l', 'l'): sigma, ('l', 'r'): sigma, ('r', 'r'): sigma}


# ── initial state ─────────────────────────────────────────────────────────

def _initial(model):
    """Σ = I_n ⊗ I_d, rank-4 (n, d+1, n, d+1)."""
    p = next(model.parameters())
    n = getattr(model, 'n_ctx', 1)
    d = model.components()[0].network().ind_size('in:d0')
    kit = dict(device=p.device, dtype=p.dtype)
    return (Tensor(torch.eye(n, **kit), inds=('l:out:s', 'r:out:s'))
          & Tensor(torch.eye(d, **kit), inds=('l:out:d', 'r:out:d'))
           ).contract(output_inds=_OUT).data


# ── core propagation ──────────────────────────────────────────────────────

def _gauge(model):
    """Self-propagation; returns rank-4 E[f · fᵀ]."""
    sigma = _initial(model)
    for c in model.components():
        sigma = _step(_self(sigma), c.terms(getattr(model, 'n_ctx', 1)), c.terms(getattr(model, 'n_ctx', 1)))
    return sigma


def _joint(a, b):
    """Propagate the triple (Σ_aa, Σ_ab, Σ_bb) through aligned components."""
    s_aa = s_ab = _initial(a)
    s_bb = _initial(b)
    for ca, cb in zip(a.components(), b.components()):
        ta = ca.terms(getattr(a, 'n_ctx', 1))
        tb = cb.terms(getattr(b, 'n_ctx', 1))
        new_aa = _step(_self(s_aa), ta, ta)
        new_ab = _step({('l','l'): s_aa, ('l','r'): s_ab, ('r','r'): s_bb}, ta, tb)
        new_bb = _step(_self(s_bb), tb, tb)
        s_aa, s_ab, s_bb = new_aa, new_ab, new_bb
    return s_aa, s_ab, s_bb


# ── CUDA graph wrapper ────────────────────────────────────────────────────

def _graphed(fn, models, name):
    """Run fn() with a CUDA graph cached per (arch, device, dtype, name)."""
    ps = tuple(p for m in models for p in m.parameters())
    if not ps or not ps[0].is_cuda:
        return fn()
    key = (tuple(p.shape for p in ps), ps[0].device, ps[0].dtype, name)
    if key not in _GRAPHS:
        _GRAPHS[key] = capture_cuda_graph(fn, ps)
    bufs, out, graph = _GRAPHS[key]
    for src, dst in zip(ps, bufs): dst.copy_(src.data)
    graph.replay()
    return tuple(o.clone() for o in out) if isinstance(out, tuple) else out.clone()


# ── public primitives ─────────────────────────────────────────────────────

@torch.no_grad()
def precompile(a, b=None):
    """Run the cold path up front: cotengra path optimization for every unique
    contraction topology this (a, b) will hit. Populates the on-disk path cache
    and the in-memory expression cache. After this, gauge/cross/similarity calls
    for the same (arch, device, dtype) are warm. Idempotent."""
    if b is None or a is b: _gauge(a)
    else:                   _joint(a, b)


@torch.no_grad()
def gauge(model):
    """E[f(x) · f(x)ᵀ] — self propagation, rank-4. First call per (arch, device,
    dtype) is cold; call `precompile(model)` up front to control when that happens."""
    return _graphed(lambda: _gauge(model), (model,), 'gauge')


@torch.no_grad()
def cross(a, b):
    """E[f_a(x) · f_b(x)ᵀ], rank-4. See `precompile` for cold/warm separation."""
    if a is b: return gauge(a)
    return _graphed(lambda: _joint(a, b), (a, b), 'joint')[1]


@torch.no_grad()
def similarity(a, b):
    """2×2 block `[[⟨a,a⟩, ⟨a,b⟩], [⟨a,b⟩ᵀ, ⟨b,b⟩]]`, shape (2, 2, n, d+1, n, d+1).
    See `precompile` for cold/warm separation."""
    if a is b:
        s = gauge(a)
        return torch.stack([torch.stack([s, s]), torch.stack([s, s])])
    s_aa, s_ab, s_bb = _graphed(lambda: _joint(a, b), (a, b), 'joint')
    s_ba = s_ab.permute(2, 3, 0, 1)
    return torch.stack([torch.stack([s_aa, s_ab]), torch.stack([s_ba, s_bb])])
