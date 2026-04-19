"""Exact Gaussian functional similarity via second-moment propagation.

Σ = I_n ⊗ I_d starts as the Gaussian input covariance in the padded rep.
Per component: Σ ← Σ_{l, r ∈ terms} E[l · rᵀ | x ~ N(·, Σ)] via Isserlis
(deduped Wick matchings + μ-correction). For a ≠ b we jointly propagate
the triple (Σ_aa, Σ_ab, Σ_bb); the Σ_ab update routes each bridge by leg
side. Self-pairs (Wick singletons) slice Σ at (pos=0, dat=0) — the padded
constant-1 reference — giving μ. See SIMILARITY.md for the math.
"""
from functools import cache, partial
from math import prod
from pathlib import Path

import cotengra as ctg
import torch
from quimb.tensor import Tensor

from src.components.utils import (
    capture_cuda_graph, direct_product, matchings, orbits,
)

_OUT = ('l:out:s', 'l:out:d', 'r:out:s', 'r:out:d')

_CACHE_DIR = Path.home() / '.cache' / 'tensor-mars' / 'ctg-paths'
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_OPT = ctg.ReusableHyperOptimizer(
    directory=str(_CACHE_DIR), minimize='size',
    methods=('greedy', 'kahypar'), max_repeats=32,
    parallel=False, progbar=False,
)
_PATHS: dict = {}     # cotengra exprs, keyed by (core_inds, bridge_inds)
_GRAPHS: dict = {}    # CUDA-graph captures, keyed by (shapes, device, dtype)


def _side(leg): return 'l' if leg[0].startswith('l:') else 'r'


# ── Isserlis combinatorics ────────────────────────────────────────────────

def _canon(matching, perm, legs):
    return tuple(sorted(tuple(sorted((legs[perm[i]], legs[perm[j]])))
                         for i, j in matching))


@cache
def _plan(legs, syms):
    """Deduped Wick matchings + μ-correction config. Pure Python."""
    wick = orbits(matchings(tuple(range(len(legs)))), syms, partial(_canon, legs=legs))
    configs = tuple(m for m, _ in wick) + (tuple((i, i) for i in range(len(legs))),)
    weights = tuple(w for _, w in wick) + (-(prod(range(1, len(legs), 2)) - 1),)
    return configs, weights


@cache
def _weights(weights_py, device, dtype):
    return torch.tensor(weights_py, device=device, dtype=dtype)


# ── joint TN E[t_l · t_rᵀ] ────────────────────────────────────────────────

def _sided(term, side):
    """Prefix every ind with `side:` (l = left half of t_l·t_rᵀ, r = right)."""
    tn = term.tn.reindex({i: f'{side}:{i}' for i in term.tn.ind_map})
    legs = tuple((f'{side}:{p}', f'{side}:{d}') for d, p in sorted(term.legs.items()))
    return tn, legs, term.symmetries


def _join(tl, tr):
    a, la, sa = _sided(tl, 'l')
    b, lb, sb = _sided(tr, 'r')
    return a | b, la + lb, direct_product(sa, len(la), sb, len(lb))


def _slots(aa, ab, bb):
    """Four (side_L, side_R) → Σ slots; Σ_ba = Σ_ab with L/R axes swapped."""
    return {('l', 'l'): aa, ('l', 'r'): ab,
            ('r', 'l'): ab.permute(2, 3, 0, 1), ('r', 'r'): bb}


def _bridge(slots, i, j):
    """Σ + attach-inds for a Wick pair. Self-pair sliced at (0, 0) = μ."""
    k = (_side(i), _side(j))
    return (slots[k][..., 0, 0], i) if i == j else (slots[k], (*i, *j))


def _contract(core, bridges):
    """Cache compiled cotengra expr per (core_inds, bridge_inds); fresh data each call
    (`Attention.network()` rebuilds tensors, so per-term data capture can't work)."""
    key = tuple(t.inds for t in core) + tuple(inds for _, inds in bridges)
    if key not in _PATHS:
        tn = core.copy()
        for data, inds in bridges: tn &= Tensor(data, inds=inds)
        _PATHS[key] = tn.contract(output_inds=_OUT, optimize=_OPT, get='expression')
    return _PATHS[key](*(t.data for t in core), *(data for data, _ in bridges))


def _moment(tl, tr, slots):
    """E[t_l · t_rᵀ] via Isserlis + μ correction."""
    tn, legs, syms = _join(tl, tr)
    configs, weights_py = _plan(legs, syms)
    s = slots[('l', 'l')]
    ws = _weights(weights_py, s.device, s.dtype)
    return sum(w * _contract(tn, [_bridge(slots, legs[i], legs[j]) for i, j in c])
               for c, w in zip(configs, ws))


def _update(slots, ts_l, ts_r):
    """Σ ← Σ_{l, r} E[l · rᵀ]."""
    return sum(_moment(l, r, slots) for l in ts_l for r in ts_r)


# ── propagation ───────────────────────────────────────────────────────────

def _initial(model):
    """Σ = I_n ⊗ I_d in the padded rep; rank-4 (n, d+1, n, d+1)."""
    p = next(model.parameters())
    d = model.components()[0].network().ind_size('in:d0')
    like = dict(device=p.device, dtype=p.dtype)
    return (Tensor(torch.eye(model.n_ctx, **like), inds=('l:out:s', 'r:out:s'))
          & Tensor(torch.eye(d, **like), inds=('l:out:d', 'r:out:d'))
           ).contract(output_inds=_OUT).data


def _propagate_self(m):
    """Σ ← self-update through each component. Returns `(σ, σ, σ)` for uniformity."""
    sigma = _initial(m)
    for c in m.components():
        ts = c.terms(m.n_ctx)
        sigma = _update(_slots(sigma, sigma, sigma), ts, ts)
    return (sigma,) * 3


def _propagate_cross(a, b):
    """Joint (Σ_aa, Σ_ab, Σ_bb) through aligned components; 3 updates per layer."""
    aa = ab = _initial(a)
    bb = _initial(b)
    for ca, cb in zip(a.components(), b.components()):
        ta, tb = ca.terms(a.n_ctx), cb.terms(b.n_ctx)
        aa, ab, bb = (_update(_slots(aa, aa, aa), ta, ta),
                      _update(_slots(aa, ab, bb), ta, tb),
                      _update(_slots(bb, bb, bb), tb, tb))
    return aa, ab, bb


# ── CUDA graph wrapper ────────────────────────────────────────────────────

def _graphed(fn, *models):
    """Cache a CUDA graph keyed by (param shapes, device, dtype)."""
    ps = tuple(p for m in models for p in m.parameters())
    if not ps[0].is_cuda: return fn()
    key = (tuple(p.shape for p in ps), ps[0].device, ps[0].dtype)
    if key not in _GRAPHS:
        _GRAPHS[key] = capture_cuda_graph(fn, ps)
    bufs, out, graph = _GRAPHS[key]
    for src, dst in zip(ps, bufs): dst.copy_(src.data)
    graph.replay()
    return tuple(o.clone() for o in out)


# ── public API ────────────────────────────────────────────────────────────

@torch.no_grad()
def similarity(a, b):
    """2×2 block `[[⟨a,a⟩, ⟨a,b⟩], [⟨a,b⟩ᵀ, ⟨b,b⟩]]`, shape (2, 2, n, d+1, n, d+1)."""
    aa, ab, bb = (_graphed(lambda: _propagate_self(a), a) if a is b
                  else _graphed(lambda: _propagate_cross(a, b), a, b))
    ba = ab.permute(2, 3, 0, 1)   # Σ_ba = E[b · aᵀ] = Σ_ab with L/R axes swapped
    return torch.stack([torch.stack([aa, ab]), torch.stack([ba, bb])])
