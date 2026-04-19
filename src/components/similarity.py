"""Exact Gaussian functional similarity via second-moment propagation.

Three sentences:
  (1) S = I_n ⊗ I_d, the Gaussian input covariance in padded representation.
  (2) For each layer, S ← Σ_{t_l, t_r in term pairs} E[t_l · t_rᵀ | x ~ N(0, S)],
      expanded via Isserlis' theorem (Wick matchings + μ-correction).
  (3) Return S, shape `(2, 2, n, d+1, n, d+1)`. Index `s[i, j]` for the
      cross-moment between model i and j (where 0=a, 1=b).

One public function: `similarity(a, b)`. `gauge(m)` is a one-liner alias.
See SIMILARITY.md for the full math.
"""
from functools import cache, partial
from math import prod

import torch
from quimb.tensor import Tensor, TensorNetwork

from src.components.utils import (
    bridged_contract, capture_cuda_graph, direct_product, matchings, orbits,
)

_OUT = ('m_l', 'm_r', 'a:out:s', 'a:out:d', 'b:out:s', 'b:out:d')
_GRAPHS: dict = {}


# ── Isserlis combinatorics ─────────────────────────────────────────────────

def _canon(matching, perm, legs):
    """Canonical form of a matching under a leg-data permutation."""
    p = dict(perm)
    sub = lambda l: (l[0], p.get(l[1], l[1]))
    return tuple(sorted(tuple(sorted((sub(legs[i]), sub(legs[j]))))
                         for i, j in matching))


@cache
def _isserlis_plan(legs, syms, device, dtype):
    """Deduped Wick matchings + the all-μ correction, with per-config weight."""
    wick = orbits(matchings(tuple(range(len(legs)))), syms, partial(_canon, legs=legs))
    configs = tuple(m for m, _ in wick) + (tuple((i, i) for i in range(len(legs))),)
    weights = tuple(w for _, w in wick) + (-(prod(range(1, len(legs), 2)) - 1),)
    return configs, torch.tensor(weights, device=device, dtype=dtype)


# ── joint TN construction ──────────────────────────────────────────────────

def _stack_side(pair, side, m_axis):
    """Stack a pair of equivalent-structure terms along `m_axis`, prefix all
    other indices with `side:`. Returns `(tn, legs, symmetries)`."""
    ta, tb = pair
    tensors = [Tensor(torch.stack([a.data, b.data]), inds=(m_axis, *a.inds), tags=a.tags)
               for a, b in zip(ta.tn, tb.tn)]
    non_m = {i for t in tensors for i in t.inds if i != m_axis}
    tn = TensorNetwork(tensors).reindex({i: f'{side}:{i}' for i in non_m})
    legs = tuple((f'{side}:{pos}', f'{side}:{dat}')
                 for dat, pos in sorted(ta.legs.items()))
    syms = tuple({f'{side}:{k}': f'{side}:{v}' for k, v in d.items()}
                 for d in ta.symmetries)
    return tn, legs, syms


def _join(pair_l, pair_r):
    """Joint TN, legs, and combined symmetry group from two paired terms."""
    tn_l, legs_l, sl = _stack_side(pair_l, 'a', 'm_l')
    tn_r, legs_r, sr = _stack_side(pair_r, 'b', 'm_r')
    return tn_l | tn_r, legs_l + legs_r, direct_product(sl, sr)


def _bridges(config, legs, s, diag, mu):
    """Per-pair bridges for one Isserlis config:
        self-pair (i==j) → μ on that leg's m-axis
        same-side        → diagonal of S on that side's m-axis
        cross-side       → full S on (m_l, m_r).
    """
    side = lambda leg: 'm_l' if leg[0].startswith('a:') else 'm_r'
    out = []
    for i, j in config:
        a, b = legs[i], legs[j]
        if i == j:                 out.append((mu,   (side(a), *a)))
        elif side(a) == side(b):   out.append((diag, (side(a), *a, *b)))
        else:                      out.append((s,    ('m_l', 'm_r', *a, *b)))
    return out


# ── the three-sentence algorithm ───────────────────────────────────────────

def _moment(pair_l, pair_r, s, diag, mu):
    """E[t_l · t_rᵀ | x ~ N(0, S)] via the Isserlis sum + μ correction."""
    tn, legs, syms = _join(pair_l, pair_r)
    configs, weights = _isserlis_plan(legs, syms, s.device, s.dtype)
    contribs = torch.stack([bridged_contract(tn, _bridges(c, legs, s, diag, mu), _OUT)
                            for c in configs])
    return (Tensor(contribs, inds=('i', *_OUT))
            & Tensor(weights, inds=('i',))).contract(output_inds=_OUT).data


def _initial_state(model):
    """S = I_n ⊗ I_d broadcast over the `(m_l, m_r)` block axes. Returns `(S, n)`."""
    p = next(model.parameters())
    n = getattr(model, 'n_ctx', 1)
    d = model.components()[0].network().ind_size('in:d0')
    kit = dict(device=p.device, dtype=p.dtype)
    s = (Tensor(torch.eye(n, **kit),     inds=('a:out:s', 'b:out:s'))
       & Tensor(torch.eye(d, **kit),     inds=('a:out:d', 'b:out:d'))
       & Tensor(torch.ones(2, 2, **kit), inds=('m_l', 'm_r'))
        ).contract(output_inds=_OUT).data
    return s, n


@torch.no_grad()
def _run(a, b):
    s, n = _initial_state(a)
    for ca, cb in zip(a.components(), b.components()):
        diag = torch.stack([s[0, 0], s[1, 1]])
        mu = diag[:, :, :, 0, 0]
        pairs = list(zip(ca.terms(n), cb.terms(n)))
        s = sum(_moment(l, r, s, diag, mu) for l in pairs for r in pairs)
    return s


# ── public API ─────────────────────────────────────────────────────────────

def similarity(a, b):
    """Exact Gaussian functional similarity, shape `(2, 2, n, d+1, n, d+1)`.
    On CUDA, captures a graph on first call per `(arch, device, dtype)`."""
    if not next(a.parameters()).is_cuda:
        return _run(a, b)
    ps = (*a.parameters(), *b.parameters())
    key = (tuple(p.shape for p in ps), ps[0].device, ps[0].dtype)
    if key not in _GRAPHS:
        _GRAPHS[key] = capture_cuda_graph(lambda: _run(a, b), ps)
    bufs, out, graph = _GRAPHS[key]
    for src, dst in zip(ps, bufs): dst.copy_(src.data)
    graph.replay()
    return out.clone()


def gauge(model):
    """Rank-4 `E[f(x) · f(x)ᵀ]` — canonical form for global SVD."""
    return similarity(model, model)[0, 0]
