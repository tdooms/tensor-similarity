"""Exact Gaussian functional similarity via second-moment propagation.

Three sentences:
  (1) Σ = I_n ⊗ I_d — the Gaussian input covariance in the padded rep.
  (2) For each component: Σ ← Σ_{l, r ∈ _Stacked(a, b).terms} E[l · rᵀ | x ~ N(·, Σ)]
      via Isserlis (deduped Wick matchings + μ-correction).
  (3) Returns Σ of shape `(2, 2, n, d+1, n, d+1)`; `s[i, j]` is the cross-moment
      between model i and j (0=a, 1=b). See SIMILARITY.md for the math.

Bridges attach Σ with its (m_L, m_R) axes renamed to each leg's side label.
Same-side legs share that label — quimb turns the repeat-ind into a diagonal.
Self-pairs (Isserlis singletons) slice Σ's right half at (pos=0, dat=0), the
padded constant-1 reference; μ falls out as a view, no materialization.
"""
from functools import cache, partial
from math import prod

import torch
from quimb.tensor import Tensor, TensorNetwork

from src.components.base import Term
from src.components.utils import (
    bridged_contract, capture_cuda_graph, direct_product, matchings, orbits,
)

_OUT = ('m_l', 'm_r', 'l:out:s', 'l:out:d', 'r:out:s', 'r:out:d')
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
    """Deduped Wick matchings + all-μ correction, weights as a ready tensor."""
    wick = orbits(matchings(tuple(range(len(legs)))), syms, partial(_canon, legs=legs))
    configs = tuple(m for m, _ in wick) + (tuple((i, i) for i in range(len(legs))),)
    weights = tuple(w for _, w in wick) + (-(prod(range(1, len(legs), 2)) - 1),)
    return configs, torch.tensor(weights, device=device, dtype=dtype)


# ── joint TN + Σ bridges ───────────────────────────────────────────────────

def _sided(term, side):
    """Prefix term's inds with `side:`; rename `m` (if present) → `m_{side}`."""
    ren = lambda n: f'm_{side}' if n == 'm' else f'{side}:{n}'
    tn = term.tn.reindex({i: ren(i) for i in term.tn.ind_map})
    legs = tuple((ren(pos), ren(dat)) for dat, pos in sorted(term.legs.items()))
    syms = tuple({ren(k): ren(v) for k, v in d.items()} for d in term.symmetries)
    return tn, legs, syms


def _join(term_l, term_r):
    """Joint TN, legs, combined symmetries for E[t_l · t_rᵀ]."""
    tn_l, legs_l, sl = _sided(term_l, 'l')
    tn_r, legs_r, sr = _sided(term_r, 'r')
    return tn_l | tn_r, legs_l + legs_r, direct_product(sl, sr)


def _bridge(sigma, leg_i, leg_j):
    """Σ between two (possibly equal) legs, with m-axes named per side.

    Self-pair: slice Σ at (pos_R=0, dat_R=0) — the constant-1 reference — and
    attach the remaining (m_L, m_R, pos_L, dat_L). m_L == m_R by construction,
    so quimb collapses the repeat ind to a diagonal → that's μ."""
    m = lambda leg: 'm_l' if leg[0].startswith('l:') else 'm_r'
    if leg_i == leg_j:
        return sigma[..., 0, 0], (m(leg_i), m(leg_j), *leg_i)
    return sigma, (m(leg_i), m(leg_j), *leg_i, *leg_j)


# ── core: inner product + fold ────────────────────────────────────────────

def _moment(term_l, term_r, sigma):
    """E[t_l · t_rᵀ | x ~ N(·, Σ)] via Isserlis + μ correction."""
    tn, legs, syms = _join(term_l, term_r)
    configs, weights = _isserlis_plan(legs, syms, sigma.device, sigma.dtype)
    return sum(w * bridged_contract(
        tn, [_bridge(sigma, legs[i], legs[j]) for i, j in c], _OUT)
        for c, w in zip(configs, weights))


def _fold(model, sigma):
    """Σ ← Σ_{l, r ∈ terms} E[l · rᵀ] folded over components."""
    for c in model.components():
        terms = c.terms(model.n_ctx)
        sigma = sum(_moment(l, r, sigma) for l in terms for r in terms)
    return sigma


# ── stacked wrapper: two models with a per-tensor m axis ──────────────────

class _Stacked:
    """Treat (a, b) as one meta-model. Each component yields m-stacked terms."""
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.n_ctx = getattr(a, 'n_ctx', 1)
    def components(self):
        return [_StackedComponent(ca, cb)
                for ca, cb in zip(self.a.components(), self.b.components())]


class _StackedComponent:
    def __init__(self, ca, cb): self.ca, self.cb = ca, cb
    def terms(self, n):
        return [_stack(ta, tb)
                for ta, tb in zip(self.ca.terms(n), self.cb.terms(n))]


def _stack(ta, tb):
    """Stack two equivalent terms: every tensor gains a leading `m` axis."""
    tensors = [Tensor(torch.stack([a.data, b.data]), inds=('m', *a.inds), tags=a.tags)
               for a, b in zip(ta.tn, tb.tn)]
    return Term(TensorNetwork(tensors), ta.legs, ta.symmetries)


def _initial(model):
    """Σ = I_n ⊗ I_d ⊗ ones(m_l, m_r) in the padded rep."""
    p = next(model.parameters())
    n = getattr(model, 'n_ctx', 1)
    d = model.components()[0].network().ind_size('in:d0')
    kit = dict(device=p.device, dtype=p.dtype)
    return (Tensor(torch.eye(n, **kit),     inds=('l:out:s', 'r:out:s'))
          & Tensor(torch.eye(d, **kit),     inds=('l:out:d', 'r:out:d'))
          & Tensor(torch.ones(2, 2, **kit), inds=('m_l', 'm_r'))
           ).contract(output_inds=_OUT).data


@torch.no_grad()
def _run(a, b):
    return _fold(_Stacked(a, b), _initial(a))


# ── public API ────────────────────────────────────────────────────────────

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
