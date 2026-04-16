"""Exact Gaussian functional similarity via second-moment propagation.

See SIMILARITY.md for the math and architecture notes.
"""
from dataclasses import dataclass
from functools import cache
from math import prod

import torch

from src.components.utils import (
    bridged_contract, capture_cuda_graph, direct_product, matchings, orbits, prefix,
)

_GRAPHS: dict = {}
_OUT = ('a:out:s', 'a:out:d', 'b:out:s', 'b:out:d')


@dataclass
class State:
    """Block second moments `s_xy = E[f_x f_y^T]` for model pair (a, b)."""
    s_aa: torch.Tensor
    s_ab: torch.Tensor
    s_bb: torch.Tensor

    def __getitem__(self, ij):
        return {(0, 0): self.s_aa, (0, 1): self.s_ab,
                (1, 0): self.s_ab, (1, 1): self.s_bb}[ij]


@cache
def _isserlis_plan(legs, syms, device, dtype):
    """Weighted bridge configurations for E[Π legs] in padded representation.

    A `config` is a tuple of pairings `(i, j)` over leg indices:
      • `i != j` → S-bridge on legs (i, j)
      • `i == j` → μ-bridge on leg i alone (S-slice at the constant-1 axes)

    Wick matchings each appear once with weight = orbit size under `syms`.
    One extra all-self-pair config appears with weight = −((2N−1)!!−1),
    correcting the padded-rep overcounting of the all-μ term.
    """
    def canon(m, perm):
        p = dict(perm)
        sub = lambda l: (l[0], p.get(l[1], l[1]), l[2])
        return tuple(sorted(tuple(sorted((sub(legs[i]), sub(legs[j])))) for i, j in m))
    wick = orbits(matchings(tuple(range(len(legs)))), syms, canon)
    configs = tuple(m for m, _ in wick) + (tuple((i, i) for i in range(len(legs))),)
    # (2N−1)!! − 1 corrects the padded-rep overcounting of the all-μ term.
    weights = tuple(w for _, w in wick) + (-(prod(range(1, len(legs), 2)) - 1),)
    return configs, torch.tensor(weights, device=device, dtype=dtype)


def _join(tl, tr, ml, mr):
    """Joint TN + tagged external legs + combined symmetry group."""
    tl, tr = prefix(tl, 'a'), prefix(tr, 'b')
    tag = lambda term, m: tuple((pos, dat, m) for dat, pos in sorted(term.legs.items()))
    return tl.tn | tr.tn, tag(tl, ml) + tag(tr, mr), direct_product(tl.symmetries, tr.symmetries)


def _bridges_for(config, legs, s):
    """(data, inds) list for one config: S-bridge if i != j, μ-bridge if i == j."""
    def one(i, j):
        a, b = legs[i], legs[j]
        if i == j: return s[a[2], a[2]][:, :, 0, 0], a[:2]
        return s[a[2], b[2]], a[:2] + b[:2]
    return [one(i, j) for i, j in config]


def _moment(tl, tr, ml, mr, s):
    """E[term_l · term_r^T] as a single weighted sum over Wick + μ configs."""
    tn, legs, syms = _join(tl, tr, ml, mr)
    configs, weights = _isserlis_plan(legs, syms, s.s_aa.device, s.s_aa.dtype)
    contribs = torch.stack([bridged_contract(tn, _bridges_for(c, legs, s), _OUT) for c in configs])
    return torch.einsum('i,i...->...', weights, contribs)


def _initial_state(model):
    """S = I_n ⊗ I_d, the second moment of the padded Gaussian input."""
    comps = model.components()
    n = next((c.n_ctx for c in comps if hasattr(c, 'n_ctx')), 1)
    d = comps[0].network().ind_size('in:d0')
    p = next(model.parameters())
    s0 = torch.einsum('ij,kl->ikjl',
                      torch.eye(n, device=p.device, dtype=p.dtype),
                      torch.eye(d, device=p.device, dtype=p.dtype))
    return State(s0, s0, s0)


def _step(s, ta, tb):
    """Propagate state through one layer pair via the 3 block moment sums."""
    return State(
        sum(_moment(x, y, 0, 0, s) for x in ta for y in ta),
        sum(_moment(x, y, 0, 1, s) for x in ta for y in tb),
        sum(_moment(x, y, 1, 1, s) for x in tb for y in tb),
    )


@torch.no_grad()
def _run(model_a, model_b):
    """The algorithm: S = I⊗I, then propagate through layers via term-pair moments."""
    state = _initial_state(model_a)
    for ca, cb in zip(model_a.components(), model_b.components()):
        n = state.s_aa.shape[0]
        device, dtype = state.s_aa.device, state.s_aa.dtype
        ta = ca.terms(n, device=device, dtype=dtype)
        tb = cb.terms(n, device=device, dtype=dtype)
        state = _step(state, ta, tb)
    return state


# --- CUDA-graph capture (JAX-like implicit cache) ---------------------------

def similarity(model_a, model_b):
    """Exact Gaussian functional similarity between two models.

    On CUDA, captures a graph on first call per (architecture, device, dtype);
    subsequent calls copy weights into static buffers and replay. CPU models
    fall through to the direct path.
    """
    if not next(model_a.parameters()).is_cuda: return _run(model_a, model_b)
    ps = (*model_a.parameters(), *model_b.parameters())
    key = (tuple(p.shape for p in ps), ps[0].device, ps[0].dtype)
    if key not in _GRAPHS:
        _GRAPHS[key] = capture_cuda_graph(lambda: _run(model_a, model_b), ps)
    bufs, out, graph = _GRAPHS[key]
    for src, dst in zip(ps, bufs): dst.copy_(src.data)
    graph.replay()
    return State(out.s_aa.clone(), out.s_ab.clone(), out.s_bb.clone())
