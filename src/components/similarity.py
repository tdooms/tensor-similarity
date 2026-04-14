"""Exact Gaussian functional similarity via second-moment propagation.

See SIMILARITY.md for the math. `similarity` does all CPU setup up front;
`propagate` is a pure-GPU hot path safe for `torch.cuda.CUDAGraph` capture.
"""
from dataclasses import dataclass
from functools import cache
from math import prod
from pathlib import Path

import cotengra as ctg
import torch
from quimb.tensor import Tensor, TensorNetwork

_EXPR_CACHE = {}

_PATH_CACHE_DIR = Path.home() / '.cache' / 'tensor-mars' / 'ctg-paths'
_PATH_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_PATH_OPTIMIZER = ctg.ReusableHyperOptimizer(
    directory=str(_PATH_CACHE_DIR),
    minimize='combo',
    methods=('greedy', 'kahypar'),
    max_repeats=32,
    parallel=False,
    progbar=False,
)


def _contract(core_tn, bridge_specs, output_inds):
    """Contract `core_tn` with bridges given as `(data, inds)` tuples. Cached."""
    key = tuple(t.inds for t in core_tn) + tuple(inds for _, inds in bridge_specs)
    expr = _EXPR_CACHE.get(key)
    if expr is None:
        full = core_tn.copy()
        for data, inds in bridge_specs:
            full &= Tensor(data, inds=inds)
        expr = full.contract(output_inds=output_inds, optimize=_PATH_OPTIMIZER, get='expression')
        _EXPR_CACHE[key] = expr
    return expr(*(t.data for t in core_tn), *(data for data, _ in bridge_specs))


@dataclass
class State:
    """Second moments `S_xy = E[f_x(·) f_y(·)^T]` for model pair (a, b)."""
    S_aa: torch.Tensor
    S_ab: torch.Tensor
    S_bb: torch.Tensor

    @staticmethod
    def from_model(model):
        comps = model.components()
        n_ctx = next((c.n_ctx for c in comps if hasattr(c, 'n_ctx')), 1)
        d = comps[0].network().ind_size('in:d0') - 1
        p = next(model.parameters())
        eye_n = Tensor(torch.eye(n_ctx, device=p.device, dtype=p.dtype), inds=('_s', '_sp'))
        eye_d = Tensor(torch.eye(d + 1, device=p.device, dtype=p.dtype), inds=('_k', '_l'))
        S = (eye_n & eye_d).contract(output_inds=('_s', '_k', '_sp', '_l')).data
        return State(S, S, S)


@cache
def _matching_indices(n):
    """All perfect matchings of {0, …, n−1} as tuples of (i, j) pairs."""
    def build(xs):
        if not xs:
            return [()]
        return [((xs[0], xs[i]),) + rest
                for i in range(1, len(xs))
                for rest in build(xs[1:i] + xs[i + 1:])]
    return tuple(build(tuple(range(n))))


def _dedupe(matchings, group):
    """Group matchings by orbit under leg-name permutations `group`.
    Returns (rep, orbit_size) so Σ orbit_size · rep equals the full Wick sum."""
    def perm_pair(perm, a, b):
        pa = (a[0], perm.get(a[1], a[1]), a[2])
        pb = (b[0], perm.get(b[1], b[1]), b[2])
        return (pa, pb) if pa <= pb else (pb, pa)

    out = {}
    for m in matchings:
        k = min(tuple(sorted(perm_pair(p, a, b) for a, b in m)) for p in group)
        out[k] = (out[k][0], out[k][1] + 1) if k in out else (m, 1)
    return list(out.values())


@cache
def _isserlis_plan(legs, group):
    """Cold structural plan for an Isserlis sum — cached, purely combinatorial.

    Returns `(matchings, mu_sides, mu_inds, correction)` where `matchings` is a
    tuple of `(side_pairs, bridge_inds, weight)` triples, one per deduped Wick
    matching. The hot path only needs to index `S[side_pair]` for each pair.
    """
    matchings = [tuple((legs[i], legs[j]) for i, j in m)
                 for m in _matching_indices(len(legs))]
    deduped = _dedupe(matchings, [dict(g) for g in group])
    plan = tuple(
        (tuple((a[2], b[2]) for a, b in m),
         tuple(a[:2] + b[:2] for a, b in m),
         w)
        for m, w in deduped)
    mu_sides = tuple((l[2], l[2]) for l in legs)
    mu_inds = tuple(l[:2] for l in legs)
    correction = prod(range(1, len(legs), 2)) - 1
    return plan, mu_sides, mu_inds, correction


@cache
def _plan_weights(plan, device, dtype):
    """Device-resident weight vector — cached per (plan, device, dtype) so the
    hot path does zero host→device copies."""
    return torch.tensor([w for _, _, w in plan], device=device, dtype=dtype)


def _isserlis(tn, legs, S, output_inds, group=({},)):
    """Corrected Isserlis sum over Wick pairings. Hot path: pure tensor ops."""
    plan, mu_sides, mu_inds, correction = _isserlis_plan(
        tuple(legs), tuple(frozenset(g.items()) for g in group))

    contribs = torch.stack([
        _contract(tn, [(S[sp], inds) for sp, inds in zip(sps, inds_list)], output_inds)
        for sps, inds_list, _ in plan])
    weights = _plan_weights(plan, contribs.device, contribs.dtype)
    result = torch.einsum('i,i...->...', weights, contribs)

    mu_specs = [(S[sp][:, :, 0, 0], inds) for sp, inds in zip(mu_sides, mu_inds)]
    return result - correction * _contract(tn, mu_specs, output_inds)


def _second_moment(term_l, term_r, ml, mr, S):
    """E[term_l(·) term_r(·)^T]. Legs tagged with model indices (ml, mr); the
    TN prefix 'a'/'b' disambiguates left/right in the joint network."""
    tn = TensorNetwork(
        list(term_l.tn.reindex({i: f'a:{i}' for i in term_l.tn.ind_map}))
        + list(term_r.tn.reindex({i: f'b:{i}' for i in term_r.tn.ind_map})),
        check_collisions=False)

    legs = [(f'{p}:{pos}', f'{p}:{data}', m)
            for term, p, m in [(term_l, 'a', ml), (term_r, 'b', mr)]
            for data, pos in sorted(term.legs.items())]

    ga = [{f'a:{k}': f'a:{v}' for k, v in p.items()} for p in term_l.symmetries]
    gb = [{f'b:{k}': f'b:{v}' for k, v in p.items()} for p in term_r.symmetries]
    group = [{}, *ga, *gb, *({**a, **b} for a in ga for b in gb)]

    return _isserlis(tn, legs, S, ('a:out:s', 'a:out:d', 'b:out:s', 'b:out:d'), group)


def propagate(state, terms_a, terms_b):
    """Propagate second moments through one layer. Pure-GPU hot path.

    Leg ordering keeps (ml, mr) ∈ {(0,0), (0,1), (1,1)}, so no S[(1,0)] entry
    is ever needed.
    """
    S = {(0, 0): state.S_aa, (0, 1): state.S_ab, (1, 1): state.S_bb}

    def block(tl, tr, ml, mr):
        return sum(_second_moment(a, b, ml, mr, S) for a in tl for b in tr)

    return State(
        block(terms_a, terms_a, 0, 0),
        block(terms_a, terms_b, 0, 1),
        block(terms_b, terms_b, 1, 1),
    )


def similarity(model_a, model_b):
    """Exact Gaussian functional similarity between two models.

    CPU-side setup (build terms, find paths, compile plans) happens up front;
    the `propagate` loop is pure GPU and safe for CUDA-graph capture.
    """
    state = State.from_model(model_a)
    like = dict(device=state.S_aa.device, dtype=state.S_aa.dtype)
    n_ctx = state.S_aa.shape[0]
    layers = [(ca.terms(n_ctx, **like), cb.terms(n_ctx, **like))
              for ca, cb in zip(model_a.components(), model_b.components())]
    for terms_a, terms_b in layers:
        state = propagate(state, terms_a, terms_b)
    return state
