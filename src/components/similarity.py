"""Exact Gaussian functional similarity via second-moment propagation.

See SIMILARITY.md for the math and architecture notes.
"""
from dataclasses import dataclass
from functools import cache
from math import prod
from pathlib import Path

import cotengra as ctg
import torch
from quimb.tensor import Tensor, TensorNetwork

_CACHE_DIR = Path.home() / '.cache' / 'tensor-mars' / 'ctg-paths'
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_OPT = ctg.ReusableHyperOptimizer(
    directory=str(_CACHE_DIR), minimize='combo',
    methods=('greedy', 'kahypar'), max_repeats=32,
    parallel=False, progbar=False,
)
_EXPRS: dict = {}
_GRAPHS: dict = {}

_OUT = ('a:out:s', 'a:out:d', 'b:out:s', 'b:out:d')


@dataclass
class State:
    """Block second moments `s_xy = E[f_x f_y^T]` for model pair (a, b)."""
    s_aa: torch.Tensor
    s_ab: torch.Tensor
    s_bb: torch.Tensor


@cache
def _matchings(xs):
    """All perfect matchings of `xs` as tuples of (a, b) pairs."""
    if not xs: return ((),)
    return tuple(((xs[0], xs[i]),) + r
                 for i in range(1, len(xs))
                 for r in _matchings(xs[1:i] + xs[i + 1:]))


@cache
def _plan(legs, syms, device, dtype):
    """Deduped matchings + precomputed weights tensor + overcounting correction."""
    def canon(m, perm):
        p = dict(perm)
        sub = lambda l: (l[0], p.get(l[1], l[1]), l[2])
        return tuple(sorted(tuple(sorted((sub(a), sub(b)))) for a, b in m))
    orbits: dict = {}
    for m in _matchings(legs):
        k = min(canon(m, s) for s in syms)
        orbits[k] = (orbits[k][0], orbits[k][1] + 1) if k in orbits else (m, 1)
    matchings, weights = zip(*orbits.values())
    return (matchings,
            torch.tensor(weights, device=device, dtype=dtype),
            prod(range(1, len(legs), 2)) - 1)


def _contract(core, bridges, out):
    """Contract `core` with `(data, inds)` bridges. Compiled expression cached."""
    key = tuple(t.inds for t in core) + tuple(inds for _, inds in bridges)
    if key not in _EXPRS:
        tn = core.copy()
        for data, inds in bridges: tn &= Tensor(data, inds=inds)
        _EXPRS[key] = tn.contract(output_inds=out, optimize=_OPT, get='expression')
    return _EXPRS[key](*(t.data for t in core), *(data for data, _ in bridges))


def _moment(tl, tr, ml, mr, s):
    """E[term_l · term_r^T] via Isserlis sum with (2N−1)!!−1 all-mu correction."""
    sides = [(tl, 'a', ml), (tr, 'b', mr)]
    tn = TensorNetwork(
        [t for term, p, _ in sides
           for t in term.tn.reindex({i: f'{p}:{i}' for i in term.tn.ind_map})],
        check_collisions=False)
    legs = tuple((f'{p}:{pos}', f'{p}:{dat}', m)
                 for term, p, m in sides
                 for dat, pos in sorted(term.legs.items()))
    sym = lambda t, p: tuple(tuple(sorted((f'{p}:{k}', f'{p}:{v}')
                                          for k, v in d.items())) for d in t.symmetries)
    sl, sr = sym(tl, 'a'), sym(tr, 'b')
    syms = ((), *sl, *sr, *(a + b for a in sl for b in sr))

    block = {(0, 0): s.s_aa, (0, 1): s.s_ab, (1, 1): s.s_bb}
    bridge = lambda a, b: (block[tuple(sorted((a[2], b[2])))], a[:2] + b[:2])

    matchings, weights, correction = _plan(legs, syms, s.s_aa.device, s.s_aa.dtype)
    contribs = torch.stack([_contract(tn, [bridge(*p) for p in m], _OUT) for m in matchings])
    wick = torch.einsum('i,i...->...', weights, contribs)
    mu = [(block[l[2], l[2]][:, :, 0, 0], l[:2]) for l in legs]
    return wick - correction * _contract(tn, mu, _OUT)


def _run(model_a, model_b):
    """The algorithm: S = I⊗I, then propagate through layers via term-pair moments."""
    comps = model_a.components()
    n = next((c.n_ctx for c in comps if hasattr(c, 'n_ctx')), 1)
    d = comps[0].network().ind_size('in:d0')
    p = next(model_a.parameters())
    eye = lambda k: torch.eye(k, device=p.device, dtype=p.dtype)
    s0 = torch.einsum('ij,kl->ikjl', eye(n), eye(d))
    s = State(s0, s0, s0)
    for ca, cb in zip(comps, model_b.components()):
        ta = ca.terms(n, device=p.device, dtype=p.dtype)
        tb = cb.terms(n, device=p.device, dtype=p.dtype)
        block = lambda tl, tr, ml, mr: sum(_moment(x, y, ml, mr, s) for x in tl for y in tr)
        s = State(block(ta, ta, 0, 0), block(ta, tb, 0, 1), block(tb, tb, 1, 1))
    return s


# --- CUDA-graph capture (JAX-like implicit cache) ---------------------------

@dataclass
class _Graphed:
    params: tuple
    state: State
    graph: "torch.cuda.CUDAGraph"


def _capture(a, b):
    """Warm up and capture `_run(a, b)` with static parameter buffers."""
    ps = (*a.parameters(), *b.parameters())
    bufs = tuple(p.data.clone() for p in ps)
    orig = [p.data for p in ps]
    for p, buf in zip(ps, bufs): p.data = buf
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): _run(a, b)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): state = _run(a, b)
    finally:
        for p, o in zip(ps, orig): p.data = o
    return _Graphed(bufs, state, graph)


def similarity(model_a, model_b):
    """Exact Gaussian functional similarity between two models.

    On CUDA, captures a graph on first call per (architecture, device, dtype);
    subsequent calls copy weights into static buffers and replay. CPU models
    fall through to the direct path.
    """
    if not next(model_a.parameters()).is_cuda: return _run(model_a, model_b)
    ps = (*model_a.parameters(), *model_b.parameters())
    key = (tuple(p.shape for p in ps), ps[0].device, ps[0].dtype)
    g = _GRAPHS.get(key) or _GRAPHS.setdefault(key, _capture(model_a, model_b))
    for src, dst in zip(ps, g.params): dst.copy_(src.data)
    g.graph.replay()
    return State(g.state.s_aa.clone(), g.state.s_ab.clone(), g.state.s_bb.clone())
