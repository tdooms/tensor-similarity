"""Generic helpers — TN contraction with parametric bridges, combinatorics,
group-theoretic orbit finding, CUDA-graph capture, and term utilities."""
import os
from functools import cache
from pathlib import Path

import cotengra as ctg
import torch
from quimb.tensor import Tensor

from src.components.base import Term


_CACHE_DIR = Path(
    os.environ.get(
        "TENSOR_MARS_CTG_CACHE_DIR",
        Path.home() / ".cache" / "tensor-mars" / "ctg-paths",
    )
)
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_OPT = ctg.ReusableHyperOptimizer(
    directory=str(_CACHE_DIR), minimize='size',
    methods=('greedy', 'kahypar'), max_repeats=32,
    parallel=False, progbar=False,
)
_EXPRS: dict = {}


def bridged_contract(core, bridges, output_inds):
    """Contract `core` TN bridged with parametric `(data, inds)` tensors.

    Caches the compiled expression per unique (core_inds, bridge_inds)
    signature, so repeated calls with the same topology but different tensor
    data just invoke the closure.
    """
    key = tuple(t.inds for t in core) + tuple(inds for _, inds in bridges)
    if key not in _EXPRS:
        tn = core.copy()
        for data, inds in bridges: tn &= Tensor(data, inds=inds)
        _EXPRS[key] = tn.contract(output_inds=output_inds, optimize=_OPT, get='expression')
    return _EXPRS[key](*(t.data for t in core), *(data for data, _ in bridges))


@cache
def matchings(xs):
    """All perfect matchings of `xs` as tuples of (a, b) pairs."""
    if not xs: return ((),)
    return tuple(((xs[0], xs[i]),) + r
                 for i in range(1, len(xs))
                 for r in matchings(xs[1:i] + xs[i + 1:]))


def orbits(items, group, canon):
    """Group `items` into orbits under `group`; return [(representative, orbit_size), ...].

    `canon(item, g)` returns a hashable canonical key. The orbit of an item is
    keyed by the minimum canonical form across all group elements.
    """
    seen: dict = {}
    for x in items:
        k = min(canon(x, g) for g in group)
        seen[k] = (seen[k][0], seen[k][1] + 1) if k in seen else (x, 1)
    return list(seen.values())


def prefix(term: Term, p: str) -> Term:
    """Return `term` with every index name prefixed by `p:`."""
    ren = lambda d: {f'{p}:{k}': f'{p}:{v}' for k, v in d.items()}
    return Term(
        tn=term.tn.reindex({i: f'{p}:{i}' for i in term.tn.ind_map}),
        legs=ren(term.legs),
        symmetries=tuple(ren(d) for d in term.symmetries),
    )


def capture_cuda_graph(fn, params):
    """Capture `fn()` as a CUDA graph with `params` rebound to static buffers.

    Returns `(bufs, output, graph)`:
      • `bufs`: static tensors — write fresh parameter data into these before
        calling `graph.replay()`.
      • `output`: whatever `fn()` returned during capture (references into the
        static buffers; caller should clone before next replay if retaining).
      • `graph`: the `torch.cuda.CUDAGraph` to replay.
    """
    params = tuple(params)
    bufs = tuple(p.data.clone() for p in params)
    orig = [p.data for p in params]
    for p, buf in zip(params, bufs): p.data = buf
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): output = fn()
    finally:
        for p, o in zip(params, orig): p.data = o
    return bufs, output, graph


def direct_product(G, H):
    """Identity-including direct product of two disjoint permutation groups.

    G, H are iterables of dict permutations on disjoint index sets. Returns
    hashable sorted-item-tuple form of every element of ({id} ∪ G) × ({id} ∪ H).
    """
    freeze = lambda d: tuple(sorted(d.items()))
    g = tuple(freeze(x) for x in G)
    h = tuple(freeze(x) for x in H)
    return ((), *g, *h, *(a + b for a in g for b in h))
