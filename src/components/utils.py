"""Generic helpers: combinatorics, group orbits, CUDA-graph capture."""
from functools import cache

import torch


@cache
def matchings(xs):
    """All perfect matchings of `xs` as tuples of (a, b) pairs."""
    if not xs: return ((),)
    return tuple(((xs[0], xs[i]),) + r
                 for i in range(1, len(xs))
                 for r in matchings(xs[1:i] + xs[i + 1:]))


def orbits(items, group, canon):
    """Group `items` into orbits under `group`; return [(rep, size), ...]."""
    seen: dict = {}
    for x in items:
        k = min(canon(x, g) for g in group)
        seen[k] = (seen[k][0], seen[k][1] + 1) if k in seen else (x, 1)
    return list(seen.values())


def direct_product(perms_l, nl, perms_r, nr):
    """Direct product of two positional-perm groups on disjoint ranges."""
    id_l = tuple(range(nl))
    id_r = tuple(nl + x for x in range(nr))
    gl = (id_l, *perms_l)
    gr = (id_r, *(tuple(nl + x for x in h) for h in perms_r))
    return tuple(g + h for g in gl for h in gr)


def capture_cuda_graph(fn, params, pool=None):
    """Capture `fn()` as a CUDA graph with `params` rebound to static buffers.

    Pass a shared `pool = torch.cuda.graph_pool_handle()` to have multiple
    captured graphs reuse one GPU memory pool — peak VRAM then = max(live
    sets across graphs) instead of sum. Our `similarity()` captures one
    self-graph and one cross-graph; sharing the pool is the difference
    between ~8 GB peak and ~16 GB peak at Mel scale."""
    params = tuple(params)
    bufs = tuple(p.data.clone() for p in params)
    orig = [p.data for p in params]
    for p, buf in zip(params, bufs): p.data = buf
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(stream)
    # Warmup intermediates are unreferenced at this point but still sit in the
    # PyTorch allocator's cache. Force-release so capture's memory pool doesn't
    # expand its reservation to cover them — they're not needed for replay.
    torch.cuda.empty_cache()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool): output = fn()
    for p, o in zip(params, orig): p.data = o
    return bufs, output, graph
