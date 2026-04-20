"""Exact Gaussian functional similarity via second-moment propagation.

Σ = I_n ⊗ I_d starts as the Gaussian input covariance in the padded rep.
Per component: Σ ← Σ_{l, r ∈ terms} E[l · rᵀ | x ~ N(·, Σ)] via Isserlis
(deduped Wick matchings + μ-correction). For a ≠ b we jointly propagate
the triple (Σ_aa, Σ_ab, Σ_bb); the Σ_ab update routes each bridge by leg
position (left-term legs < right-term legs in the joined list). Self-pairs
(Wick singletons) slice Σ at (pos=0, dat=0) — the padded constant-1 reference
— giving μ. See SIMILARITY.md for the math.
"""
import pickle
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

# Persistent caches. Two levels:
#   * cotengra's on-disk path cache (`_CACHE_DIR`) — managed by `_OPT`.
#   * `_PATHS` — compiled cotengra `Contractor` callables, pickled to
#     `_PATHS_FILE`. Loaded at import, saved when `_precompile_mode` exits
#     back to strict mode. Contractors pickle cleanly with stdlib pickle.
#   * `_GRAPHS` — CUDA-graph captures. In-memory only; graphs hold GPU memory
#     addresses and cannot be serialized. All captures SHARE a single memory
#     pool (`_POOL`), so peak VRAM = max(graph size) not sum.
#   * `_PRECOMPILE` — fail-loud guard. `_contract` raises outside this mode;
#     precompile() is the only entrypoint that may build new contractors or
#     CUDA graphs.
_PATHS_FILE = _CACHE_DIR / 'exprs.pkl'
_PATHS: dict = pickle.loads(_PATHS_FILE.read_bytes()) if _PATHS_FILE.exists() else {}
_GRAPHS: dict = {}
_POOL = None          # lazily initialised by `_graphed` on first CUDA capture
_PRECOMPILE: bool = False


def _save_paths():
    """Atomic pickle write so a mid-flight crash cannot corrupt the cache."""
    tmp = _PATHS_FILE.with_suffix('.pkl.tmp')
    tmp.write_bytes(pickle.dumps(_PATHS))
    tmp.rename(_PATHS_FILE)


class _precompile_mode:
    """Context manager: while inside, `_contract` compiles uncached topologies
    into `_PATHS` instead of raising. On return to strict mode (outermost
    exit) the cache is persisted to disk so the next Python session finds
    every contractor already built — no re-compilation ever on the hot path.

    Nestable; inner exits do NOT save so we write at most once per precompile."""
    def __enter__(self):
        global _PRECOMPILE
        self.prev = _PRECOMPILE
        _PRECOMPILE = True
    def __exit__(self, *_):
        global _PRECOMPILE
        _PRECOMPILE = self.prev
        if not self.prev: _save_paths()


# ── Isserlis combinatorics ────────────────────────────────────────────────

def _canon(matching, perm, legs):
    return tuple(sorted(tuple(sorted((legs[perm[i]], legs[perm[j]])))
                         for i, j in matching))


@cache
def _plan(legs, syms):
    """Deduped Wick matchings + μ-correction config. Pure Python.

    The `-((n-1)!!-1)·μⁿ` correction cancels the all-singletons overcount from
    the homogeneous identification Σ̂ = Σ + μμᵀ. For n ≥ 6, mixed partitions
    (pairs + singletons) are also overcounted — fine under bias=False because
    μ=0 on active-term legs kills every singleton-containing partition and the
    formula is silent. Turning biases on would require explicit partial-matching
    enumeration instead of this scalar fix-up."""
    wick = orbits(matchings(tuple(range(len(legs)))), syms, partial(_canon, legs=legs))
    configs = tuple(m for m, _ in wick) + (tuple((i, i) for i in range(len(legs))),)
    weights = tuple(w for _, w in wick) + (-(prod(range(1, len(legs), 2)) - 1),)
    return configs, weights


@cache
def _weights(weights_py, device, dtype):
    return torch.tensor(weights_py, device=device, dtype=dtype)


# ── joint TN E[t_l · t_rᵀ] ────────────────────────────────────────────────

def _sided(term, side):
    """Prefix every ind with `side:` (l/r mark the two halves of t_l·t_rᵀ)."""
    tn = term.tn.reindex({i: f'{side}:{i}' for i in term.tn.ind_map})
    legs = tuple((f'{side}:{p}', f'{side}:{d}') for d, p in sorted(term.legs.items()))
    return tn, legs


def _join(tl, tr):
    """Joint TN, legs, left-leg count n_l, and combined positional symmetries.

    When `tl is tr` (Σ_aa or Σ_bb update — self-like sum), the joint TN is
    l↔r exchange-symmetric (M(t,t)[s_l,d_l,s_r,d_r] = M(t,t)[s_r,d_r,s_l,d_l]).
    We augment `syms` with the positional swap `i ↔ i+n_l` to dedup Wick orbits
    — cuts active×active from ~236 to ~60 configs."""
    a, la = _sided(tl, 'l')
    b, lb = _sided(tr, 'r')
    n_l, n_r = len(la), len(lb)
    syms = direct_product(tl.symmetries, n_l, tr.symmetries, n_r)
    if tl is tr:
        swap = tuple(i + n_l if i < n_l else i - n_l for i in range(n_l + n_r))
        syms = syms + tuple(tuple(g[swap[i]] for i in range(n_l + n_r)) for g in syms)
    return a | b, la + lb, n_l, syms


def _contract(core, bridges):
    """Run a cached cotengra expression for this (core_inds, bridge_inds) topology.
    Raises outside `_precompile_mode` if the topology isn't in `_PATHS` — no
    hidden path-opt / expression build on first call. `Attention.network()`
    rebuilds its tensors each `.terms()` call, so the cache is topology-only
    and we always feed fresh tensor data into the compiled expression."""
    key = tuple(t.inds for t in core) + tuple(inds for _, inds in bridges)
    if key not in _PATHS:
        if not _PRECOMPILE:
            raise RuntimeError(
                f'uncompiled topology; call precompile(*models) first. key={key!r}')
        tn = core.copy()
        for data, inds in bridges: tn &= Tensor(data, inds=inds)
        _PATHS[key] = tn.contract(output_inds=_OUT, optimize=_OPT, get='expression')
    return _PATHS[key](*(t.data for t in core), *(data for data, _ in bridges))


def _moment(tl, tr, aa, ab, bb):
    """E[t_l · t_rᵀ] via Isserlis + μ correction. Wick pair (i, j) with i ≤ j
    routes to aa if j < n_l, bb if i ≥ n_l, else ab; self-pair slices to μ."""
    tn, legs, n_l, syms = _join(tl, tr)
    configs, weights_py = _plan(legs, syms)
    ws = _weights(weights_py, aa.device, aa.dtype)
    def bridge(i, j):
        σ = aa if j < n_l else bb if i >= n_l else ab
        return (σ[..., 0, 0], legs[i]) if i == j else (σ, (*legs[i], *legs[j]))
    return sum(w * _contract(tn, [bridge(i, j) for i, j in c])
               for c, w in zip(configs, ws))


def _update(aa, ab, bb, ts_l, ts_r):
    """Σ ← Σ_{l, r} E[l · rᵀ]."""
    return sum(_moment(l, r, aa, ab, bb) for l in ts_l for r in ts_r)


# ── propagation ───────────────────────────────────────────────────────────

def _initial(model):
    """Σ = I_n ⊗ I_d in the padded rep; rank-4 (n, d+1, n, d+1)."""
    p = next(model.parameters())
    d = model.components()[0].network().ind_size('in:d0')
    like = dict(device=p.device, dtype=p.dtype)
    return (Tensor(torch.eye(model.n_ctx, **like), inds=('l:out:s', 'r:out:s'))
          & Tensor(torch.eye(d, **like), inds=('l:out:d', 'r:out:d'))
           ).contract(output_inds=_OUT).data


def _propagate(a, b):
    """Joint (Σ_aa, Σ_ab, Σ_bb) through aligned components. When a is b, all
    three slots stay aliased → 1 update per layer; else 3. Input covariance
    Σ = I is common, so the three start from a single `_initial` regardless."""
    aa = ab = bb = _initial(a)
    for ca, cb in zip(a.components(), b.components()):
        ta = ca.terms(a.n_ctx)
        tb = ta if a is b else cb.terms(b.n_ctx)
        aa_ = _update(aa, aa, aa, ta, ta)
        aa, ab, bb = ((aa_,) * 3 if a is b
                      else (aa_, _update(aa, ab, bb, ta, tb), _update(bb, bb, bb, tb, tb)))
    return aa, ab, bb


# ── CUDA graph wrapper ────────────────────────────────────────────────────

def _graphed(fn, *models):
    """Cache a CUDA graph keyed by (param shapes, device, dtype). All graphs
    captured here share one memory pool (`_POOL`) so peak VRAM is max(graph
    size), not sum — critical with vocab-scale outputs where each graph's
    live set is ~3 GB and two graphs (self + cross) would otherwise double it."""
    global _POOL
    ps = tuple(p for m in models for p in m.parameters())
    if not ps[0].is_cuda: return fn()
    key = (tuple(p.shape for p in ps), ps[0].device, ps[0].dtype)
    if key not in _GRAPHS:
        if _POOL is None: _POOL = torch.cuda.graph_pool_handle()
        _GRAPHS[key] = capture_cuda_graph(fn, ps, pool=_POOL)
    bufs, out, graph = _GRAPHS[key]
    for src, dst in zip(ps, bufs): dst.copy_(src.data)
    graph.replay()
    return tuple(o.clone() for o in out)


# ── public API ────────────────────────────────────────────────────────────

@torch.no_grad()
def similarity(a, b):
    """2×2 block `[[⟨a,a⟩, ⟨a,b⟩], [⟨a,b⟩ᵀ, ⟨b,b⟩]]`, shape (2, 2, n, d+1, n, d+1)."""
    aa, ab, bb = _graphed(lambda: _propagate(a, b), *((a,) if a is b else (a, b)))
    ba = ab.permute(2, 3, 0, 1)   # Σ_ba = E[b · aᵀ] = Σ_ab with L/R axes swapped
    return torch.stack([torch.stack([aa, ab]), torch.stack([ba, bb])])


@torch.no_grad()
def precompile(*models):
    """The ONLY entry point that may build new contraction expressions or CUDA
    graphs. Every self-path hits the same cache keys (topology = term-pair
    structure, inds-only; graph = param shapes) so ONE `similarity(m, m)` call
    covers all self-pairs over same-arch `models`. Same for cross: one call
    covers every (a, b) with a ≠ b. Two calls total, not O(N²).

    Contractors then persist to disk via `_precompile_mode.__exit__`; a fresh
    Python session finds them already built and only re-captures CUDA graphs
    (which hold GPU memory addresses and can't be serialized)."""
    with _precompile_mode():
        if models:
            similarity(models[0], models[0])              # one self graph
        if len(models) >= 2:
            similarity(models[0], models[1])              # one cross graph
