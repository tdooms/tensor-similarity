"""Time and memory benchmark for TN similarity.

Compares two routes for computing ``cosine_similarity`` on equivalent
models:

* **wrapper**: mel's ``AttentionLM`` wrapped with ``AttentionLMComponent``.
* **direct**:  a model built from ``src.components.{Linear, Attention}``
  with mel's weights copied in (bypasses the adapter entirely).

Both represent the same mathematical function (mel's forward now uses
``lerp``, matching src), so their ``cosine_similarity`` values are
identical and the only difference is the code path. This isolates
adapter overhead (per-call Python, extra index renames, bias padding)
from the core TN cost.

Each row reports cold/warm wall-clock and peak memory. Cold is the first
call for a given (route, shape) — dominated by cotengra path search.
Warm is the steady-state cost once the expression cache and per-shape
CUDA-graph cache are populated.

To eliminate ordering bias (earlier cases warming caches used by later
ones — the cotengra path cache at ``~/.cache/tensor-mars/ctg-paths``, the
cuDNN/cuBLAS workspace, the Python import cache, …), every (case, route)
row is measured in a **fresh subprocess** with ``HOME``/``USERPROFILE``
redirected to a per-row ``TemporaryDirectory``. That guarantees the
cotengra cache is empty for each cold measurement, so cold numbers
between rows are comparable.

Usage:
    python -m tn_sim.benchmark                     # CPU, float64
    python -m tn_sim.benchmark --device cuda --dtype float32
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

import psutil
import torch
from torch import nn

from models import AttentionLM
from models.components import AttentionLMComponent
from tn_sim.similarity import cosine_similarity

from src.components.attention import Attention as SrcAttention
from src.components.linear import Linear as SrcLinear
from src.models.base import Model as SrcModel


@dataclass
class Case:
    label: str
    vocab_size: int
    n_ctx: int
    d_model: int
    n_head: int
    n_layers: int
    attn_type: str = "bilinear"


CASES = [
    Case("tiny-1L",  vocab_size=8,  n_ctx=4, d_model=8,  n_head=2, n_layers=1),
    Case("tiny-2L",  vocab_size=8,  n_ctx=4, d_model=8,  n_head=2, n_layers=2),
    Case("small-1L", vocab_size=16, n_ctx=8, d_model=16, n_head=2, n_layers=1),
]


def _cfg(case: Case) -> dict:
    return {
        "model": dict(
            vocab_size=case.vocab_size, n_ctx=case.n_ctx, d_model=case.d_model,
            n_head=case.n_head, n_layers=case.n_layers,
            attn_scale=0.5, attn_type=case.attn_type,
            use_bias_qk=True, use_rmsnorm_qk=False,
            norm_type="none", norm_places=[], rope_base=10000,
        ),
        "init": dict(std_embed=0.1, std_qkv=0.1, std_o=0.1),
    }


class _SrcDirectModel(SrcModel):
    """Functionally-equivalent model built from ``src.components`` primitives."""

    def __init__(self, mel: AttentionLM) -> None:
        super().__init__(None)
        V, D = mel.vocab_size, mel.d_model
        n_head, n_ctx = mel.n_head, mel.n_ctx
        bias = mel.layers[0].q1.bias is not None
        scale = mel.layers[0].scale

        self.embed = SrcLinear(V, D, bias=False)
        self.layers = nn.ModuleList([
            SrcAttention(D, n_head, n_ctx, mask='causal', bias=bias, scale=scale)
            for _ in range(mel.n_layers)
        ])
        self.unembed = SrcLinear(D, V, bias=False)
        with torch.no_grad():
            self.embed.weight.copy_(mel.embed.weight.T)
            for ref, src in zip(self.layers, mel.layers):
                for name in ("q1", "k1", "q2", "k2", "v", "o"):
                    getattr(ref, name).weight.copy_(getattr(src, name).weight)
                    if getattr(ref, name).bias is not None:
                        getattr(ref, name).bias.copy_(getattr(src, name).bias)
            self.unembed.weight.copy_(mel.unembed.weight)

    def components(self):
        return [self.embed] + list(self.layers) + [self.unembed]


def _build_pair(case: Case, route: str, *, device: str, dtype: torch.dtype):
    torch.manual_seed(0)
    mel_a = AttentionLM.from_config(_cfg(case)).to(device=device, dtype=dtype)
    torch.manual_seed(1)
    mel_b = AttentionLM.from_config(_cfg(case)).to(device=device, dtype=dtype)
    if route == "wrapper":
        return (
            AttentionLMComponent.from_trained_model(mel_a).to(device=device, dtype=dtype),
            AttentionLMComponent.from_trained_model(mel_b).to(device=device, dtype=dtype),
        )
    if route == "direct":
        return (
            _SrcDirectModel(mel_a).to(device=device, dtype=dtype),
            _SrcDirectModel(mel_b).to(device=device, dtype=dtype),
        )
    raise ValueError(route)


def _measure(fn, device: str) -> tuple[float, float, float]:
    """Return (elapsed_seconds, peak_memory_bytes, sim_value).

    CUDA: ``torch.cuda.max_memory_allocated`` (true tensor peak).
    CPU:  psutil RSS delta (captures torch + Python, approximate).
    """
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        val = fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated()
    else:
        proc = psutil.Process(os.getpid())
        rss_before = proc.memory_info().rss
        t0 = time.perf_counter()
        val = fn()
        elapsed = time.perf_counter() - t0
        rss_after = proc.memory_info().rss
        peak = max(0, rss_after - rss_before)
    return elapsed, peak, float(val)


def _fmt_mem(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:6.2f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:6.2f} TB"


_RESULT_TAG = "__BENCH_RESULT__"


def _worker(case: Case, route: str, device: str, dtype_name: str) -> None:
    """Run a single (case, route) and emit a one-line JSON result on stdout.

    Runs in a child process whose ``HOME``/``USERPROFILE`` point at a fresh
    temp dir, so ``Path.home()/.cache/tensor-mars/ctg-paths`` starts empty
    and the cold measurement reflects a genuine cotengra path search.
    """
    dtype = getattr(torch, dtype_name)
    a, b = _build_pair(case, route, device=device, dtype=dtype)
    call = lambda: cosine_similarity(a, b, device=device, dtype=dtype)
    cold_t, cold_mem, cold_val = _measure(call, device)
    warm_t, warm_mem, warm_val = _measure(call, device)
    # Equivalence sanity within the child: cold and warm sim must match.
    assert abs(cold_val - warm_val) < 1e-10, (
        f"cold/warm sim disagree in worker: {cold_val!r} vs {warm_val!r}"
    )
    print(_RESULT_TAG + json.dumps({
        "cold_t": cold_t, "cold_mem": cold_mem,
        "warm_t": warm_t, "warm_mem": warm_mem,
        "sim": warm_val,
    }), flush=True)


def _run_one_in_subprocess(
    case: Case, route: str, device: str, dtype_name: str,
) -> dict:
    payload = json.dumps({
        "case": dataclasses.asdict(case),
        "route": route,
        "device": device,
        "dtype": dtype_name,
    })
    with tempfile.TemporaryDirectory(prefix="tn-bench-home-") as tmp_home:
        env = os.environ.copy()
        # Make ``Path.home()`` resolve to an empty directory in the child so
        # the cotengra on-disk cache is guaranteed cold per row.
        env["HOME"] = tmp_home
        env["USERPROFILE"] = tmp_home
        env["XDG_CACHE_HOME"] = os.path.join(tmp_home, ".cache")
        proc = subprocess.run(
            [sys.executable, "-m", "tn_sim.benchmark",
             "--_worker", "--payload", payload],
            capture_output=True, text=True, env=env, check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"worker failed for {case.label}/{route} (exit {proc.returncode})\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_TAG):
            return json.loads(line[len(_RESULT_TAG):])
    raise RuntimeError(
        f"worker for {case.label}/{route} produced no result line\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )


def run(device: str, dtype_name: str) -> None:
    header = (
        f"{'case':<10} | {'route':<8} | {'cold (s)':>9} | {'warm (s)':>9} "
        f"| {'peak mem':>10} | {'sim':>12}"
    )
    print(header)
    print("-" * len(header))

    for case in CASES:
        sims_per_route: dict[str, float] = {}
        for route in ("wrapper", "direct"):
            r = _run_one_in_subprocess(case, route, device, dtype_name)
            sims_per_route[route] = r["sim"]
            print(
                f"{case.label:<10} | {route:<8} | {r['cold_t']:9.2f} | {r['warm_t']:9.2f} "
                f"| {_fmt_mem(max(r['cold_mem'], r['warm_mem'])):>10} | {r['sim']:12.6f}"
            )
        # Equivalence check: wrapper and direct routes should give the same value.
        diff = abs(sims_per_route["wrapper"] - sims_per_route["direct"])
        assert diff < 1e-8, (
            f"{case.label}: wrapper vs direct disagree by {diff:.3e} "
            f"({sims_per_route['wrapper']!r} vs {sims_per_route['direct']!r})"
        )
        print(f"{'':<10} | {'|dsim|':<8} | {diff:>31.2e}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    # Internal worker mode — not intended for direct user invocation.
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--payload", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args._worker:
        payload = json.loads(args.payload)
        _worker(
            case=Case(**payload["case"]),
            route=payload["route"],
            device=payload["device"],
            dtype_name=payload["dtype"],
        )
        return

    run(args.device, args.dtype)


if __name__ == "__main__":
    main()
