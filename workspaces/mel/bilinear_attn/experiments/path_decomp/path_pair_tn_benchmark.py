#!/usr/bin/env python3
"""Benchmark whole TN against full path-pair TN for one checkpoint pair."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[5]
for _path in (str(ROOT), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from experiments.path_decomp.no_sym_tn_similarity import (  # noqa: E402
    similarity_no_sym,
    trace_nonconstant,
)
from experiments.path_decomp.path_pair_tn_heatmaps import (  # noqa: E402
    cache_stats,
    choose_device,
    configure_runtime_cache,
    load_component_with_n_ctx,
    path_pair_inner_products_component,
    print_cache_status,
)


def rel_error(got: float, ref: float) -> float:
    return abs(got - ref) / max(abs(ref), 1e-300)


def whole_raw(model_a, model_b) -> float:
    state = similarity_no_sym(model_a, model_b)
    return trace_nonconstant(state.s_ab)


def path_raw(model_a, model_b) -> float:
    return float(path_pair_inner_products_component(model_a, model_b).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir_a", required=True)
    parser.add_argument("--run_dir_b", default=None)
    parser.add_argument("--step_a", type=int, required=True)
    parser.add_argument("--step_b", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--cache_dir", default=".cache/ctg-paths")
    parser.add_argument("--n_ctx", type=int, default=None, help="Optional checkpoint-load context override.")
    args = parser.parse_args()

    run_dir_a = Path(args.run_dir_a)
    run_dir_b = Path(args.run_dir_b) if args.run_dir_b is not None else run_dir_a
    step_b = args.step_b if args.step_b is not None else args.step_a

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    configure_runtime_cache(cache_dir)
    print_cache_status(cache_dir)

    device = choose_device(args.device)
    dtype = torch.float64

    print(f"run_dir_a={run_dir_a}", flush=True)
    print(f"run_dir_b={run_dir_b}", flush=True)
    print(f"step_a={args.step_a}", flush=True)
    print(f"step_b={step_b}", flush=True)
    print(f"device={device}", flush=True)
    print(f"n_ctx={args.n_ctx}", flush=True)

    t0 = time.perf_counter()
    model_a = load_component_with_n_ctx(run_dir_a, args.step_a, args.n_ctx, dtype, device)
    model_b = load_component_with_n_ctx(run_dir_b, step_b, args.n_ctx, dtype, device)
    print(f"load_sec={time.perf_counter() - t0:.3f}", flush=True)

    t0 = time.perf_counter()
    whole_ab = whole_raw(model_a, model_b)
    whole_ab_sec = time.perf_counter() - t0
    print(f"whole_ab_sec={whole_ab_sec:.3f}", flush=True)

    t0 = time.perf_counter()
    path_ab = path_raw(model_a, model_b)
    path_ab_sec = time.perf_counter() - t0
    print(f"path_ab_sec={path_ab_sec:.3f}", flush=True)

    t0 = time.perf_counter()
    whole_aa = whole_raw(model_a, model_a)
    whole_bb = whole_raw(model_b, model_b)
    whole_norm_sec = time.perf_counter() - t0
    print(f"whole_norm_sec={whole_norm_sec:.3f}", flush=True)

    t0 = time.perf_counter()
    path_aa = path_raw(model_a, model_a)
    path_bb = path_raw(model_b, model_b)
    path_norm_sec = time.perf_counter() - t0
    print(f"path_norm_sec={path_norm_sec:.3f}", flush=True)

    whole_cos = whole_ab / ((whole_aa * whole_bb) ** 0.5)
    path_cos = path_ab / ((path_aa * path_bb) ** 0.5)
    n_files, n_bytes = cache_stats(cache_dir)

    print("\nRESULTS")
    print(f"whole_raw={whole_ab:.12e}")
    print(f"path_raw={path_ab:.12e}")
    print(f"raw_abs_err={abs(path_ab - whole_ab):.12e}")
    print(f"raw_rel_err={rel_error(path_ab, whole_ab):.12e}")
    print(f"whole_cos={whole_cos:.12e}")
    print(f"path_cos={path_cos:.12e}")
    print(f"cos_abs_err={abs(path_cos - whole_cos):.12e}")
    print(f"cos_rel_err={rel_error(path_cos, whole_cos):.12e}")
    print(f"whole_self_a={whole_aa:.12e}")
    print(f"path_self_a={path_aa:.12e}")
    print(f"whole_self_b={whole_bb:.12e}")
    print(f"path_self_b={path_bb:.12e}")
    print(f"cache_files_after={n_files}")
    print(f"cache_bytes_after={n_bytes}")


if __name__ == "__main__":
    main()
