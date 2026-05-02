#!/usr/bin/env python3
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

from src.components import utils as comp_utils

from experiments.path_decomp.no_sym_tn_similarity import load_component
from experiments.path_decomp_fast import install_fast_ctg
from experiments.path_decomp_fast._pair_engine import build_step_artifacts, compute_pair


def _cache_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def run_warmup(run_dir: Path, step: int, device: torch.device, cache_dir: Path) -> None:
    install_fast_ctg(cache_dir)
    before_keys = len(comp_utils._EXPRS)
    before_size = _cache_size_bytes(cache_dir)

    t0 = time.perf_counter()
    comp = load_component(run_dir, step, torch.float32, device)
    art = build_step_artifacts(comp)
    compute_pair(art, art, families="all", use_orbit_master=True, bridges_to_f32=False, is_self=True)
    elapsed = time.perf_counter() - t0

    after_keys = len(comp_utils._EXPRS)
    after_size = _cache_size_bytes(cache_dir)
    print(f"warmup step={step} done in {elapsed:.2f}s")
    print(f"new in-memory keys: {after_keys - before_keys}")
    print(f"cache size bytes: {before_size} -> {after_size} (delta {after_size - before_size})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile CTG paths once for family-diagonal TN.")
    parser.add_argument("--run_dir", default="experiments/induction_heads/runs/small-big-experiment-runs")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--cache_dir", default=".cache/ctg-paths-fast")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    run_warmup(Path(args.run_dir), args.step, device, Path(args.cache_dir))


if __name__ == "__main__":
    main()
