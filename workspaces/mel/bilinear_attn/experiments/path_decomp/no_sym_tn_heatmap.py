#!/usr/bin/env python3
"""Checkpoint heatmap for whole-model no-symmetry TN cosine similarity."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.path_decomp.no_sym_tn_similarity import (  # noqa: E402
    configure_cache,
    load_component,
    similarity_no_sym,
    trace_nonconstant,
)


DEFAULT_STEPS = list(range(0, 15001, 500))


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return device


def load_existing(path: Path, steps: list[int]) -> np.ndarray:
    n = len(steps)
    mat = np.full((n, n), np.nan, dtype=np.float64)
    np.fill_diagonal(mat, 1.0)
    if not path.exists():
        return mat

    old = np.load(path)
    old_steps = [int(x) for x in old["steps"]]
    old_mat = old["sim_matrix"]
    old_index = {s: i for i, s in enumerate(old_steps)}
    for i, step_i in enumerate(steps):
        for j, step_j in enumerate(steps):
            if step_i in old_index and step_j in old_index:
                v = old_mat[old_index[step_i], old_index[step_j]]
                if not np.isnan(v):
                    mat[i, j] = v
    return mat


def save_matrix(path: Path, steps: list[int], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, steps=np.array(steps), sim_matrix=matrix, method="no_sym_tn")


def plot_heatmap(path: Path, steps: list[int], matrix: np.ndarray, window: int | None) -> None:
    fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
    shown = np.ma.masked_invalid(matrix)
    im = ax.imshow(shown, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_title(f"Whole no-sym TN cosine similarity" + (f" (window={window})" if window else ""))
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("checkpoint step")
    ax.set_xticks(range(len(steps)))
    ax.set_yticks(range(len(steps)))
    ax.set_xticklabels(steps, rotation=90, fontsize=7)
    ax.set_yticklabels(steps, fontsize=7)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("cosine similarity")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def pair_indices(n: int, window: int | None):
    for i in range(n):
        for j in range(i + 1, n):
            if window is None or (j - i) <= window:
                yield i, j


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", default="experiments/induction_heads/runs/small-big-experiment-runs")
    parser.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--cache_dir", default=".cache/ctg-paths")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    configure_cache(cache_dir)

    # Import-time cache path is controlled by the env var set in configure_cache.
    device = choose_device(args.device)
    dtype = torch.float64
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else run_dir / "path_decomp_trajectory" / "whole_no_sym_tn_heatmap"
    )
    data_path = output_dir / "whole_no_sym_tn_similarity.npz"
    image_path = output_dir / "whole_no_sym_tn_similarity.png"

    matrix = load_existing(data_path, args.steps)
    components: dict[int, object] = {}
    traces: dict[int, float] = {}

    print(f"device={device}")
    print(f"cache_dir={cache_dir}")
    print(f"steps={args.steps}")
    print(f"window={args.window}")

    for i, j in pair_indices(len(args.steps), args.window):
        if not np.isnan(matrix[i, j]):
            continue
        step_i = args.steps[i]
        step_j = args.steps[j]
        if step_i not in components:
            components[step_i] = load_component(run_dir, step_i, dtype, device)
        if step_j not in components:
            components[step_j] = load_component(run_dir, step_j, dtype, device)

        start = time.perf_counter()
        state = similarity_no_sym(components[step_i], components[step_j])
        elapsed = time.perf_counter() - start
        tr_aa = trace_nonconstant(state.s_aa)
        tr_ab = trace_nonconstant(state.s_ab)
        tr_bb = trace_nonconstant(state.s_bb)
        traces.setdefault(step_i, tr_aa)
        traces.setdefault(step_j, tr_bb)
        sim = tr_ab / ((tr_aa * tr_bb) ** 0.5)
        matrix[i, j] = sim
        matrix[j, i] = sim
        save_matrix(data_path, args.steps, matrix)
        print(f"step {step_i} vs {step_j}: sim={sim:.8f} time={elapsed:.2f}s")

    plot_heatmap(image_path, args.steps, matrix, args.window)
    print(f"Wrote data: {data_path}")
    print(f"Wrote heatmap: {image_path}")


if __name__ == "__main__":
    main()
