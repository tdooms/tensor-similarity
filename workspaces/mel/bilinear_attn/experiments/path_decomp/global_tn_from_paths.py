#!/usr/bin/env python3
"""Aggregate a 4D path-pair TN tensor into whole-model TN heatmaps.

Input is the path matrix produced by ``path_pair_tn_heatmaps.py``:

    T[i, j, p, q] = <checkpoint_i path_p, checkpoint_j path_q>

The raw whole-model TN similarity is the sum over every path pair:

    raw[i, j] = sum_p sum_q T[i, j, p, q]

The global cosine-style similarity uses the full-model self norms:

    sim[i, j] = raw[i, j] / sqrt(raw[i, i] * raw[j, j])
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def finite_sum(matrix: np.ndarray, *, allow_partial: bool) -> float:
    if allow_partial:
        return float(np.nansum(matrix))
    if not np.isfinite(matrix).all():
        return float("nan")
    return float(matrix.sum())


def compute_global_tn(path_values: np.ndarray, *, allow_partial: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if path_values.ndim != 4:
        raise ValueError(f"Expected shape (n, n, n_paths, n_paths), got {path_values.shape}")
    n = path_values.shape[0]
    if path_values.shape[1] != n:
        raise ValueError(f"Expected square checkpoint axes, got {path_values.shape[:2]}")
    if path_values.shape[2] != path_values.shape[3]:
        raise ValueError(f"Expected square path axes, got {path_values.shape[2:]}")

    raw = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            raw[i, j] = finite_sum(path_values[i, j], allow_partial=allow_partial)

    norms = np.diag(raw).copy()
    sims = np.full_like(raw, np.nan)
    for i in range(n):
        for j in range(n):
            denom = norms[i] * norms[j]
            if np.isfinite(raw[i, j]) and np.isfinite(denom) and denom > 0:
                sims[i, j] = raw[i, j] / np.sqrt(denom)
    return raw, sims, norms


def plot_heatmap(matrix: np.ndarray, steps: np.ndarray, output_path: Path, *, title: str, vmin, vmax, cmap: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    im = ax.imshow(np.ma.masked_invalid(matrix), vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("checkpoint step")
    ax.set_xticks(range(len(steps)))
    ax.set_yticks(range(len(steps)))
    ax.set_xticklabels([str(int(s)) for s in steps], rotation=90, fontsize=7)
    ax.set_yticklabels([str(int(s)) for s in steps], fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.82)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_summary_csv(path: Path, values: np.ndarray, steps: np.ndarray) -> None:
    rows = []
    for i, step_i in enumerate(steps):
        for j, step_j in enumerate(steps):
            rows.append(
                {
                    "step_i": int(step_i),
                    "step_j": int(step_j),
                    "value": float(values[i, j]) if np.isfinite(values[i, j]) else float("nan"),
                }
            )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step_i", "step_j", "value"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to path_pair_tn_matrices.npz")
    parser.add_argument("--output", default=None, help="Output .npz path")
    parser.add_argument("--plot_dir", default=None, help="If set, write raw/global-normalized heatmap PNGs here.")
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="Treat missing path-pair entries as zero. By default, any NaN in a 34x34 block makes that pair NaN.",
    )
    parser.add_argument("--vmin", type=float, default=-1.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--cmap", default="coolwarm")
    args = parser.parse_args()

    input_path = Path(args.input)
    data = np.load(input_path, allow_pickle=True)
    steps = data["steps"]
    path_values = data["path_pair_values"]
    raw, sims, norms = compute_global_tn(path_values, allow_partial=args.allow_partial)

    output_path = Path(args.output) if args.output else input_path.with_name("global_tn_sims_from_paths.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        steps=steps,
        global_tn_values=raw,
        global_tn_sims=sims,
        global_tn_norms=norms,
        source=str(input_path),
        allow_partial=args.allow_partial,
    )
    print(f"Wrote global TN data: {output_path}")
    print(f"steps={len(steps)} finite_global_sims={np.isfinite(sims).sum()}/{sims.size}")

    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        plot_heatmap(raw, steps, plot_dir / "global_tn_raw.png", title="Global TN raw", vmin=None, vmax=None, cmap=args.cmap)
        plot_heatmap(
            sims,
            steps,
            plot_dir / "global_tn_sim.png",
            title="Global TN cosine",
            vmin=args.vmin,
            vmax=args.vmax,
            cmap=args.cmap,
        )
        write_summary_csv(plot_dir / "global_tn_sim_summary.csv", sims, steps)
        print(f"Wrote global TN heatmaps: {plot_dir}")


if __name__ == "__main__":
    main()
