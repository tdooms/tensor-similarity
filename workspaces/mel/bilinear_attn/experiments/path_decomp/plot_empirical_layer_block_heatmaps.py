#!/usr/bin/env python3
"""Plot empirical layer-block heatmaps, individually and as a 3x3 grid."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_heatmap(matrix, steps, title, output_path, *, vmin, vmax, cmap):
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


def plot_grid(values, labels, output_path, *, vmin, vmax, cmap):
    fig, axes = plt.subplots(3, 3, figsize=(8, 8), constrained_layout=True)
    last_im = None
    for gi, g in enumerate(labels):
        for hi, h in enumerate(labels):
            ax = axes[gi, hi]
            last_im = ax.imshow(
                np.ma.masked_invalid(values[:, :, gi, hi]),
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
            ax.set_title(f"{g}, {h}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    if last_im is not None:
        fig.colorbar(last_im, ax=axes, shrink=0.75, pad=0.01)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_summary_csv(path: Path, values: np.ndarray, labels: list[str]) -> None:
    rows = []
    for gi, g in enumerate(labels):
        for hi, h in enumerate(labels):
            mat = values[:, :, gi, hi]
            finite = mat[np.isfinite(mat)]
            rows.append(
                {
                    "source_group": g,
                    "target_group": h,
                    "mean": float(finite.mean()) if finite.size else float("nan"),
                    "min": float(finite.min()) if finite.size else float("nan"),
                    "max": float(finite.max()) if finite.size else float("nan"),
                    "finite_count": int(finite.size),
                }
            )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to empirical_layer_block_sims.npz")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--raw", action="store_true", help="Plot raw empirical dot products instead of local sims.")
    parser.add_argument("--vmin", type=float, default=-1.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--cmap", default="coolwarm")
    parser.add_argument("--grid", action="store_true", help="Also write one compact 3x3 image containing all heatmaps.")
    args = parser.parse_args()

    input_path = Path(args.input)
    data = np.load(input_path, allow_pickle=True)
    steps = data["steps"]
    labels = [str(x) for x in data["group_labels"]]
    key = "empirical_block_values" if args.raw else "empirical_block_local_sims"
    values = data[key]
    mode = "raw" if args.raw else "local"
    output_dir = Path(args.output_dir) if args.output_dir else input_path.with_suffix("").parent / "empirical_layer_block_heatmaps"
    output_dir.mkdir(parents=True, exist_ok=True)

    vmin = None if args.raw else args.vmin
    vmax = None if args.raw else args.vmax
    for gi, g in enumerate(labels):
        for hi, h in enumerate(labels):
            title = f"{g} vs {h} empirical {'raw' if args.raw else 'local cosine'}"
            output_path = output_dir / f"{mode}_{g}_vs_{h}.png"
            plot_heatmap(values[:, :, gi, hi], steps, title, output_path, vmin=vmin, vmax=vmax, cmap=args.cmap)

    if args.grid:
        grid_path = output_dir / f"{mode}_all_layer_blocks.png"
        plot_grid(values, labels, grid_path, vmin=vmin, vmax=vmax, cmap=args.cmap)
        print(f"Wrote grid: {grid_path}")

    summary_path = output_dir / f"{mode}_summary.csv"
    write_summary_csv(summary_path, values, labels)
    print(f"Wrote heatmaps: {output_dir}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
