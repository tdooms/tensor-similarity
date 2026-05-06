#!/usr/bin/env python3
"""Plot layer-block TN heatmaps produced by layer_block_tn_from_paths.py."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_heatmap(
    matrix: np.ndarray,
    steps: np.ndarray,
    title: str,
    output_path: Path,
    *,
    vmin: float | None,
    vmax: float | None,
    cmap: str,
) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to layer_block_tn_sims.npz")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--raw", action="store_true", help="Plot raw unnormalized block sums instead of local sims.")
    parser.add_argument("--vmin", type=float, default=-1.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--cmap", default="coolwarm")
    args = parser.parse_args()

    input_path = Path(args.input)
    data = np.load(input_path, allow_pickle=True)
    steps = data["steps"]
    labels = [str(x) for x in data["group_labels"]]
    values_key = "layer_block_values" if args.raw else "layer_block_local_sims"
    values = data[values_key]

    output_dir = Path(args.output_dir) if args.output_dir else input_path.with_suffix("").parent / "layer_block_heatmaps"
    output_dir.mkdir(parents=True, exist_ok=True)

    vmin = None if args.raw else args.vmin
    vmax = None if args.raw else args.vmax
    mode = "raw" if args.raw else "local"

    for gi, g in enumerate(labels):
        for hi, h in enumerate(labels):
            matrix = values[:, :, gi, hi]
            output_path = output_dir / f"{mode}_{g}_vs_{h}.png"
            title = f"{g} vs {h} layer-block TN {'raw' if args.raw else 'local cosine'}"
            plot_heatmap(matrix, steps, title, output_path, vmin=vmin, vmax=vmax, cmap=args.cmap)

    write_summary_csv(output_dir / f"{mode}_summary.csv", values, labels)
    print(f"Wrote heatmaps: {output_dir}")
    print(f"Wrote summary: {output_dir / f'{mode}_summary.csv'}")


if __name__ == "__main__":
    main()
