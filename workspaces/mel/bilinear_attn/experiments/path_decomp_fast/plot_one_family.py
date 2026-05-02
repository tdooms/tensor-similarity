#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[5]
for _path in (str(ROOT), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _local_normalize(values: np.ndarray) -> np.ndarray:
    sims = np.full_like(values, np.nan)
    for f in range(values.shape[0]):
        diag = np.diag(values[f])
        denom = np.sqrt(np.outer(diag, diag))
        with np.errstate(invalid="ignore", divide="ignore"):
            s = values[f] / denom
        sims[f] = np.where(np.isfinite(s) & np.isfinite(denom) & (denom > 0), s, np.nan)
    return sims


def _resolve_family(labels: list[str], family: str | None, family_index: int | None) -> tuple[int, str]:
    if family_index is not None:
        return family_index, labels[family_index]
    if family is None:
        raise ValueError("Pass --family or --family-index")
    if family not in labels:
        raise ValueError(f"Unknown family '{family}'. Available: {labels}")
    idx = labels.index(family)
    return idx, labels[idx]


def _plot_one(matrix: np.ndarray, steps: list[int], label: str, output: Path, vmin: float, vmax: float, title_suffix: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    im = ax.imshow(np.ma.masked_invalid(matrix), vmin=vmin, vmax=vmax, cmap="coolwarm")
    ax.set_title(f"Family {label}: locally normalized diagonal TN sim {title_suffix}".strip())
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("checkpoint step")
    ax.set_xticks(range(len(steps)))
    ax.set_yticks(range(len(steps)))
    ax.set_xticklabels(steps, rotation=90, fontsize=7)
    ax.set_yticklabels(steps, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.82).set_label("local cosine")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot one family heatmap from family_diag_tn_sims.npz")
    parser.add_argument("--data", required=True)
    parser.add_argument("--family", default=None)
    parser.add_argument("--family-index", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vmin", type=float, default=-1.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--title-suffix", default="")
    parser.add_argument("--recompute_local_norm", action="store_true")
    args = parser.parse_args()

    data = np.load(Path(args.data), allow_pickle=True)
    labels = [str(x) for x in data["family_labels"]]
    steps = [int(x) for x in data["steps"]]
    values = data["family_diag_values"]
    sims = _local_normalize(values) if args.recompute_local_norm else data["family_local_sims"]

    idx, label = _resolve_family(labels, args.family, args.family_index)
    _plot_one(sims[idx], steps, label, Path(args.output), args.vmin, args.vmax, args.title_suffix)


if __name__ == "__main__":
    main()
