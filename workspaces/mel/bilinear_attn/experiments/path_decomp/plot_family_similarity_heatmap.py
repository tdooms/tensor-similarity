#!/usr/bin/env python3
"""Plot canonical path-family similarity heatmaps from a trajectory JSON."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LAYER2_GROUPS = [
    ("00000", ["00000"]),
    ("00001", ["00001", "00100"]),
    ("00010", ["00010", "01000"]),
    ("00011", ["00011", "01100"]),
    ("00101", ["00101"]),
    ("00110", ["00110", "01001"]),
    ("00111", ["00111", "01101"]),
    ("01010", ["01010"]),
    ("01011", ["01011", "01110"]),
    ("01111", ["01111"]),
    ("10000", ["10000"]),
    ("10001", ["10001", "10100"]),
    ("10010", ["10010", "11000"]),
    ("10011", ["10011", "11100"]),
    ("10101", ["10101"]),
    ("10110", ["10110", "11001"]),
    ("10111", ["10111", "11101"]),
    ("11010", ["11010"]),
    ("11011", ["11011", "11110"]),
    ("11111", ["11111"]),
]


def family_label(name: str) -> str:
    if name in ("direct", "layer1"):
        return name
    prefix, rho_s = name.split(":")
    assert prefix == "layer2"
    return f"{int(rho_s):05b}"


def canonical_maps():
    bit_to_canon = {}
    for canon, members in LAYER2_GROUPS:
        for member in members:
            bit_to_canon[member] = canon

    labels = ["direct", "layer1", *[canon for canon, _ in LAYER2_GROUPS]]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    return bit_to_canon, labels, label_to_idx


def canonical_matrix(family_pairs: list[str], values: list[float], reduction: str) -> tuple[list[str], np.ndarray]:
    bit_to_canon, labels, label_to_idx = canonical_maps()
    cells: dict[tuple[int, int], list[float]] = defaultdict(list)

    for pair, value in zip(family_pairs, values):
        left, right = pair.split("|")
        left_label = family_label(left)
        right_label = family_label(right)
        left_canon = bit_to_canon.get(left_label, left_label)
        right_canon = bit_to_canon.get(right_label, right_label)
        i = label_to_idx[left_canon]
        j = label_to_idx[right_canon]
        cells[(i, j)].append(float(value))

    mat = np.zeros((len(labels), len(labels)), dtype=float)
    for (i, j), cell_values in cells.items():
        if reduction == "mean":
            mat[i, j] = float(np.mean(cell_values))
        elif reduction == "first":
            mat[i, j] = cell_values[0]
        elif reduction == "sum":
            mat[i, j] = float(np.sum(cell_values))
        else:
            raise ValueError(f"unknown reduction: {reduction}")
    return labels, mat


def duplicate_spread_rows(family_pairs: list[str], values: list[float]) -> list[dict]:
    bit_to_canon, labels, label_to_idx = canonical_maps()
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for pair, value in zip(family_pairs, values):
        left, right = pair.split("|")
        left_label = family_label(left)
        right_label = family_label(right)
        left_canon = bit_to_canon.get(left_label, left_label)
        right_canon = bit_to_canon.get(right_label, right_label)
        cells[(left_canon, right_canon)].append(float(value))

    rows = []
    for (left, right), cell_values in cells.items():
        if len(cell_values) == 1:
            continue
        spread = max(cell_values) - min(cell_values)
        denom = max(max(abs(v) for v in cell_values), 1e-300)
        rows.append({
            "left_family": left,
            "right_family": right,
            "n_values": len(cell_values),
            "min": min(cell_values),
            "max": max(cell_values),
            "mean": float(np.mean(cell_values)),
            "spread": spread,
            "relative_spread": spread / denom,
        })
    rows.sort(key=lambda r: abs(r["spread"]), reverse=True)
    return rows


def plot_heatmap(labels: list[str], mat: np.ndarray, title: str, output_path: Path) -> None:
    vmax = np.nanmax(np.abs(mat))
    if vmax == 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(13, 11), constrained_layout=True)
    im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("family B")
    ax.set_ylabel("family A")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("summed signed contribution")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_top_pairs(labels: list[str], mat: np.ndarray, output_path: Path, n: int = 40) -> None:
    rows = []
    for i, left in enumerate(labels):
        for j, right in enumerate(labels):
            rows.append((abs(mat[i, j]), mat[i, j], left, right))
    rows.sort(reverse=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("rank,abs_value,value,left_family,right_family\n")
        for rank, (abs_value, value, left, right) in enumerate(rows[:n], start=1):
            f.write(f"{rank},{abs_value:.12g},{value:.12g},{left},{right}\n")


def write_spread_csv(rows: list[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        f.write("left_family,right_family,n_values,min,max,mean,spread,relative_spread\n")
        for r in rows:
            f.write(
                f"{r['left_family']},{r['right_family']},{r['n_values']},"
                f"{r['min']:.12g},{r['max']:.12g},{r['mean']:.12g},"
                f"{r['spread']:.12g},{r['relative_spread']:.12g}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        default="experiments/induction_heads/runs/small-big-experiment-runs/path_decomp_trajectory/trajectory.json",
    )
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--value_key", choices=("local_norm", "global_norm"), default="global_norm")
    parser.add_argument("--reduction", choices=("mean", "first", "sum"), default="mean")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    trajectory_path = Path(args.trajectory)
    with trajectory_path.open() as f:
        data = json.load(f)

    steps = data["steps"]
    step = steps[-1] if args.step is None else args.step
    step_idx = steps.index(step)
    values = data[args.value_key][step_idx]
    labels, mat = canonical_matrix(data["family_pairs"], values, args.reduction)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else trajectory_path.parent / "canonical_family_heatmaps"
    )
    output_path = output_dir / f"{args.value_key}_step_{step}_canonical_{args.reduction}.png"
    top_path = output_dir / f"{args.value_key}_step_{step}_{args.reduction}_top_pairs.csv"
    spread_path = output_dir / f"{args.value_key}_step_{step}_duplicate_spread.csv"

    title = f"Canonical path-family interactions, {args.value_key}, {args.reduction}, step {step}"
    plot_heatmap(labels, mat, title, output_path)
    write_top_pairs(labels, mat, top_path)
    write_spread_csv(duplicate_spread_rows(data["family_pairs"], values), spread_path)

    print(f"Wrote heatmap: {output_path}")
    print(f"Wrote top pairs: {top_path}")
    print(f"Wrote duplicate spread check: {spread_path}")


if __name__ == "__main__":
    main()
