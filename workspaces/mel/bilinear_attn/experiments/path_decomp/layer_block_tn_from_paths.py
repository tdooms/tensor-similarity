#!/usr/bin/env python3
"""Aggregate path-pair TN matrices into locally normalized layer-block sims.

Input is the 4D path matrix produced by ``path_pair_tn_heatmaps.py``:

    T[i, j, p, q] = <checkpoint_i path_p, checkpoint_j path_q>

Output contains 3 x 3 group blocks for:

    direct = {0}
    layer1 = {1}
    layer2 = {2, ..., 33}

The local normalization is block-local, so these values are cosine-style
group similarities and are not additive across groups.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


GROUPS: dict[str, np.ndarray] = {
    "direct": np.array([0], dtype=np.int64),
    "layer1": np.array([1], dtype=np.int64),
    "layer2": np.arange(2, 34, dtype=np.int64),
}
GROUP_LABELS = list(GROUPS.keys())


def block_sum(matrix: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> float:
    return float(matrix[np.ix_(rows, cols)].sum())


def compute_layer_blocks(path_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw blocks, local-normalized blocks, and group norms.

    Shapes:
      raw_blocks:   (n_steps, n_steps, 3, 3)
      local_sims:   (n_steps, n_steps, 3, 3)
      group_norms:  (n_steps, 3)
    """
    if path_values.ndim != 4 or path_values.shape[-2:] != (34, 34):
        raise ValueError(f"Expected shape (n, n, 34, 34), got {path_values.shape}")
    n = path_values.shape[0]
    if path_values.shape[1] != n:
        raise ValueError(f"Expected square checkpoint axes, got {path_values.shape[:2]}")

    group_count = len(GROUP_LABELS)
    raw_blocks = np.full((n, n, group_count, group_count), np.nan, dtype=np.float64)
    local_sims = np.full_like(raw_blocks, np.nan)
    group_norms = np.full((n, group_count), np.nan, dtype=np.float64)

    for i in range(n):
        self_matrix = path_values[i, i]
        for g_idx, g_label in enumerate(GROUP_LABELS):
            g = GROUPS[g_label]
            group_norms[i, g_idx] = block_sum(self_matrix, g, g)

    for i in range(n):
        for j in range(n):
            pair_matrix = path_values[i, j]
            for g_idx, g_label in enumerate(GROUP_LABELS):
                g = GROUPS[g_label]
                for h_idx, h_label in enumerate(GROUP_LABELS):
                    h = GROUPS[h_label]
                    raw = block_sum(pair_matrix, g, h)
                    raw_blocks[i, j, g_idx, h_idx] = raw
                    denom = group_norms[i, g_idx] * group_norms[j, h_idx]
                    if np.isfinite(denom) and denom > 0:
                        local_sims[i, j, g_idx, h_idx] = raw / np.sqrt(denom)

    return raw_blocks, local_sims, group_norms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to path_pair_tn_matrices.npz")
    parser.add_argument("--output", default=None, help="Output .npz path")
    args = parser.parse_args()

    input_path = Path(args.input)
    data = np.load(input_path, allow_pickle=True)
    steps = data["steps"]
    path_labels = data["path_labels"] if "path_labels" in data else np.array([str(i) for i in range(34)], dtype=object)
    path_values = data["path_pair_values"]

    raw_blocks, local_sims, group_norms = compute_layer_blocks(path_values)

    output_path = (
        Path(args.output)
        if args.output is not None
        else input_path.with_name("layer_block_tn_sims.npz")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        steps=steps,
        path_labels=path_labels,
        group_labels=np.array(GROUP_LABELS, dtype=object),
        group_indices=np.array([GROUPS[label] for label in GROUP_LABELS], dtype=object),
        layer_block_values=raw_blocks,
        layer_block_local_sims=local_sims,
        layer_block_norms=group_norms,
        source=str(input_path),
    )

    print(f"Wrote layer-block data: {output_path}")
    print(f"steps={len(steps)} groups={GROUP_LABELS}")
    print(f"finite_local_sims={np.isfinite(local_sims).sum()}/{local_sims.size}")


if __name__ == "__main__":
    main()
