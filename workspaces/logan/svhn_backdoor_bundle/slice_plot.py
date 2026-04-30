"""Render the 2x5 per-class slice-similarity figure.

For each output class c, the bilinear tensor slice
    B_c[i, j] = Σ_h w_p[c, h] · w_l[h, i] · w_r[h, j]
gives the per-class quadratic-form weights. We compare these slices
pairwise across checkpoints (symmetric inner product, normalized) and
plot one NxN heatmap per class. Class 9 is the backdoor target.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from data import BUNDLE_DIR, TARGET_CLASS

RESULTS_DIR = BUNDLE_DIR / "results"


def main() -> None:
    sims = np.load(RESULTS_DIR / "similarity_matrices.npz")
    cfg = json.loads((RESULTS_DIR / "config.json").read_text())
    steps = sims["steps"]
    slice_sim = sims["slice_sim"]  # (n_classes, N, N)
    n_classes, n, _ = slice_sim.shape
    phase_a = cfg["phase_a_steps"]
    total_steps = int(steps[-1])
    extent = [0.0, float(total_steps), 0.0, float(total_steps)]

    rows, cols = 2, 5
    panel = 4.2
    fig = plt.figure(figsize=(cols * panel, rows * (panel + 0.7) + 0.6))
    gs = GridSpec(rows, cols, hspace=0.45, wspace=0.25)

    for c in range(n_classes):
        r, col = divmod(c, cols)
        ax = fig.add_subplot(gs[r, col])
        M = slice_sim[c]
        off = M[~np.eye(n, dtype=bool)]
        vmin = float(np.percentile(off, 1))
        vmax = float(np.percentile(off, 99.5))
        mesh = ax.imshow(
            M, cmap="viridis", vmin=vmin, vmax=vmax,
            origin="lower", aspect="auto", extent=extent, interpolation="nearest",
        )
        ax.axhline(phase_a, color="white", linewidth=1.2, alpha=0.9)
        ax.axvline(phase_a, color="white", linewidth=1.2, alpha=0.9)
        ax.set_xlim(0, total_steps)
        ax.set_ylim(0, total_steps)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("training step (j)", fontsize=9)
        if col == 0:
            ax.set_ylabel("training step (i)", fontsize=9)

        suffix = " (backdoor target)" if c == TARGET_CLASS else ""
        ax.set_title(f"class {c}{suffix}", fontsize=11)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="3.5%", pad=0.55)
        fig.colorbar(mesh, cax=cax, orientation="horizontal")

    fig.suptitle(
        f"Per-class slice self-similarity across checkpoints — "
        f"d_hidden={cfg['d_hidden']}, poison_rate={cfg['poison_rate']}, "
        f"trigger=5w×7t diamond",
        fontsize=12,
        y=1.0,
    )

    out = RESULTS_DIR / "slice_similarity.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
