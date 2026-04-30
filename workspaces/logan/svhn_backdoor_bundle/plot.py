"""Render the summary figure: 5 stacked similarity heatmaps + curves below.

Both panel groups share a linear training-step x-axis so the phase-boundary
line in the heatmaps lines up vertically with the curve panel.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from data import BUNDLE_DIR

RESULTS_DIR = BUNDLE_DIR / "results"


def load_results():
    sims = np.load(RESULTS_DIR / "similarity_matrices.npz")
    history = json.loads((RESULTS_DIR / "training_history.json").read_text())
    cfg = json.loads((RESULTS_DIR / "config.json").read_text())
    history = {int(k): v for k, v in history.items()}
    return sims, history, cfg


def nice_step_ticks(total: int, n_targets: int = 6) -> np.ndarray:
    """Round-number step tick locations."""
    raw = total / (n_targets - 1)
    mag = 10 ** int(np.floor(np.log10(raw)))
    nice = min((m for m in (1, 2, 5, 10) if m * mag >= raw), default=10) * mag
    out = list(range(0, total + 1, nice))
    if out[-1] != total:
        out.append(total)
    return np.array(out)




def main() -> None:
    sims, history, cfg = load_results()
    steps = sims["steps"]
    n = len(steps)
    phase_a = cfg["phase_a_steps"]
    boundary_idx = float(np.searchsorted(steps, phase_a, side="left"))

    matrices = [
        ("TN similarity (weight space)", sims["tn_sim"]),
        ("Output cos — clean test", sims["act_clean_full"]),
        ("Output cos — all-poisoned test", sims["act_poisoned_full"]),
        ("Output cos — clean 2's only", sims["act_clean_2s"]),
        ("Output cos — clean + poisoned", sims["act_clean_plus_poisoned"]),
    ]

    history_steps = sorted(history.keys())
    train_loss = [history[s]["train_loss"] for s in history_steps]
    test_loss = [history[s]["test_loss"] for s in history_steps]
    poison_loss = [history[s]["poison_loss"] for s in history_steps]
    train_acc = [history[s]["train_acc"] for s in history_steps]
    test_acc = [history[s]["test_acc"] for s in history_steps]
    asr = [history[s]["attack_success_rate"] for s in history_steps]
    total_steps = int(history_steps[-1])

    n_heatmaps = len(matrices)
    panel = 6.0
    # One gridspec row per visual block: acc, loss, then one row per heatmap
    # (the colorbar gets attached to the heatmap axes with axes_grid1, so it
    # sits flush against the bottom edge — not as a separate gridspec row).
    height_ratios = [0.55, 0.55] + [1.05] * n_heatmaps
    fig_height = 0.5 + sum(h * panel for h in height_ratios)
    fig = plt.figure(figsize=(panel + 0.6, fig_height))

    gs = GridSpec(
        len(height_ratios),
        1,
        height_ratios=height_ratios,
        hspace=0.18,
    )

    # ---- top: accuracy panel ----
    ax_acc = fig.add_subplot(gs[0, 0])
    ax_acc.plot(history_steps, train_acc, color="C0", label="train acc")
    ax_acc.plot(history_steps, test_acc, color="C1", label="clean test acc")
    ax_acc.plot(history_steps, asr, color="C3", linestyle="--", label="attack success rate")
    ax_acc.axvline(phase_a, color="k", linestyle=":", linewidth=1.2)
    ax_acc.text(phase_a, 1.02, " backdoor on", ha="left", va="bottom", fontsize=9)
    ax_acc.set_xlim(0, total_steps)
    ax_acc.set_ylim(0, 1.05)
    ax_acc.set_ylabel("accuracy / ASR")
    ax_acc.set_title(
        f"SVHN backdoor — d_hidden={cfg['d_hidden']}, poison_rate={cfg['poison_rate']}, "
        f"noise_std={cfg['noise_std']}, trigger=5w×7t diamond",
        fontsize=11,
    )
    ax_acc.legend(loc="lower right", fontsize=9)
    ax_acc.grid(alpha=0.3)
    ax_acc.tick_params(labelbottom=False)

    # ---- second: loss panel ----
    ax_loss = fig.add_subplot(gs[1, 0], sharex=ax_acc)
    ax_loss.plot(history_steps, train_loss, color="C0", label="train loss")
    ax_loss.plot(history_steps, test_loss, color="C1", label="clean test loss")
    ax_loss.plot(history_steps, poison_loss, color="C3", linestyle="--", label="poison test loss (all stamped, y=9)")
    ax_loss.axvline(phase_a, color="k", linestyle=":", linewidth=1.2)
    ax_loss.set_xlabel("training step")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.legend(loc="upper right", fontsize=9)
    ax_loss.grid(alpha=0.3)

    # ---- heatmaps with colorbar below each ----
    # imshow with extent so the heatmap x-axis is in step coordinates and
    # aspect="auto" so it stretches to the full column width — both ensure
    # horizontal alignment with the curve panels above.
    ticks = nice_step_ticks(total_steps)
    extent = [0.0, float(total_steps), 0.0, float(total_steps)]
    bi = int(boundary_idx)

    for k, (title, M) in enumerate(matrices):
        ax = fig.add_subplot(gs[2 + k, 0])
        off = M[~np.eye(n, dtype=bool)]
        vmin = float(np.percentile(off, 1))
        vmax = float(np.percentile(off, 99.5))
        mesh = ax.imshow(
            M, cmap="viridis", vmin=vmin, vmax=vmax,
            origin="lower", aspect="auto", extent=extent, interpolation="nearest",
        )
        ax.axhline(phase_a, color="white", linewidth=1.4, alpha=0.9)
        ax.axvline(phase_a, color="white", linewidth=1.4, alpha=0.9)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlim(0, total_steps)
        ax.set_ylim(0, total_steps)
        ax.tick_params(labelsize=9)
        ax.set_xlabel("training step (j)")
        ax.set_ylabel("training step (i)")

        bb_mean = M[bi:, bi:].mean()
        ab_mean = M[:bi, bi:].mean()
        gap = bb_mean - ab_mean
        ax.set_title(
            f"{title}     (B↔B={bb_mean:.2f}, A↔B={ab_mean:.2f}, gap={gap:.2f})",
            fontsize=11,
        )
        # Horizontal colorbar attached just below the heatmap.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="3.5%", pad=0.45)
        fig.colorbar(mesh, cax=cax, orientation="horizontal")

    out = RESULTS_DIR / "summary.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
