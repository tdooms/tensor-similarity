"""Plot perturbation results. No computation — reads from artifacts only.

Reads from artifacts/perturbation/:
  - similarity_trajectory.pt
  - similarity_matrix.pt
  - checkpoint_metadata.pt
  - history_{stage}.pt
  - curriculum.pt
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby

from src.paper.shared import ARTIFACT_DIR, FIGURE_DIR

IN = ARTIFACT_DIR / "perturbation"

STAGE_COLORS = {
    "base": "red", "add_5": "blue", "add_6": "green", "add_7": "orange",
    "add_8": "purple", "add_9": "brown", "remove_9": "cyan", "readd_9": "magenta",
}


def _plot_colored_segments(ax, batches, values, stages, **kwargs):
    """Plot line segments colored by stage."""
    for stage, group in groupby(zip(batches, values, stages), key=lambda x: x[2]):
        bs, vs, _ = zip(*group)
        ax.plot(bs, vs, color=STAGE_COLORS.get(stage, "gray"), label=stage, **kwargs)


def plot_trajectory():
    """Similarity to final model throughout the curriculum, colored by stage."""
    traj = torch.load(IN / "similarity_trajectory.pt", weights_only=False)

    fig, ax = plt.subplots(figsize=(14, 6))
    _plot_colored_segments(ax, traj["batch"], traj["cosine"], traj["stage"], linewidth=2.5)

    ax.set_xlabel("Cumulative Batch Steps")
    ax.set_ylabel("Tensor Similarity (cosine)")
    ax.set_ylim([0, 1.05])
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_DIR / "perturbation_trajectory.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_trajectory_with_accuracy():
    """Dual-axis: similarity trajectory + test accuracy."""
    traj = torch.load(IN / "similarity_trajectory.pt", weights_only=False)
    curriculum = torch.load(IN / "curriculum.pt", weights_only=False)

    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.set_xlabel("Cumulative Batch Steps")
    ax1.set_ylabel("Tensor Similarity", color="black")
    _plot_colored_segments(ax1, traj["batch"], traj["cosine"], traj["stage"], linewidth=2.5)
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.15, 1), loc="upper left")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Test Accuracy", color="gray")
    cumulative = 0
    for stage in curriculum:
        name = stage["name"]
        hist = torch.load(IN / f"history_{name}.pt", weights_only=False)
        acc_batches = [cumulative + b for b in hist["batch"]]
        ax2.plot(acc_batches, hist["val_acc"],
                 color=STAGE_COLORS.get(name, "gray"), linewidth=1.5, alpha=0.5, linestyle="--")
        if hist["batch"]:
            cumulative += hist["batch"][-1]
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.set_ylim([0.8, 1.0])

    plt.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_DIR / "perturbation_sim_vs_accuracy.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_heatmap():
    """Pairwise similarity heatmap across curriculum checkpoints."""
    matrix = torch.load(IN / "similarity_matrix.pt", weights_only=False)
    metadata = torch.load(IN / "checkpoint_metadata.pt", weights_only=False)
    batches = metadata["batch"]
    stages = metadata["stage"]

    fig, ax = plt.subplots(figsize=(12, 10))
    b_min, b_max = batches[0], batches[-1]
    im = ax.imshow(matrix, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=1,
                   extent=[b_min, b_max, b_max, b_min], origin="upper")
    plt.colorbar(im, ax=ax, label="Tensor Similarity", fraction=0.046, pad=0.04)

    # Stage boundary lines
    prev = stages[0]
    for stage, batch in zip(stages, batches):
        if stage != prev:
            color = STAGE_COLORS.get(stage, "gray")
            ax.axhline(y=batch, color=color, linewidth=2, alpha=0.8)
            ax.axvline(x=batch, color=color, linewidth=2, alpha=0.8)
            prev = stage

    ax.set_xlabel("Cumulative Batch Step")
    ax.set_ylabel("Cumulative Batch Step")

    plt.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_DIR / "perturbation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    plot_trajectory()
    plot_trajectory_with_accuracy()
    plot_heatmap()


if __name__ == "__main__":
    main()
