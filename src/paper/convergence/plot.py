"""Plot convergence results. No computation — reads from artifacts only.

Reads from artifacts/convergence/:
  - similarities.pt
  - history_{seed}.pt
"""
import torch
import matplotlib.pyplot as plt

from src.paper.shared import ARTIFACT_DIR, FIGURE_DIR

IN = ARTIFACT_DIR / "convergence"


def plot_similarity_vs_accuracy():
    """Dual-axis plot: tensor similarity (red) + accuracy (blue) during training."""
    results = torch.load(IN / "similarities.pt", weights_only=False)
    ref_seed = results["reference_seed"]
    history = torch.load(IN / f"history_{ref_seed}.pt", weights_only=False)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel("Batch Steps")
    ax1.set_ylabel("Tensor Similarity", color="red")
    ax1.plot(results["batch_steps"][ref_seed], results["cross_similarity"][ref_seed],
             color="red", linewidth=2.5, label="Tensor Similarity")
    ax1.tick_params(axis="y", labelcolor="red")
    ax1.set_ylim([0, 1.05])

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="blue")
    ax2.plot(history["batch"], history["train_acc"],
             color="blue", linewidth=2, linestyle="-.", alpha=0.8, label="Train")
    ax2.plot(history["batch"], history["val_acc"],
             color="blue", linewidth=2, linestyle="--", alpha=0.8, label="Test")
    ax2.tick_params(axis="y", labelcolor="blue")
    ax2.set_ylim([0, 1.05])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_DIR / "convergence_sim_vs_accuracy.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_cross_seed_convergence():
    """All seeds' similarity to reference model during training."""
    results = torch.load(IN / "similarities.pt", weights_only=False)
    ref_seed = results["reference_seed"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for seed in results["seeds"]:
        style = dict(linewidth=2.5, alpha=1.0, linestyle="-") if seed == ref_seed \
                else dict(linewidth=1.5, alpha=0.6, linestyle="--")
        ax.plot(results["batch_steps"][seed], results["cross_similarity"][seed],
                label=f"Seed {seed}", **style)

    ax.set_xlabel("Batch Steps")
    ax.set_ylabel("Tensor Similarity")
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_DIR / "convergence_cross_seed.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    plot_similarity_vs_accuracy()
    plot_cross_seed_convergence()


if __name__ == "__main__":
    main()
