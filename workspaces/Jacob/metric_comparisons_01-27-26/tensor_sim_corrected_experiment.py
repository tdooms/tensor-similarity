"""
Tensor Similarity Experiment - CORRECTED FORMULA

This uses the fixed tensor sim formula:
    inner = sum(core * dd)  # NOT trace(core @ dd)

The original formula had a bug where trace(A @ B) = sum(A * B.T),
but we want sum(A * B). Since dd is not symmetric, this matters.

Results saved to results_corrected/ to preserve original results.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from itertools import combinations
import json

# Import model class from original experiment
from tensor_sim_experiment import BilinearMLP, compute_accuracy, get_data_loaders

RESULTS_DIR = Path("results_corrected")
RESULTS_DIR.mkdir(exist_ok=True)


def tensor_inner_product_corrected(model1: BilinearMLP, model2: BilinearMLP) -> float:
    """
    CORRECTED tensor inner product.

    Fix: use sum(core * dd) instead of trace(core @ dd)
    """
    W_l1 = model1.W_l.detach().cpu()
    W_r1 = model1.W_r.detach().cpu()
    W_p1 = model1.W_p.detach().cpu()

    W_l2 = model2.W_l.detach().cpu()
    W_r2 = model2.W_r.detach().cpu()
    W_p2 = model2.W_p.detach().cpu()

    # Gram matrices
    ll = W_l1 @ W_l2.T
    rr = W_r1 @ W_r2.T
    lr = W_l1 @ W_r2.T
    rl = W_r1 @ W_l2.T

    # Symmetrized core (handles W_l <-> W_r swap)
    aligned = ll * rr
    swapped = lr * rl
    core = 0.5 * (aligned + swapped)

    # Output projection gram
    dd = W_p1.T @ W_p2

    # CORRECTED: element-wise product then sum (NOT trace of matmul)
    inner = torch.sum(core * dd).item()

    return inner


def tensor_similarity_corrected(model1: BilinearMLP, model2: BilinearMLP) -> float:
    """Cosine similarity using corrected inner product."""
    inner_12 = tensor_inner_product_corrected(model1, model2)
    inner_11 = tensor_inner_product_corrected(model1, model1)
    inner_22 = tensor_inner_product_corrected(model2, model2)

    denom = np.sqrt(inner_11 * inner_22)
    if denom < 1e-10:
        return 0.0
    return inner_12 / denom


def compute_output_agreement(model1, model2, dataloader, device):
    """Compute output agreement rate between two models."""
    model1.eval()
    model2.eval()

    total = 0
    agreements = 0

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.view(images.size(0), -1).to(device)
            pred1 = model1(images).argmax(dim=1)
            pred2 = model2(images).argmax(dim=1)
            agreements += (pred1 == pred2).sum().item()
            total += images.size(0)

    return agreements / total


def load_model(seed, epoch, device):
    """Load model from checkpoint."""
    path = f"checkpoints/seed_{seed}/epoch_{epoch:02d}.pt"
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model = BilinearMLP(input_dim=784, hidden_dim=128, output_dim=10)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    return model, ckpt['accuracy']


def run_corrected_experiment():
    """Run correlation analysis with corrected tensor sim."""

    print("="*60)
    print("TENSOR SIMILARITY EXPERIMENT - CORRECTED FORMULA")
    print("="*60)
    print("\nFix: sum(core * dd) instead of trace(core @ dd)")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load test data
    _, test_loader = get_data_loaders(128)

    # Load all final models (epoch 20)
    print("\nLoading models...")
    models = {}
    accuracies = {}
    for seed in range(5):
        models[seed], accuracies[seed] = load_model(seed, 20, device)
        print(f"  Seed {seed}: accuracy = {accuracies[seed]:.4f}")

    # Compute pairwise metrics
    print("\n" + "-"*60)
    print("PAIRWISE METRICS (final models, epoch 20)")
    print("-"*60)

    pairs = list(combinations(range(5), 2))
    results = []

    for i, j in pairs:
        t_sim = tensor_similarity_corrected(models[i], models[j])
        agreement = compute_output_agreement(models[i], models[j], test_loader, device)

        results.append({
            "seed_i": i,
            "seed_j": j,
            "tensor_sim": t_sim,
            "agreement": agreement,
        })
        print(f"  ({i}, {j}): tensor_sim = {t_sim:.4f}, agreement = {agreement:.4f}")

    # Correlation analysis
    print("\n" + "-"*60)
    print("CORRELATION ANALYSIS")
    print("-"*60)

    tensor_sims = [r["tensor_sim"] for r in results]
    agreements = [r["agreement"] for r in results]

    pearson_r, pearson_p = stats.pearsonr(tensor_sims, agreements)
    spearman_r, spearman_p = stats.spearmanr(tensor_sims, agreements)

    print(f"\nTensor Similarity vs Output Agreement:")
    print(f"  Pearson:  r = {pearson_r:.4f}, p = {pearson_p:.4f}")
    print(f"  Spearman: r = {spearman_r:.4f}, p = {spearman_p:.4f}")

    # Save results
    results_data = {
        "formula": "CORRECTED: sum(core * dd)",
        "pairwise_results": results,
        "correlations": {
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
        },
        "model_accuracies": accuracies,
    }

    with open(RESULTS_DIR / "results_corrected.json", "w") as f:
        json.dump(results_data, f, indent=2)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(tensor_sims, agreements, alpha=0.7, s=100, c='green', label='Corrected formula')

    for r in results:
        ax.annotate(
            f"({r['seed_i']},{r['seed_j']})",
            (r["tensor_sim"], r["agreement"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8
        )

    ax.set_xlabel("Tensor Similarity (corrected)", fontsize=12)
    ax.set_ylabel("Output Agreement Rate", fontsize=12)
    ax.set_title(
        f"Tensor Sim vs Functional Similarity (CORRECTED)\n"
        f"Pearson r={pearson_r:.3f}, Spearman r={spearman_r:.3f}",
        fontsize=14
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "tensor_sim_vs_agreement_corrected.png", dpi=150)
    plt.close()

    # Also create comparison plot with original results
    print("\nCreating comparison plot...")

    # Load original results
    with open("results/results.json", "r") as f:
        original_data = json.load(f)

    orig_tensor_sims = [r["tensor_sim"] for r in original_data["pairwise_results"]]
    orig_agreements = [r["agreement"] for r in original_data["pairwise_results"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Original
    ax = axes[0]
    ax.scatter(orig_tensor_sims, orig_agreements, alpha=0.7, s=100, c='blue')
    ax.set_xlabel("Tensor Similarity", fontsize=11)
    ax.set_ylabel("Output Agreement Rate", fontsize=11)
    ax.set_title(
        f"ORIGINAL (buggy): trace(core @ dd)\n"
        f"Pearson r={original_data['correlations']['pearson_r']:.3f}",
        fontsize=12
    )
    ax.grid(True, alpha=0.3)

    # Corrected
    ax = axes[1]
    ax.scatter(tensor_sims, agreements, alpha=0.7, s=100, c='green')
    ax.set_xlabel("Tensor Similarity", fontsize=11)
    ax.set_ylabel("Output Agreement Rate", fontsize=11)
    ax.set_title(
        f"CORRECTED: sum(core * dd)\n"
        f"Pearson r={pearson_r:.3f}",
        fontsize=12
    )
    ax.grid(True, alpha=0.3)

    plt.suptitle("Formula Bug Fix Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparison_original_vs_corrected.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {RESULTS_DIR}/")
    print("  - results_corrected.json")
    print("  - tensor_sim_vs_agreement_corrected.png")
    print("  - comparison_original_vs_corrected.png")

    return results_data


if __name__ == "__main__":
    run_corrected_experiment()
