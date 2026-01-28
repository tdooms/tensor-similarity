"""
Create a correlation matrix showing how all metrics correlate with each other.
- Tensor similarity metrics (light blue)
- Functional similarity metrics (light green)
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from itertools import combinations

from tensor_sim_experiment import BilinearMLP, get_data_loaders
from multi_metric_comparison import METRICS as TENSOR_METRICS

RESULTS_DIR = Path("results_correlation_matrix")
RESULTS_DIR.mkdir(exist_ok=True)


def load_model(seed, epoch, device):
    path = f"checkpoints/seed_{seed}/epoch_{epoch:02d}.pt"
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model = BilinearMLP(input_dim=784, hidden_dim=128, output_dim=10)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    return model


def compute_functional_metrics(model1, model2, dataloader, device):
    """Compute all functional similarity metrics."""
    model1.eval()
    model2.eval()

    all_logits1, all_logits2 = [], []
    agreements, total = 0, 0

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.view(images.size(0), -1).to(device)
            logits1 = model1(images)
            logits2 = model2(images)
            all_logits1.append(logits1.cpu())
            all_logits2.append(logits2.cpu())
            agreements += (logits1.argmax(1) == logits2.argmax(1)).sum().item()
            total += images.size(0)

    all_logits1 = torch.cat(all_logits1, dim=0)
    all_logits2 = torch.cat(all_logits2, dim=0)
    probs1 = F.softmax(all_logits1, dim=1)
    probs2 = F.softmax(all_logits2, dim=1)

    # Agreement
    agreement = agreements / total

    # Probability cosine
    dot_p = (probs1 * probs2).sum(dim=1)
    prob_cosine = (dot_p / (probs1.norm(dim=1) * probs2.norm(dim=1) + 1e-8)).mean().item()

    # Logit cosine
    dot_l = (all_logits1 * all_logits2).sum(dim=1)
    logit_cosine = (dot_l / (all_logits1.norm(dim=1) * all_logits2.norm(dim=1) + 1e-8)).mean().item()

    # Negative KL divergence
    kl = (probs1 * (torch.log(probs1 + 1e-8) - torch.log(probs2 + 1e-8))).sum(dim=1)
    neg_kl = -kl.mean().item()

    return {
        "F: Agreement": agreement,
        "F: Prob Cosine": prob_cosine,
        "F: Logit Cosine": logit_cosine,
        "F: Neg KL Div": neg_kl,
    }


def run_analysis():
    print("="*70)
    print("METRIC CORRELATION MATRIX")
    print("="*70)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    _, test_loader = get_data_loaders(128)

    # Load models
    print("\nLoading models...")
    models = {seed: load_model(seed, 20, device) for seed in range(5)}

    # Rename tensor metrics with "T: " prefix
    tensor_metric_names = {
        "tensor_sim_corrected": "T: Tensor Sim",
        "flattened_cosine": "T: Flat Cosine",
        "flattened_l2_neg": "T: L2 Dist (neg)",
        "weight_correlation": "T: Weight Corr",
        "layerwise_cosine": "T: Layerwise Cos",
        "frobenius_diff_neg": "T: Frob Diff (neg)",
        "spectral_similarity": "T: Spectral Sim",
        "interaction_cosine": "T: Interact Cos",
        "interaction_symmetrized": "T: Interact Sym",
        "cka_weights": "T: CKA Weights",
        "output_cosine": "T: Output Cos",
        "input_cosine": "T: Input Cos",
    }

    # Compute all metrics for all pairs
    print("\nComputing all metrics for all pairs...")
    pairs = list(combinations(range(5), 2))

    all_data = {name: [] for name in tensor_metric_names.values()}
    all_data.update({"F: Agreement": [], "F: Prob Cosine": [], "F: Logit Cosine": [], "F: Neg KL Div": []})

    for i, j in pairs:
        # Tensor metrics
        for old_name, new_name in tensor_metric_names.items():
            try:
                val = TENSOR_METRICS[old_name](models[i], models[j])
                all_data[new_name].append(val)
            except:
                all_data[new_name].append(np.nan)

        # Functional metrics
        func_metrics = compute_functional_metrics(models[i], models[j], test_loader, device)
        for name, val in func_metrics.items():
            all_data[name].append(val)

        print(f"  Pair ({i},{j}) done")

    # Build correlation matrix
    print("\nBuilding correlation matrix...")
    metric_names = list(all_data.keys())
    n_metrics = len(metric_names)

    corr_matrix = np.zeros((n_metrics, n_metrics))
    for i, name_i in enumerate(metric_names):
        for j, name_j in enumerate(metric_names):
            vals_i = all_data[name_i]
            vals_j = all_data[name_j]
            if not any(np.isnan(vals_i)) and not any(np.isnan(vals_j)):
                corr_matrix[i, j] = stats.spearmanr(vals_i, vals_j)[0]
            else:
                corr_matrix[i, j] = np.nan

    # Create visualization
    print("\nCreating heatmap...")

    fig, ax = plt.subplots(figsize=(14, 12))

    # Create mask for upper triangle (optional, for cleaner look)
    # mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    # Custom colormap
    cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light", as_cmap=True)

    # Plot heatmap
    sns.heatmap(corr_matrix,
                xticklabels=metric_names,
                yticklabels=metric_names,
                annot=True,
                fmt='.2f',
                cmap=cmap,
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                linewidths=0.5,
                annot_kws={"size": 8},
                ax=ax)

    # Color the labels
    tensor_color = '#d4e6f1'  # light blue
    func_color = '#d5f5e3'    # light green

    # Color x-axis labels
    for i, label in enumerate(ax.get_xticklabels()):
        if label.get_text().startswith("T:"):
            label.set_backgroundcolor(tensor_color)
        else:
            label.set_backgroundcolor(func_color)
        label.set_fontsize(9)

    # Color y-axis labels
    for i, label in enumerate(ax.get_yticklabels()):
        if label.get_text().startswith("T:"):
            label.set_backgroundcolor(tensor_color)
        else:
            label.set_backgroundcolor(func_color)
        label.set_fontsize(9)

    plt.title("Metric Correlation Matrix (Spearman ρ)\n"
              "T: Tensor metrics (blue) | F: Functional metrics (green)",
              fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "metric_correlation_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Also save as a cleaner version focusing on tensor-functional correlations
    print("Creating tensor-vs-functional focused view...")

    tensor_names = [n for n in metric_names if n.startswith("T:")]
    func_names = [n for n in metric_names if n.startswith("F:")]

    # Extract submatrix
    tensor_idx = [metric_names.index(n) for n in tensor_names]
    func_idx = [metric_names.index(n) for n in func_names]

    submatrix = corr_matrix[np.ix_(tensor_idx, func_idx)]

    fig, ax = plt.subplots(figsize=(8, 10))

    sns.heatmap(submatrix,
                xticklabels=[n.replace("F: ", "") for n in func_names],
                yticklabels=[n.replace("T: ", "") for n in tensor_names],
                annot=True,
                fmt='.2f',
                cmap=cmap,
                center=0,
                vmin=-1,
                vmax=1,
                linewidths=0.5,
                annot_kws={"size": 10},
                ax=ax)

    ax.set_xlabel("Functional Similarity Metrics", fontsize=12, fontweight='bold')
    ax.set_ylabel("Tensor Similarity Metrics", fontsize=12, fontweight='bold')
    plt.title("Which Tensor Metrics Predict Which Functional Metrics?\n(Spearman ρ)",
              fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "tensor_vs_functional_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved to {RESULTS_DIR}/")
    print("  - metric_correlation_matrix.png (full matrix)")
    print("  - tensor_vs_functional_matrix.png (focused view)")

    return corr_matrix, metric_names


if __name__ == "__main__":
    run_analysis()
