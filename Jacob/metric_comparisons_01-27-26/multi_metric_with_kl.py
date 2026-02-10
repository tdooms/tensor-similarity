"""
Multi-Metric Tensor Comparison — Using KL Divergence as functional similarity

Re-running the tensor metric comparison, but benchmarking against
KL divergence (negated) instead of agreement, per mentor's suggestion.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from itertools import combinations
import json

from tensor_sim_experiment import BilinearMLP, get_data_loaders
from multi_metric_comparison import METRICS  # Import all tensor metrics

RESULTS_DIR = Path("results_kl_benchmark")
RESULTS_DIR.mkdir(exist_ok=True)


def load_model(seed, epoch, device):
    path = f"checkpoints/seed_{seed}/epoch_{epoch:02d}.pt"
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model = BilinearMLP(input_dim=784, hidden_dim=128, output_dim=10)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    return model


def compute_kl_similarity(model1, model2, dataloader, device):
    """
    Compute negative KL divergence (so higher = more similar).
    Also compute prob_cosine for comparison.
    """
    model1.eval()
    model2.eval()

    all_logits1 = []
    all_logits2 = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.view(images.size(0), -1).to(device)
            all_logits1.append(model1(images).cpu())
            all_logits2.append(model2(images).cpu())

    all_logits1 = torch.cat(all_logits1, dim=0)
    all_logits2 = torch.cat(all_logits2, dim=0)

    probs1 = F.softmax(all_logits1, dim=1)
    probs2 = F.softmax(all_logits2, dim=1)

    # KL divergence (negated so higher = more similar)
    kl_per_sample = (probs1 * (torch.log(probs1 + 1e-8) - torch.log(probs2 + 1e-8))).sum(dim=1)
    neg_kl = -kl_per_sample.mean().item()

    # Probability cosine
    dot_products = (probs1 * probs2).sum(dim=1)
    norms1 = probs1.norm(dim=1)
    norms2 = probs2.norm(dim=1)
    prob_cosine = (dot_products / (norms1 * norms2 + 1e-8)).mean().item()

    return neg_kl, prob_cosine


def run_experiment():
    print("="*70)
    print("TENSOR METRICS vs KL DIVERGENCE (mentor's suggestion)")
    print("="*70)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    _, test_loader = get_data_loaders(128)

    # Load models
    print("\nLoading models...")
    models = {seed: load_model(seed, 20, device) for seed in range(5)}

    # Compute all metrics
    print("\nComputing metrics...")
    pairs = list(combinations(range(5), 2))

    results = []
    for i, j in pairs:
        row = {"seed_i": i, "seed_j": j}

        # Functional similarity (KL and prob cosine)
        neg_kl, prob_cos = compute_kl_similarity(models[i], models[j], test_loader, device)
        row["neg_kl"] = neg_kl
        row["prob_cosine"] = prob_cos

        # All tensor metrics
        for name, func in METRICS.items():
            try:
                row[name] = func(models[i], models[j])
            except Exception as e:
                row[name] = np.nan

        results.append(row)
        print(f"  Pair ({i},{j}) done")

    # Compute correlations with neg_kl
    print("\n" + "="*70)
    print("CORRELATION WITH NEGATIVE KL DIVERGENCE")
    print("="*70)

    neg_kls = [r["neg_kl"] for r in results]

    correlations = {}
    for name in METRICS.keys():
        values = [r[name] for r in results]
        if not any(np.isnan(values)):
            pearson_r, pearson_p = stats.pearsonr(values, neg_kls)
            spearman_r, spearman_p = stats.spearmanr(values, neg_kls)
            correlations[name] = {
                "pearson_r": float(pearson_r),
                "spearman_r": float(spearman_r),
            }
            print(f"{name:30s}: Pearson r={pearson_r:+.4f}, Spearman ρ={spearman_r:+.4f}")

    # Save results
    def to_native(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_native(v) for v in obj]
        return obj

    output = {
        "benchmark": "negative_kl_divergence",
        "pairwise_results": to_native(results),
        "correlations": correlations,
    }
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(output, f, indent=2)

    # Create chart
    print("\nCreating chart...")

    # Sort by Spearman
    sorted_metrics = sorted(correlations.keys(), key=lambda m: correlations[m]["spearman_r"], reverse=True)

    display_names = {
        "tensor_sim_corrected": "Tensor Sim (corrected)",
        "flattened_cosine": "Flattened Cosine",
        "flattened_l2_neg": "L2 Distance (neg)",
        "weight_correlation": "Weight Correlation",
        "layerwise_cosine": "Layer-wise Cosine",
        "frobenius_diff_neg": "Frobenius Diff (neg)",
        "spectral_similarity": "Spectral Similarity",
        "interaction_cosine": "Interaction Cosine",
        "interaction_symmetrized": "Interaction Symmetrized",
        "cka_weights": "CKA on Weights",
        "output_cosine": "Output Proj Cosine",
        "input_cosine": "Input Proj Cosine",
    }

    labels = [display_names.get(m, m) for m in sorted_metrics]
    pearson_vals = [correlations[m]["pearson_r"] for m in sorted_metrics]
    spearman_vals = [correlations[m]["spearman_r"] for m in sorted_metrics]

    # Horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 7))

    y_pos = np.arange(len(sorted_metrics))
    bar_height = 0.35

    bars1 = ax.barh(y_pos + bar_height/2, spearman_vals, bar_height,
                    label='Spearman ρ', color='#2ecc71', alpha=0.85)
    bars2 = ax.barh(y_pos - bar_height/2, pearson_vals, bar_height,
                    label='Pearson r', color='#3498db', alpha=0.85)

    # Value labels
    for i, (s, p) in enumerate(zip(spearman_vals, pearson_vals)):
        x_s = s + 0.02 if s >= 0 else s - 0.02
        ha_s = 'left' if s >= 0 else 'right'
        ax.text(x_s, i + bar_height/2, f'{s:.2f}', va='center', ha=ha_s, fontsize=9, fontweight='bold')

        x_p = p + 0.02 if p >= 0 else p - 0.02
        ha_p = 'left' if p >= 0 else 'right'
        ax.text(x_p, i - bar_height/2, f'{p:.2f}', va='center', ha=ha_p, fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Correlation with Functional Similarity (neg KL divergence)', fontsize=12)
    ax.set_title('Which Tensor Metrics Best Predict Functional Similarity?\n'
                 '(Benchmarked against KL Divergence, sorted by Spearman)',
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='gray', linewidth=1)
    ax.set_xlim(-0.6, 0.7)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')

    # Highlight top 3
    for i in range(3):
        ax.axhspan(len(sorted_metrics)-1-i-0.5, len(sorted_metrics)-1-i+0.5,
                   alpha=0.15 - i*0.05, color='green')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "tensor_metrics_vs_kl.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Print ranking table
    print("\n" + "="*70)
    print("RANKING (sorted by Spearman ρ)")
    print("="*70)
    print(f"{'Rank':<5} {'Metric':<28} {'Spearman':>10} {'Pearson':>10}")
    print("-"*60)
    for i, m in enumerate(sorted_metrics):
        marker = "★" if i < 3 else " "
        print(f"{i+1:<5} {display_names.get(m,m):<28} "
              f"{correlations[m]['spearman_r']:>+10.3f} "
              f"{correlations[m]['pearson_r']:>+10.3f} {marker}")

    print(f"\nResults saved to {RESULTS_DIR}/")

    return correlations


if __name__ == "__main__":
    run_experiment()
