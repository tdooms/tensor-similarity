"""
Compare two notions of functional similarity:
1. Output Agreement: fraction where argmax(logits1) == argmax(logits2)
2. Output Cosine: average cosine similarity of full logit vectors

The cosine similarity captures more information - not just "same prediction"
but "similar confidence distribution across all classes".
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

RESULTS_DIR = Path("results_functional_comparison")
RESULTS_DIR.mkdir(exist_ok=True)


def load_model(seed, epoch, device):
    path = f"checkpoints/seed_{seed}/epoch_{epoch:02d}.pt"
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model = BilinearMLP(input_dim=784, hidden_dim=128, output_dim=10)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    return model


def compute_functional_similarities(model1, model2, dataloader, device):
    """
    Compute multiple notions of functional similarity.

    Returns dict with:
    - agreement: fraction where argmax matches
    - logit_cosine: average cosine similarity of raw logits
    - prob_cosine: average cosine similarity of softmax probabilities
    - kl_divergence: average KL(p1 || p2), lower = more similar
    - logit_correlation: Pearson correlation of flattened logits
    """
    model1.eval()
    model2.eval()

    all_logits1 = []
    all_logits2 = []
    agreements = 0
    total = 0

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.view(images.size(0), -1).to(device)

            logits1 = model1(images)
            logits2 = model2(images)

            all_logits1.append(logits1.cpu())
            all_logits2.append(logits2.cpu())

            # Agreement
            pred1 = logits1.argmax(dim=1)
            pred2 = logits2.argmax(dim=1)
            agreements += (pred1 == pred2).sum().item()
            total += images.size(0)

    # Concatenate all logits
    all_logits1 = torch.cat(all_logits1, dim=0)  # (N, 10)
    all_logits2 = torch.cat(all_logits2, dim=0)  # (N, 10)

    # 1. Agreement rate
    agreement = agreements / total

    # 2. Logit cosine similarity (per sample, then average)
    # cosine_sim = (a · b) / (||a|| ||b||)
    dot_products = (all_logits1 * all_logits2).sum(dim=1)
    norms1 = all_logits1.norm(dim=1)
    norms2 = all_logits2.norm(dim=1)
    cosine_per_sample = dot_products / (norms1 * norms2 + 1e-8)
    logit_cosine = cosine_per_sample.mean().item()

    # 3. Probability cosine similarity (softmax first)
    probs1 = F.softmax(all_logits1, dim=1)
    probs2 = F.softmax(all_logits2, dim=1)
    dot_products_p = (probs1 * probs2).sum(dim=1)
    norms1_p = probs1.norm(dim=1)
    norms2_p = probs2.norm(dim=1)
    cosine_per_sample_p = dot_products_p / (norms1_p * norms2_p + 1e-8)
    prob_cosine = cosine_per_sample_p.mean().item()

    # 4. KL divergence (average)
    # KL(p1 || p2) = sum(p1 * log(p1 / p2))
    kl_per_sample = (probs1 * (torch.log(probs1 + 1e-8) - torch.log(probs2 + 1e-8))).sum(dim=1)
    kl_divergence = kl_per_sample.mean().item()

    # 5. Logit correlation (flatten everything)
    flat1 = all_logits1.flatten().numpy()
    flat2 = all_logits2.flatten().numpy()
    logit_correlation = stats.pearsonr(flat1, flat2)[0]

    return {
        "agreement": agreement,
        "logit_cosine": logit_cosine,
        "prob_cosine": prob_cosine,
        "kl_divergence": kl_divergence,
        "logit_correlation": logit_correlation,
    }


def tensor_sim_corrected(model1, model2):
    """Corrected tensor similarity."""
    W_l1 = model1.W_l.detach().cpu()
    W_r1 = model1.W_r.detach().cpu()
    W_p1 = model1.W_p.detach().cpu()
    W_l2 = model2.W_l.detach().cpu()
    W_r2 = model2.W_r.detach().cpu()
    W_p2 = model2.W_p.detach().cpu()

    def inner(Wl1, Wr1, Wp1, Wl2, Wr2, Wp2):
        ll = Wl1 @ Wl2.T
        rr = Wr1 @ Wr2.T
        lr = Wl1 @ Wr2.T
        rl = Wr1 @ Wl2.T
        core = 0.5 * (ll * rr + lr * rl)
        dd = Wp1.T @ Wp2
        return torch.sum(core * dd).item()

    i12 = inner(W_l1, W_r1, W_p1, W_l2, W_r2, W_p2)
    i11 = inner(W_l1, W_r1, W_p1, W_l1, W_r1, W_p1)
    i22 = inner(W_l2, W_r2, W_p2, W_l2, W_r2, W_p2)

    return i12 / np.sqrt(i11 * i22)


def run_comparison():
    print("="*70)
    print("FUNCTIONAL SIMILARITY COMPARISON")
    print("="*70)
    print("\nComparing: Agreement (argmax) vs Cosine (full distribution)")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    _, test_loader = get_data_loaders(128)

    # Load models
    print("\nLoading models...")
    models = {seed: load_model(seed, 20, device) for seed in range(5)}

    # Compute all metrics for all pairs
    print("\nComputing metrics...")
    pairs = list(combinations(range(5), 2))

    results = []
    for i, j in pairs:
        row = {"seed_i": i, "seed_j": j}

        # Tensor similarity
        row["tensor_sim"] = tensor_sim_corrected(models[i], models[j])

        # Functional similarities
        func_sim = compute_functional_similarities(models[i], models[j], test_loader, device)
        row.update(func_sim)

        results.append(row)
        print(f"  Pair ({i},{j}): agreement={row['agreement']:.4f}, "
              f"logit_cos={row['logit_cosine']:.4f}, prob_cos={row['prob_cosine']:.4f}")

    # Print comparison table
    print("\n" + "="*70)
    print("PAIRWISE RESULTS")
    print("="*70)
    print(f"{'Pair':<10} {'Tensor Sim':>12} {'Agreement':>12} {'Logit Cos':>12} {'Prob Cos':>12}")
    print("-"*70)
    for r in results:
        print(f"({r['seed_i']},{r['seed_j']})      {r['tensor_sim']:>12.4f} {r['agreement']:>12.4f} "
              f"{r['logit_cosine']:>12.4f} {r['prob_cosine']:>12.4f}")

    # Correlation analysis
    print("\n" + "="*70)
    print("CORRELATION OF TENSOR SIM WITH EACH FUNCTIONAL METRIC")
    print("="*70)

    tensor_sims = [r["tensor_sim"] for r in results]

    func_metrics = ["agreement", "logit_cosine", "prob_cosine", "logit_correlation", "kl_divergence"]
    correlations = {}

    for metric in func_metrics:
        values = [r[metric] for r in results]
        # For KL divergence, negate because lower = more similar
        if metric == "kl_divergence":
            values = [-v for v in values]

        pearson_r, pearson_p = stats.pearsonr(tensor_sims, values)
        spearman_r, spearman_p = stats.spearmanr(tensor_sims, values)

        correlations[metric] = {
            "pearson_r": pearson_r,
            "spearman_r": spearman_r,
        }

        print(f"{metric:20s}: Pearson r={pearson_r:+.4f}, Spearman ρ={spearman_r:+.4f}")

    # Save results (convert numpy/torch types to native Python)
    def to_native(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_native(v) for v in obj]
        return obj

    output = {
        "pairwise_results": to_native(results),
        "correlations": to_native(correlations),
    }
    with open(RESULTS_DIR / "functional_comparison_results.json", "w") as f:
        json.dump(output, f, indent=2)

    # Create visualization
    print("\nCreating plots...")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    metrics_to_plot = [
        ("agreement", "Agreement (argmax)", "blue"),
        ("logit_cosine", "Logit Cosine", "green"),
        ("prob_cosine", "Probability Cosine", "orange"),
        ("logit_correlation", "Logit Correlation", "purple"),
    ]

    for ax, (metric, label, color) in zip(axes, metrics_to_plot):
        values = [r[metric] for r in results]
        ax.scatter(tensor_sims, values, alpha=0.7, s=80, c=color)

        pr = correlations[metric]["pearson_r"]
        sr = correlations[metric]["spearman_r"]

        ax.set_xlabel("Tensor Similarity", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f"{label}\nPearson r={pr:.3f}, Spearman ρ={sr:.3f}", fontsize=11)
        ax.grid(True, alpha=0.3)

        # Add pair labels
        for r in results:
            ax.annotate(f"({r['seed_i']},{r['seed_j']})",
                       (r['tensor_sim'], r[metric]),
                       textcoords="offset points", xytext=(3, 3), fontsize=7)

    plt.suptitle("Tensor Similarity vs Different Functional Similarity Metrics",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "functional_comparison_scatter.png", dpi=150)
    plt.close()

    # Bar chart comparing correlations
    fig, ax = plt.subplots(figsize=(10, 5))

    metrics = list(correlations.keys())
    pearson_vals = [correlations[m]["pearson_r"] for m in metrics]
    spearman_vals = [correlations[m]["spearman_r"] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    # Clean up names
    display_names = {
        "agreement": "Agreement\n(argmax)",
        "logit_cosine": "Logit\nCosine",
        "prob_cosine": "Probability\nCosine",
        "logit_correlation": "Logit\nCorrelation",
        "kl_divergence": "KL Divergence\n(negated)",
    }
    labels = [display_names[m] for m in metrics]

    bars1 = ax.bar(x - width/2, pearson_vals, width, label='Pearson r', color='steelblue')
    bars2 = ax.bar(x + width/2, spearman_vals, width, label='Spearman ρ', color='darkorange')

    ax.set_ylabel('Correlation with Tensor Similarity')
    ax.set_title('Which Functional Similarity Metric Correlates Best with Tensor Sim?', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars1, pearson_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, spearman_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "functional_correlation_bar.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {RESULTS_DIR}/")

    return results, correlations


if __name__ == "__main__":
    run_comparison()
