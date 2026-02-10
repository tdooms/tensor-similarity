"""Generate visualizations for tensor similarity experiment results."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    # Load results
    results_path = Path("results/evaluation_results.json")
    if not results_path.exists():
        print(f"Error: {results_path} not found. Run evaluate.py first.")
        return

    with open(results_path) as f:
        data = json.load(f)

    results = data["pairwise_results"]
    corrs = data["correlations"]

    # Extract data
    ts_std = [r["tensor_sim_standard"] for r in results]
    ts_gen = [r["tensor_sim_generalized"] for r in results]
    agreement = [r["output_agreement"] for r in results]
    logit_cos = [r["logit_cosine"] for r in results]

    results_dir = Path("results")

    # Figure 1: Scatter comparison - Tensor Sim vs Output Agreement
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Standard tensor sim
    ax = axes[0]
    ax.scatter(ts_std, agreement, alpha=0.3, s=10, c='steelblue')
    pr = corrs["standard_vs_agreement"]["pearson_r"]
    sr = corrs["standard_vs_agreement"]["spearman_r"]
    ax.set_xlabel("Tensor Similarity (Standard, M=I)", fontsize=11)
    ax.set_ylabel("Output Agreement", fontsize=11)
    ax.set_title(f"Standard: r={pr:.4f} (Spearman: {sr:.4f})", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(ts_std, agreement, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(ts_std), max(ts_std), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.7, linewidth=2)

    # Generalized tensor sim
    ax = axes[1]
    ax.scatter(ts_gen, agreement, alpha=0.3, s=10, c='darkorange')
    pr = corrs["generalized_vs_agreement"]["pearson_r"]
    sr = corrs["generalized_vs_agreement"]["spearman_r"]
    ax.set_xlabel("Tensor Similarity (Generalized, M=Σ)", fontsize=11)
    ax.set_ylabel("Output Agreement", fontsize=11)
    ax.set_title(f"Generalized: r={pr:.4f} (Spearman: {sr:.4f})", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(ts_gen, agreement, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(ts_gen), max(ts_gen), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.7, linewidth=2)

    plt.suptitle("Tensor Similarity vs Output Agreement\n(Modular Addition mod 113)", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "scatter_agreement.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {results_dir / 'scatter_agreement.png'}")

    # Figure 2: Scatter comparison - Tensor Sim vs Logit Cosine
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Standard tensor sim
    ax = axes[0]
    ax.scatter(ts_std, logit_cos, alpha=0.3, s=10, c='steelblue')
    pr = corrs["standard_vs_logit_cosine"]["pearson_r"]
    sr = corrs["standard_vs_logit_cosine"]["spearman_r"]
    ax.set_xlabel("Tensor Similarity (Standard, M=I)", fontsize=11)
    ax.set_ylabel("Logit Cosine Similarity", fontsize=11)
    ax.set_title(f"Standard: r={pr:.4f} (Spearman: {sr:.4f})", fontsize=12)
    ax.grid(True, alpha=0.3)

    z = np.polyfit(ts_std, logit_cos, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(ts_std), max(ts_std), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.7, linewidth=2)

    # Generalized tensor sim
    ax = axes[1]
    ax.scatter(ts_gen, logit_cos, alpha=0.3, s=10, c='darkorange')
    pr = corrs["generalized_vs_logit_cosine"]["pearson_r"]
    sr = corrs["generalized_vs_logit_cosine"]["spearman_r"]
    ax.set_xlabel("Tensor Similarity (Generalized, M=Σ)", fontsize=11)
    ax.set_ylabel("Logit Cosine Similarity", fontsize=11)
    ax.set_title(f"Generalized: r={pr:.4f} (Spearman: {sr:.4f})", fontsize=12)
    ax.grid(True, alpha=0.3)

    z = np.polyfit(ts_gen, logit_cos, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(ts_gen), max(ts_gen), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.7, linewidth=2)

    plt.suptitle("Tensor Similarity vs Logit Cosine Similarity\n(Modular Addition mod 113)", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "scatter_logit_cosine.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {results_dir / 'scatter_logit_cosine.png'}")

    # Figure 3: Bar chart comparing correlations
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = [
        ("standard_vs_agreement", "Standard"),
        ("generalized_vs_agreement", "Generalized"),
        ("standard_vs_logit_cosine", "Standard"),
        ("generalized_vs_logit_cosine", "Generalized"),
    ]
    labels = [
        "Agreement\n(Standard)",
        "Agreement\n(Generalized)",
        "Logit Cosine\n(Standard)",
        "Logit Cosine\n(Generalized)",
    ]

    pearson_vals = [corrs[m[0]]["pearson_r"] for m in metrics]
    spearman_vals = [corrs[m[0]]["spearman_r"] for m in metrics]

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, pearson_vals, width, label='Pearson r', color='steelblue')
    bars2 = ax.bar(x + width/2, spearman_vals, width, label='Spearman r', color='darkorange')

    ax.set_ylabel("Correlation Coefficient", fontsize=12)
    ax.set_title("Standard vs Generalized Tensor Similarity:\nCorrelation with Functional Metrics", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9)

    plt.tight_layout()
    plt.savefig(results_dir / "correlation_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {results_dir / 'correlation_comparison.png'}")

    # Figure 4: Head-to-head comparison
    fig, ax = plt.subplots(figsize=(8, 6))

    func_metrics = ["agreement", "logit_cosine"]
    func_labels = ["Output Agreement", "Logit Cosine"]

    x = np.arange(len(func_metrics))
    width = 0.35

    std_vals = [corrs[f"standard_vs_{m}"]["pearson_r"] for m in func_metrics]
    gen_vals = [corrs[f"generalized_vs_{m}"]["pearson_r"] for m in func_metrics]

    bars1 = ax.bar(x - width/2, std_vals, width, label='Standard (M=I)', color='steelblue')
    bars2 = ax.bar(x + width/2, gen_vals, width, label='Generalized (M=Σ)', color='darkorange')

    ax.set_ylabel("Pearson Correlation", fontsize=12)
    ax.set_title("Standard vs Generalized Tensor Similarity:\nWhich Correlates Better with Functional Similarity?", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(func_labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold')

    # Add difference annotation
    for i, (s, g) in enumerate(zip(std_vals, gen_vals)):
        diff = g - s
        color = 'green' if diff > 0 else 'red'
        sign = '+' if diff > 0 else ''
        ax.annotate(f'Δ={sign}{diff:.4f}',
                    xy=(x[i], max(s, g) + 0.02),
                    ha='center', fontsize=10, color=color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(results_dir / "head_to_head.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {results_dir / 'head_to_head.png'}")

    # Figure 5: Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Tensor sim distributions
    ax = axes[0, 0]
    ax.hist(ts_std, bins=50, alpha=0.7, label='Standard (M=I)', color='steelblue')
    ax.axvline(np.mean(ts_std), color='steelblue', linestyle='--', linewidth=2)
    ax.set_xlabel("Tensor Similarity (Standard)")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution: mean={np.mean(ts_std):.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.hist(ts_gen, bins=50, alpha=0.7, label='Generalized (M=Σ)', color='darkorange')
    ax.axvline(np.mean(ts_gen), color='darkorange', linestyle='--', linewidth=2)
    ax.set_xlabel("Tensor Similarity (Generalized)")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution: mean={np.mean(ts_gen):.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Functional similarity distributions
    ax = axes[1, 0]
    ax.hist(agreement, bins=50, alpha=0.7, color='green')
    ax.axvline(np.mean(agreement), color='darkgreen', linestyle='--', linewidth=2)
    ax.set_xlabel("Output Agreement")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution: mean={np.mean(agreement):.4f}")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.hist(logit_cos, bins=50, alpha=0.7, color='purple')
    ax.axvline(np.mean(logit_cos), color='darkviolet', linestyle='--', linewidth=2)
    ax.set_xlabel("Logit Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution: mean={np.mean(logit_cos):.4f}")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Distribution of Similarity Metrics\n(100 models, 4950 pairs)", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {results_dir / 'distributions.png'}")

    print("\nAll visualizations complete!")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    std_vs_agr = corrs["standard_vs_agreement"]["pearson_r"]
    gen_vs_agr = corrs["generalized_vs_agreement"]["pearson_r"]
    diff = gen_vs_agr - std_vs_agr
    better = "GENERALIZED" if diff > 0 else "STANDARD"
    print(f"Output Agreement: Standard r={std_vs_agr:.4f}, Generalized r={gen_vs_agr:.4f}")
    print(f"  --> {better} is better by {abs(diff):.4f}")

    std_vs_cos = corrs["standard_vs_logit_cosine"]["pearson_r"]
    gen_vs_cos = corrs["generalized_vs_logit_cosine"]["pearson_r"]
    diff = gen_vs_cos - std_vs_cos
    better = "GENERALIZED" if diff > 0 else "STANDARD"
    print(f"Logit Cosine: Standard r={std_vs_cos:.4f}, Generalized r={gen_vs_cos:.4f}")
    print(f"  --> {better} is better by {abs(diff):.4f}")


if __name__ == "__main__":
    main()
