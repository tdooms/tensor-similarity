"""
Create a clean, helpful summary chart of the multi-metric comparison.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open("results_multi_metric/multi_metric_results.json", "r") as f:
    data = json.load(f)

correlations = data["correlations"]

# Extract and sort by Spearman (more robust with small samples)
metrics = list(correlations.keys())
pearson = [correlations[m]["pearson_r"] for m in metrics]
spearman = [correlations[m]["spearman_r"] for m in metrics]

# Sort by Spearman correlation
sorted_idx = np.argsort(spearman)[::-1]
metrics = [metrics[i] for i in sorted_idx]
pearson = [pearson[i] for i in sorted_idx]
spearman = [spearman[i] for i in sorted_idx]

# Clean up metric names for display
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
labels = [display_names.get(m, m) for m in metrics]

# Create figure
fig, ax = plt.subplots(figsize=(10, 7))

y_pos = np.arange(len(metrics))
bar_height = 0.35

# Plot bars
bars1 = ax.barh(y_pos + bar_height/2, spearman, bar_height,
                label='Spearman ρ', color='#2ecc71', alpha=0.85)
bars2 = ax.barh(y_pos - bar_height/2, pearson, bar_height,
                label='Pearson r', color='#3498db', alpha=0.85)

# Add value labels
for i, (s, p) in enumerate(zip(spearman, pearson)):
    # Spearman label
    x_pos = s + 0.02 if s >= 0 else s - 0.02
    ha = 'left' if s >= 0 else 'right'
    ax.text(x_pos, i + bar_height/2, f'{s:.2f}', va='center', ha=ha, fontsize=9, fontweight='bold')

    # Pearson label
    x_pos = p + 0.02 if p >= 0 else p - 0.02
    ha = 'left' if p >= 0 else 'right'
    ax.text(x_pos, i - bar_height/2, f'{p:.2f}', va='center', ha=ha, fontsize=9)

# Styling
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel('Correlation with Output Agreement', fontsize=12)
ax.set_title('Which Metrics Best Predict Functional Similarity?\n(5 seeds, 10 pairs, sorted by Spearman)',
             fontsize=14, fontweight='bold')
ax.axvline(x=0, color='gray', linewidth=1, linestyle='-')
ax.set_xlim(-0.5, 0.6)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3, axis='x')

# Add annotation
ax.annotate('← Less similar outputs', xy=(-0.45, -0.8), fontsize=9, color='gray')
ax.annotate('More similar outputs →', xy=(0.25, -0.8), fontsize=9, color='gray')

# Highlight top performers
ax.axhspan(len(metrics)-1.5, len(metrics)-0.5, alpha=0.15, color='green')
ax.axhspan(len(metrics)-2.5, len(metrics)-1.5, alpha=0.1, color='green')
ax.axhspan(len(metrics)-3.5, len(metrics)-2.5, alpha=0.05, color='green')

plt.tight_layout()
plt.savefig("results_multi_metric/metric_comparison_summary.png", dpi=150, bbox_inches='tight')
plt.close()

print("Saved: results_multi_metric/metric_comparison_summary.png")

# Also create a simple table view
print("\n" + "="*60)
print("METRIC RANKING (sorted by Spearman correlation)")
print("="*60)
print(f"{'Rank':<5} {'Metric':<25} {'Spearman':>10} {'Pearson':>10}")
print("-"*60)
for i, (m, s, p) in enumerate(zip(labels, spearman, pearson)):
    marker = "★" if i < 3 else " "
    print(f"{i+1:<5} {m:<25} {s:>+10.3f} {p:>+10.3f} {marker}")
