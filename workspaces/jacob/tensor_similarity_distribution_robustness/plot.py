"""
Regenerate all figures from saved results.json — no model training required.
Run from the experiment directory: python plot.py
"""

import json
import math
import os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = 'results_neg10_no_residual'

with open(os.path.join(RESULTS_DIR, 'results.json')) as f:
    data = json.load(f)

dist_names = data['config']['distributions']
checkpoint_steps = data['config']['checkpoint_steps']
num_seeds = data['config']['num_seeds']
within_dist_results = data['within_dist']
cross_dist_results = data['cross_dist']

# Reconstruct accuracies dict keyed by (dist_name, seed) -> {step: acc}
accuracies = {}
for key, step_accs in data['accuracies'].items():
    dist_name, seed_str = key.rsplit('_seed', 1)
    seed = int(seed_str)
    accuracies[(dist_name, seed)] = {int(k): v for k, v in step_accs.items()}

final_step = checkpoint_steps[-1]
n_dists = len(dist_names)
n_cols = 3
n_rows = math.ceil(n_dists / n_cols)

# --- Chart 1: Scatter grid ---
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
axes = axes.flatten()
for idx, dist_name in enumerate(dist_names):
    ax = axes[idx]
    rows = [r for r in within_dist_results if r['dist'] == dist_name]
    ts = [r['tensor_sim_standard'] for r in rows]
    fs = [r['func_sim_train'] for r in rows]
    same_step = [r['same_step'] for r in rows]

    ax.scatter([t for t, s in zip(ts, same_step) if not s],
               [f for f, s in zip(fs, same_step) if not s],
               alpha=0.3, s=8, color='C0', label='cross-step')
    ax.scatter([t for t, s in zip(ts, same_step) if s],
               [f for f, s in zip(fs, same_step) if s],
               alpha=0.3, s=8, color='C1', label='same-step')

    r_val, _ = stats.pearsonr(ts, fs)
    ax.set_title(f'{dist_name}\nr={r_val:.3f}')
    ax.set_xlabel('Tensor Sim (standard)')
    ax.set_ylabel('Func Sim (train dist)')
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=7)

for idx in range(n_dists, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Tensor Sim vs Functional Sim — No Residual, 1 Layer', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'scatter_grid.png'), dpi=150)
plt.close()
print("Saved scatter_grid.png")

# --- Chart 2: Standard vs Generalized bar chart ---
r_standard, r_generalized = [], []
for dist_name in dist_names:
    rows = [r for r in within_dist_results if r['dist'] == dist_name]
    fs = [r['func_sim_train'] for r in rows]
    r_s, _ = stats.pearsonr([r['tensor_sim_standard'] for r in rows], fs)
    r_g, _ = stats.pearsonr([r['tensor_sim_generalized'] for r in rows], fs)
    r_standard.append(r_s)
    r_generalized.append(r_g)

fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(n_dists)
width = 0.35
ax.bar(x - width/2, r_standard, width, label='Standard (Sigma=I)')
ax.bar(x + width/2, r_generalized, width, label='Generalized (Sigma=estimated)')
ax.set_xticks(x)
ax.set_xticklabels(dist_names, rotation=45, ha='right')
ax.set_ylabel('Pearson r (tensor sim vs func sim)')
ax.set_title('Standard vs Generalized Tensor Similarity — No Residual, 1 Layer')
ax.legend()
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'standard_vs_generalized.png'), dpi=150)
plt.close()
print("Saved standard_vs_generalized.png")

# --- Chart 3: Gaussian vs neg10_last_gaussian focus ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, dname, color in [(axes[0], 'gaussian', 'C0'), (axes[1], 'neg10_last_gaussian', 'C1')]:
    rows = [r for r in within_dist_results if r['dist'] == dname]
    ts = [r['tensor_sim_standard'] for r in rows]
    fs = [r['func_sim_train'] for r in rows]
    r_val, _ = stats.pearsonr(ts, fs)
    ax.scatter(ts, fs, alpha=0.3, s=8, color=color)
    ax.set_title(f'{dname}\nStandard r={r_val:.3f}')
    ax.set_xlabel('Tensor Sim (standard)')
    ax.set_ylabel('Func Sim (train dist)')
    ax.grid(True, alpha=0.3)

plt.suptitle('Gaussian vs neg10-Last-Gaussian (no residual, 1 layer)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'gaussian_vs_neg10.png'), dpi=150)
plt.close()
print("Saved gaussian_vs_neg10.png")

# --- Chart 4: Training curves ---
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
axes = axes.flatten()
for idx, dist_name in enumerate(dist_names):
    ax = axes[idx]
    for seed in range(num_seeds):
        accs = accuracies[(dist_name, seed)]
        steps = sorted(accs.keys())
        ax.plot(steps, [accs[s] for s in steps], alpha=0.5, marker='o', markersize=3)
    ax.set_title(dist_name)
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    ax.set_xscale('symlog', linthresh=1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

for idx in range(n_dists, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Training Curves (No Residual, 1 Layer)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'training_curves.png'), dpi=150)
plt.close()
print("Saved training_curves.png")

# --- Chart 5: Cross-distribution heatmap ---
cross_matrix = np.zeros((n_dists, n_dists))
for i in range(n_dists):
    cross_matrix[i, i] = 1.0
    for j in range(n_dists):
        if i < j:
            rows = [r for r in cross_dist_results
                    if (r['dist1'] == dist_names[i] and r['dist2'] == dist_names[j]) or
                       (r['dist1'] == dist_names[j] and r['dist2'] == dist_names[i])]
            if rows:
                avg = np.mean([r['tensor_sim_standard'] for r in rows])
                cross_matrix[i, j] = avg
                cross_matrix[j, i] = avg

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cross_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
ax.set_xticks(range(n_dists))
ax.set_yticks(range(n_dists))
ax.set_xticklabels(dist_names, rotation=45, ha='right')
ax.set_yticklabels(dist_names)
for i in range(n_dists):
    for j in range(n_dists):
        ax.text(j, i, f'{cross_matrix[i, j]:.2f}', ha='center', va='center', fontsize=8)
plt.colorbar(im, ax=ax, label='Tensor Sim (standard)')
ax.set_title('Cross-Distribution Tensor Similarity (no residual, final checkpoint)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'cross_distribution_heatmap.png'), dpi=150)
plt.close()
print("Saved cross_distribution_heatmap.png")
