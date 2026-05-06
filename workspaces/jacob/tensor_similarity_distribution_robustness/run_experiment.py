"""
Experiment: neg10_last_gaussian — No Residual, 1 Layer

Tests a distribution where the first 3 dimensions are N(0,1) and the last
dimension is always -10. Uses PureBilinear (no skip connection, 1 layer)
where tensor similarity theory is exact for Gaussian inputs.
"""

import os
import json
import importlib.util
import itertools
import torch
import torch.nn as nn
import numpy as np
from scipy import stats
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Import hyphenated module ---
spec = importlib.util.spec_from_file_location("bilinear", "bilinear-2nd-argmax.py")
bilinear = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bilinear)

from metrics import tensor_similarity, estimate_covariance


# --- Pure bilinear model (no residual) ---

class PureBilinear(nn.Module):
    """Single-layer bilinear WITHOUT residual connections."""
    def __init__(self, n, num_layers=1, rank=32):
        super().__init__()
        self.n = n
        self.num_layers = num_layers
        self.Ls = nn.ParameterList()
        self.Rs = nn.ParameterList()
        self.Ds = nn.ParameterList()
        for _ in range(num_layers):
            self.Ls.append(nn.Parameter(torch.randn(rank, n) * 0.1))
            self.Rs.append(nn.Parameter(torch.randn(rank, n) * 0.1))
            self.Ds.append(nn.Parameter(torch.randn(n, rank) * 0.1))

    def forward(self, x):
        h = x
        for i in range(self.num_layers):
            Lh = h @ self.Ls[i].T
            Rh = h @ self.Rs[i].T
            h = (Lh * Rh) @ self.Ds[i].T  # NO residual
        return h


# --- Config ---
NUM_SEEDS = 5
NUM_LAYERS = 1
RANK = 32
CHECKPOINT_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
TRAIN_STEPS = CHECKPOINT_STEPS[-1]
N = bilinear.N
BATCH_SIZE = bilinear.BATCH_SIZE
FUNC_SIM_SAMPLES = 50000

DISTRIBUTIONS = {
    'gaussian': bilinear.gaussian,
    'half_gaussian': bilinear.half_gaussian,
    'bimodal': bilinear.bimodal,
    'uniform': bilinear.uniform,
    'laplace': bilinear.laplace,
    'sparse_spikes': bilinear.sparse_spikes,
    'permutation': bilinear.permutation,
    'correlated_gaussian': bilinear.correlated_gaussian,
    'neg10_last_gaussian': bilinear.neg10_last_gaussian,
}

RESULTS_DIR = 'results_neg10_no_residual'
os.makedirs(RESULTS_DIR, exist_ok=True)


# --- Continuous functional similarity ---

def continuous_func_sim(m1, m2, dist_fn, n=4, num_samples=50000):
    m1.eval(); m2.eval()
    with torch.no_grad():
        x = dist_fn(num_samples, n)
        y1 = m1(x).reshape(-1)
        y2 = m2(x).reshape(-1)
        cos = torch.dot(y1, y2) / (y1.norm() * y2.norm() + 1e-10)
    return cos.item()


# --- Phase 1: Train ---

def train_with_checkpoints(dist_fn, seed):
    torch.manual_seed(seed)
    model = PureBilinear(N, num_layers=NUM_LAYERS, rank=RANK)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    checkpoints = {}

    def save_checkpoint(step):
        model.eval()
        with torch.no_grad():
            x_eval = dist_fn(FUNC_SIM_SAMPLES, N)
            targets = bilinear.task_2nd_argmax(x_eval)
            acc = (model(x_eval).argmax(-1) == targets).float().mean().item()
        checkpoints[step] = {
            'state_dict': {k: v.clone() for k, v in model.state_dict().items()},
            'accuracy': acc,
        }
        model.train()

    if 0 in CHECKPOINT_STEPS:
        save_checkpoint(0)
    for step in range(1, TRAIN_STEPS + 1):
        x = dist_fn(BATCH_SIZE, N)
        targets = bilinear.task_2nd_argmax(x)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in CHECKPOINT_STEPS:
            save_checkpoint(step)
    return checkpoints


def load_model(state_dict):
    model = PureBilinear(N, num_layers=NUM_LAYERS, rank=RANK)
    model.load_state_dict(state_dict)
    model.eval()
    return model


print("=" * 60)
print("Phase 1: Training models (NO RESIDUAL, 1 layer)")
print(f"  {len(DISTRIBUTIONS)} distributions x {NUM_SEEDS} seeds x {len(CHECKPOINT_STEPS)} checkpoints")
print("=" * 60)

models = {}
accuracies = {}

for dist_name, dist_fn in DISTRIBUTIONS.items():
    for seed in range(NUM_SEEDS):
        print(f"  {dist_name} seed={seed}...", end=" ", flush=True)
        checkpoints = train_with_checkpoints(dist_fn, seed)
        accs = {}
        for step, ckpt in checkpoints.items():
            models[(dist_name, seed, step)] = load_model(ckpt['state_dict'])
            accs[step] = ckpt['accuracy']
        accuracies[(dist_name, seed)] = accs
        print(f"final acc={accs[TRAIN_STEPS]:.1%}")


# --- Phase 2: Estimate covariances ---

print("\n" + "=" * 60)
print("Phase 2: Estimating covariance matrices")
print("=" * 60)

covariances = {}
for dist_name, dist_fn in DISTRIBUTIONS.items():
    cov = estimate_covariance(dist_fn, n=N)
    covariances[dist_name] = cov
    print(f"  {dist_name}: trace={cov.trace().item():.3f}, rank={torch.linalg.matrix_rank(cov).item()}")


# --- Phase 3: Within-distribution pairwise metrics ---

print("\n" + "=" * 60)
print("Phase 3: Computing within-distribution pairwise metrics")
print("=" * 60)

within_dist_results = []

for dist_name, dist_fn in DISTRIBUTIONS.items():
    keys = [(dist_name, seed, step)
            for seed in range(NUM_SEEDS)
            for step in CHECKPOINT_STEPS]
    pairs = list(itertools.combinations(keys, 2))
    print(f"  {dist_name}: {len(pairs)} pairs...", flush=True)

    Sigma = covariances[dist_name]

    for i, ((d1, s1, st1), (d2, s2, st2)) in enumerate(pairs):
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(pairs)}", flush=True)

        m1 = models[(d1, s1, st1)]
        m2 = models[(d2, s2, st2)]

        ts_standard = tensor_similarity(m1, m2)
        ts_generalized = tensor_similarity(m1, m2, Sigma=Sigma)
        fs_train = continuous_func_sim(m1, m2, dist_fn, n=N, num_samples=FUNC_SIM_SAMPLES)
        fs_gaussian = continuous_func_sim(m1, m2, bilinear.gaussian, n=N, num_samples=FUNC_SIM_SAMPLES)

        within_dist_results.append({
            'dist': dist_name,
            'seed1': s1, 'step1': st1,
            'seed2': s2, 'step2': st2,
            'tensor_sim_standard': ts_standard,
            'tensor_sim_generalized': ts_generalized,
            'func_sim_train': fs_train,
            'func_sim_gaussian': fs_gaussian,
            'same_step': st1 == st2,
            'same_seed': s1 == s2,
        })


# --- Phase 4: Cross-distribution (final checkpoint) ---

print("\n" + "=" * 60)
print("Phase 4: Cross-distribution comparison (final checkpoint)")
print("=" * 60)

cross_dist_results = []
dist_names = list(DISTRIBUTIONS.keys())
final_step = CHECKPOINT_STEPS[-1]

for d1, d2 in itertools.combinations(dist_names, 2):
    print(f"  {d1} vs {d2}...", flush=True)
    for s1 in range(NUM_SEEDS):
        for s2 in range(NUM_SEEDS):
            m1 = models[(d1, s1, final_step)]
            m2 = models[(d2, s2, final_step)]

            ts_standard = tensor_similarity(m1, m2)
            fs_d1 = continuous_func_sim(m1, m2, DISTRIBUTIONS[d1], n=N, num_samples=FUNC_SIM_SAMPLES)
            fs_d2 = continuous_func_sim(m1, m2, DISTRIBUTIONS[d2], n=N, num_samples=FUNC_SIM_SAMPLES)
            fs_gauss = continuous_func_sim(m1, m2, bilinear.gaussian, n=N, num_samples=FUNC_SIM_SAMPLES)

            cross_dist_results.append({
                'dist1': d1, 'dist2': d2,
                'seed1': s1, 'seed2': s2,
                'tensor_sim_standard': ts_standard,
                'func_sim_d1': fs_d1,
                'func_sim_d2': fs_d2,
                'func_sim_gaussian': fs_gauss,
            })


# --- Phase 5: Save results ---

print("\n" + "=" * 60)
print("Phase 5: Saving results")
print("=" * 60)

acc_serializable = {}
for (dist_name, seed), accs in accuracies.items():
    key = f"{dist_name}_seed{seed}"
    acc_serializable[key] = {str(step): acc for step, acc in accs.items()}

results = {
    'config': {
        'num_seeds': NUM_SEEDS,
        'num_layers': NUM_LAYERS,
        'rank': RANK,
        'residual': False,
        'func_sim_metric': 'continuous_cosine',
        'checkpoint_steps': CHECKPOINT_STEPS,
        'distributions': dist_names,
    },
    'accuracies': acc_serializable,
    'within_dist': within_dist_results,
    'cross_dist': cross_dist_results,
}

results_path = os.path.join(RESULTS_DIR, 'results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Saved {results_path}")


# --- Phase 6: Generate charts ---

print("\n" + "=" * 60)
print("Phase 6: Generating charts")
print("=" * 60)

# Chart 1: Scatter grid — standard tensor sim vs func sim (train dist)
n_dists = len(dist_names)
n_cols = 3
n_rows = math.ceil(n_dists / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
axes = axes.flatten()
for idx, dist_name in enumerate(dist_names):
    ax = axes[idx]
    rows = [r for r in within_dist_results if r['dist'] == dist_name]
    ts = [r['tensor_sim_standard'] for r in rows]
    fs = [r['func_sim_train'] for r in rows]
    same_step = [r['same_step'] for r in rows]

    cross_ts = [t for t, s in zip(ts, same_step) if not s]
    cross_fs = [f for f, s in zip(fs, same_step) if not s]
    same_ts = [t for t, s in zip(ts, same_step) if s]
    same_fs = [f for f, s in zip(fs, same_step) if s]

    ax.scatter(cross_ts, cross_fs, alpha=0.3, s=8, color='C0', label='cross-step')
    ax.scatter(same_ts, same_fs, alpha=0.3, s=8, color='C1', label='same-step')

    r_val, _ = stats.pearsonr(ts, fs) if len(ts) > 2 else (0, 1)
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
print("  Saved scatter_grid.png")

# Chart 2: Standard vs Generalized bar chart
fig, ax = plt.subplots(figsize=(14, 7))
r_standard = []
r_generalized = []
for dist_name in dist_names:
    rows = [r for r in within_dist_results if r['dist'] == dist_name]
    ts_std = [r['tensor_sim_standard'] for r in rows]
    ts_gen = [r['tensor_sim_generalized'] for r in rows]
    fs = [r['func_sim_train'] for r in rows]
    r_s, _ = stats.pearsonr(ts_std, fs)
    r_g, _ = stats.pearsonr(ts_gen, fs)
    r_standard.append(r_s)
    r_generalized.append(r_g)

x = np.arange(len(dist_names))
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
print("  Saved standard_vs_generalized.png")

# Chart 3: Focus comparison — gaussian vs neg10_last_gaussian
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
print("  Saved gaussian_vs_neg10.png")

# Chart 4: Training curves
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
axes = axes.flatten()
for idx, dist_name in enumerate(dist_names):
    ax = axes[idx]
    for seed in range(NUM_SEEDS):
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
print("  Saved training_curves.png")

# Chart 5: Cross-distribution heatmap
n_d = len(dist_names)
cross_matrix = np.zeros((n_d, n_d))
for i in range(n_d):
    cross_matrix[i, i] = 1.0
    for j in range(n_d):
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
ax.set_xticks(range(n_d))
ax.set_yticks(range(n_d))
ax.set_xticklabels(dist_names, rotation=45, ha='right')
ax.set_yticklabels(dist_names)
for i in range(n_d):
    for j in range(n_d):
        ax.text(j, i, f'{cross_matrix[i, j]:.2f}', ha='center', va='center', fontsize=8)
plt.colorbar(im, ax=ax, label='Tensor Sim (standard)')
ax.set_title('Cross-Distribution Tensor Similarity (no residual, final checkpoint)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'cross_distribution_heatmap.png'), dpi=150)
plt.close()
print("  Saved cross_distribution_heatmap.png")


# Print summary
print("\n" + "=" * 60)
print("Summary of Pearson r values (within-distribution)")
print("=" * 60)
print(f"{'Distribution':<22} {'Standard':>10} {'Generalized':>12}")
print("-" * 46)
for i, dist_name in enumerate(dist_names):
    print(f"{dist_name:<22} {r_standard[i]:>10.3f} {r_generalized[i]:>12.3f}")

print("\n" + "=" * 60)
print("Final accuracies")
print("=" * 60)
for dist_name in dist_names:
    accs = [accuracies[(dist_name, s)][TRAIN_STEPS] for s in range(NUM_SEEDS)]
    print(f"  {dist_name:<22} mean={np.mean(accs):.1%}  std={np.std(accs):.1%}")

print("\n" + "=" * 60)
print("Done! All results saved to", RESULTS_DIR)
print("=" * 60)
