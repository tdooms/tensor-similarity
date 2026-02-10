"""
Experiment: Tensor Sim vs Functional Sim Across Distributions

Measures how well tensor similarity (weight-based) tracks functional similarity
(output-based) across non-Gaussian input distributions. Tensor similarity is
theoretically equivalent to functional similarity for Gaussian inputs; this
experiment measures how that relationship breaks down for other distributions.
"""

import os
import json
import importlib.util
import itertools
import torch
import torch.nn as nn
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Import hyphenated module ---
spec = importlib.util.spec_from_file_location("bilinear", "bilinear-2nd-argmax.py")
bilinear = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bilinear)

from metrics import tensor_similarity, functional_similarity, estimate_covariance

# --- Config ---
NUM_SEEDS = 5
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
}

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


# --- Phase 1: Train all models with checkpoints ---

def train_with_checkpoints(dist_fn, seed):
    """Train a model, returning state dicts and accuracies at each checkpoint step."""
    torch.manual_seed(seed)
    model = bilinear.BilinearStack(N, num_layers=3, rank=32, use_linear=True)
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
    """Create a fresh model and load weights."""
    model = bilinear.BilinearStack(N, num_layers=3, rank=32, use_linear=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


print("=" * 60)
print("Phase 1: Training models")
print(f"  {len(DISTRIBUTIONS)} distributions × {NUM_SEEDS} seeds × {len(CHECKPOINT_STEPS)} checkpoints")
print("=" * 60)

# models[(dist_name, seed, step)] = model
# accuracies[(dist_name, seed)] = {step: acc}
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


# --- Phase 2: Estimate covariance matrices ---

print("\n" + "=" * 60)
print("Phase 2: Estimating covariance matrices")
print("=" * 60)

covariances = {}
for dist_name, dist_fn in DISTRIBUTIONS.items():
    cov = estimate_covariance(dist_fn, n=N)
    covariances[dist_name] = cov
    print(f"  {dist_name}: trace={cov.trace().item():.3f}")


# --- Phase 3: Compute pairwise metrics ---

print("\n" + "=" * 60)
print("Phase 3: Computing pairwise metrics")
print("=" * 60)

within_dist_results = []
cross_dist_results = []

# Within-distribution: all pairs within same distribution
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
        fs_train = functional_similarity(m1, m2, dist_fn, n=N, num_samples=FUNC_SIM_SAMPLES)
        fs_gaussian = functional_similarity(m1, m2, bilinear.gaussian, n=N, num_samples=FUNC_SIM_SAMPLES)

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

# Cross-distribution: final checkpoint only, across different distributions
print(f"\n  Cross-distribution (step {TRAIN_STEPS} only)...", flush=True)
cross_keys = [(dist_name, seed, TRAIN_STEPS)
              for dist_name in DISTRIBUTIONS
              for seed in range(NUM_SEEDS)]
cross_pairs = [(a, b) for a, b in itertools.combinations(cross_keys, 2)
               if a[0] != b[0]]
print(f"  {len(cross_pairs)} pairs")

for i, ((d1, s1, st1), (d2, s2, st2)) in enumerate(cross_pairs):
    if (i + 1) % 200 == 0:
        print(f"    {i+1}/{len(cross_pairs)}", flush=True)

    m1 = models[(d1, s1, st1)]
    m2 = models[(d2, s2, st2)]

    ts_standard = tensor_similarity(m1, m2)
    fs_gaussian = functional_similarity(m1, m2, bilinear.gaussian, n=N, num_samples=FUNC_SIM_SAMPLES)

    cross_dist_results.append({
        'dist1': d1, 'dist2': d2,
        'seed1': s1, 'seed2': s2,
        'tensor_sim_standard': ts_standard,
        'func_sim_gaussian': fs_gaussian,
    })


# --- Phase 4: Save results ---

print("\n" + "=" * 60)
print("Phase 4: Saving results")
print("=" * 60)

acc_serializable = {}
for (dist_name, seed), accs in accuracies.items():
    key = f"{dist_name}_seed{seed}"
    acc_serializable[key] = {str(step): acc for step, acc in accs.items()}

results = {
    'config': {
        'num_seeds': NUM_SEEDS,
        'checkpoint_steps': CHECKPOINT_STEPS,
        'distributions': list(DISTRIBUTIONS.keys()),
    },
    'accuracies': acc_serializable,
    'within_dist': within_dist_results,
    'cross_dist': cross_dist_results,
}

results_path = os.path.join(RESULTS_DIR, 'results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Saved {results_path}")


# --- Phase 5: Generate charts ---

print("\n" + "=" * 60)
print("Phase 5: Generating charts")
print("=" * 60)

# Chart 1: Training curves
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
for idx, dist_name in enumerate(DISTRIBUTIONS):
    ax = axes[idx]
    seed_accs = []
    for seed in range(NUM_SEEDS):
        accs = accuracies[(dist_name, seed)]
        steps_sorted = sorted(accs.keys())
        vals = [accs[s] for s in steps_sorted]
        seed_accs.append(vals)
        ax.plot(steps_sorted, vals, alpha=0.3, color='C0')
    mean_accs = np.mean(seed_accs, axis=0)
    std_accs = np.std(seed_accs, axis=0)
    ax.plot(steps_sorted, mean_accs, color='C0', linewidth=2)
    ax.fill_between(steps_sorted, mean_accs - std_accs, mean_accs + std_accs,
                    alpha=0.2, color='C0')
    ax.set_title(dist_name)
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    ax.set_xscale('symlog', linthresh=1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

plt.suptitle('Training Curves by Distribution', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'training_curves.png'), dpi=150)
plt.close()
print("  Saved training_curves.png")

# Chart 2: Scatter grid — tensor sim vs functional sim per distribution
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
for idx, dist_name in enumerate(DISTRIBUTIONS):
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

plt.suptitle('Tensor Similarity vs Functional Similarity (within-distribution)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'scatter_grid.png'), dpi=150)
plt.close()
print("  Saved scatter_grid.png")

# Chart 3: Standard vs Generalized bar chart
fig, ax = plt.subplots(figsize=(12, 6))
dist_names = list(DISTRIBUTIONS.keys())
r_standard = []
r_generalized = []
for dist_name in dist_names:
    rows = [r for r in within_dist_results if r['dist'] == dist_name]
    ts_std = [r['tensor_sim_standard'] for r in rows]
    ts_gen = [r['tensor_sim_generalized'] for r in rows]
    fs = [r['func_sim_train'] for r in rows]
    r_s, _ = stats.pearsonr(ts_std, fs) if len(ts_std) > 2 else (0, 1)
    r_g, _ = stats.pearsonr(ts_gen, fs) if len(ts_gen) > 2 else (0, 1)
    r_standard.append(r_s)
    r_generalized.append(r_g)

x = np.arange(len(dist_names))
width = 0.35
ax.bar(x - width/2, r_standard, width, label='Standard (Σ=I)')
ax.bar(x + width/2, r_generalized, width, label='Generalized (Σ=estimated)')
ax.set_xticks(x)
ax.set_xticklabels(dist_names, rotation=45, ha='right')
ax.set_ylabel('Pearson r (tensor sim vs func sim)')
ax.set_title('Standard vs Generalized Tensor Similarity Correlation')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'standard_vs_generalized.png'), dpi=150)
plt.close()
print("  Saved standard_vs_generalized.png")

# Chart 4: Cross-distribution scatter
fig, ax = plt.subplots(figsize=(10, 8))
pair_labels = {}
for r in cross_dist_results:
    pair = tuple(sorted([r['dist1'], r['dist2']]))
    if pair not in pair_labels:
        pair_labels[pair] = len(pair_labels)

colors = plt.cm.tab20(np.linspace(0, 1, max(len(pair_labels), 1)))
for r in cross_dist_results:
    pair = tuple(sorted([r['dist1'], r['dist2']]))
    c = colors[pair_labels[pair] % len(colors)]
    ax.scatter(r['tensor_sim_standard'], r['func_sim_gaussian'],
               color=c, alpha=0.4, s=12)

ts_all = [r['tensor_sim_standard'] for r in cross_dist_results]
fs_all = [r['func_sim_gaussian'] for r in cross_dist_results]
r_cross, _ = stats.pearsonr(ts_all, fs_all) if len(ts_all) > 2 else (0, 1)

ax.set_xlabel('Tensor Sim (standard)')
ax.set_ylabel('Func Sim (Gaussian)')
ax.set_title(f'Cross-Distribution: Tensor Sim vs Functional Sim (r={r_cross:.3f})')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'cross_distribution.png'), dpi=150)
plt.close()
print("  Saved cross_distribution.png")

print("\n" + "=" * 60)
print("Done! All results saved to", RESULTS_DIR)
print("=" * 60)
