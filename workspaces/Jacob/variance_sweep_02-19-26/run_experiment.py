"""
Experiment: Variance Sweep on Last Dimension

The previous experiment showed:
  - gaussian:              r=0.997 (theory works perfectly)
  - neg10_last_gaussian:   r=0.605 (theory breaks down)

The last dim of neg10_last_gaussian is a constant (-10), giving it zero
variance and making the input covariance rank-deficient.

This experiment sweeps epsilon (the std of the last dim) from 0 to 1,
holding mean=-10. This traces the recovery curve and answers:
  Is the breakdown due to zero variance, the -10 offset, or both?

Distributions: last_dim ~ N(-10, eps^2) for eps in {0, 0.1, 0.5, 1.0}
  eps=0   -> constant -10 (the broken case)
  eps=1.0 -> nearly Gaussian offset (should recover)

Fast settings: 3 seeds, 4 checkpoints, 10k func-sim samples.
"""

import os
import sys
import json
import math
import itertools

import torch
import torch.nn as nn
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import metrics from sibling folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '2nd_argmax_02-10-26'))
from metrics import tensor_similarity, estimate_covariance

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# --- Model ---

class PureBilinear(nn.Module):
    """Single-layer bilinear WITHOUT residual connections."""
    def __init__(self, n, rank=32):
        super().__init__()
        self.n = n
        self.num_layers = 1
        self.Ls = nn.ParameterList([nn.Parameter(torch.randn(rank, n) * 0.1)])
        self.Rs = nn.ParameterList([nn.Parameter(torch.randn(rank, n) * 0.1)])
        self.Ds = nn.ParameterList([nn.Parameter(torch.randn(n, rank) * 0.1)])

    def forward(self, x):
        Lh = x @ self.Ls[0].T
        Rh = x @ self.Rs[0].T
        return (Lh * Rh) @ self.Ds[0].T


def task_2nd_argmax(x):
    return x.argsort(-1)[..., -2]


# --- Distributions ---

N = 4
EPSILONS = [0.0, 0.1, 0.5, 1.0]

def make_dist(eps):
    """Return a distribution function where last dim ~ N(-10, eps^2)."""
    def dist_fn(n_samples, n):
        x = torch.randn(n_samples, n)
        if eps == 0.0:
            x[:, -1] = -10.0
        else:
            x[:, -1] = -10.0 + eps * torch.randn(n_samples)
        return x
    dist_fn.__name__ = f'eps={eps}'
    return dist_fn

def gaussian(n_samples, n):
    return torch.randn(n_samples, n)

DISTRIBUTIONS = {f'eps={e}': make_dist(e) for e in EPSILONS}


# --- Config ---

NUM_SEEDS = 3
RANK = 32
CHECKPOINT_STEPS = [0, 32, 256, 4096]
TRAIN_STEPS = CHECKPOINT_STEPS[-1]
BATCH_SIZE = 512
FUNC_SIM_SAMPLES = 10_000


# --- Training ---

def train_with_checkpoints(dist_fn, seed):
    torch.manual_seed(seed)
    model = PureBilinear(N, rank=RANK)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    checkpoints = {}

    def save_checkpoint(step):
        model.eval()
        with torch.no_grad():
            x_eval = dist_fn(FUNC_SIM_SAMPLES, N)
            targets = task_2nd_argmax(x_eval)
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
        targets = task_2nd_argmax(x)
        loss = nn.functional.cross_entropy(model(x), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in CHECKPOINT_STEPS:
            save_checkpoint(step)
    return checkpoints


def load_model(state_dict):
    model = PureBilinear(N, rank=RANK)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def continuous_func_sim(m1, m2, dist_fn):
    m1.eval(); m2.eval()
    with torch.no_grad():
        x = dist_fn(FUNC_SIM_SAMPLES, N)
        y1 = m1(x).reshape(-1)
        y2 = m2(x).reshape(-1)
        cos = torch.dot(y1, y2) / (y1.norm() * y2.norm() + 1e-10)
    return cos.item()


# --- Phase 1: Train ---

print("=" * 60)
print("Phase 1: Training models")
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
    rank = torch.linalg.matrix_rank(cov).item()
    print(f"  {dist_name}: trace={cov.trace().item():.3f}, rank={rank}")


# --- Phase 3: Pairwise metrics ---

print("\n" + "=" * 60)
print("Phase 3: Computing pairwise metrics")
print("=" * 60)

results = []

for dist_name, dist_fn in DISTRIBUTIONS.items():
    keys = [(dist_name, seed, step)
            for seed in range(NUM_SEEDS)
            for step in CHECKPOINT_STEPS]
    pairs = list(itertools.combinations(keys, 2))
    print(f"  {dist_name}: {len(pairs)} pairs...", flush=True)

    Sigma = covariances[dist_name]

    for (d1, s1, st1), (d2, s2, st2) in pairs:
        m1 = models[(d1, s1, st1)]
        m2 = models[(d2, s2, st2)]

        ts_std = tensor_similarity(m1, m2)
        ts_gen = tensor_similarity(m1, m2, Sigma=Sigma)
        fs_train = continuous_func_sim(m1, m2, dist_fn)
        fs_gauss = continuous_func_sim(m1, m2, gaussian)

        results.append({
            'dist': dist_name,
            'seed1': s1, 'step1': st1,
            'seed2': s2, 'step2': st2,
            'tensor_sim_standard': ts_std,
            'tensor_sim_generalized': ts_gen,
            'func_sim_train': fs_train,
            'func_sim_gaussian': fs_gauss,
            'same_step': st1 == st2,
            'same_seed': s1 == s2,
        })


# --- Phase 4: Save results ---

print("\n" + "=" * 60)
print("Phase 4: Saving results")
print("=" * 60)

acc_serializable = {
    f"{dn}_seed{s}": {str(step): acc for step, acc in accs.items()}
    for (dn, s), accs in accuracies.items()
}

output = {
    'config': {
        'epsilons': EPSILONS,
        'num_seeds': NUM_SEEDS,
        'rank': RANK,
        'checkpoint_steps': CHECKPOINT_STEPS,
        'func_sim_samples': FUNC_SIM_SAMPLES,
    },
    'accuracies': acc_serializable,
    'results': results,
}

with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
    json.dump(output, f, indent=2)
print("  Saved results.json")


# --- Phase 5: Charts ---

print("\n" + "=" * 60)
print("Phase 5: Generating charts")
print("=" * 60)

dist_names = list(DISTRIBUTIONS.keys())

# Compute Pearson r for each dist and metric
r_standard = {}
r_generalized = {}
r_std_gauss = {}

for dist_name in dist_names:
    rows = [r for r in results if r['dist'] == dist_name]
    ts_std = [r['tensor_sim_standard'] for r in rows]
    ts_gen = [r['tensor_sim_generalized'] for r in rows]
    fs_train = [r['func_sim_train'] for r in rows]
    fs_gauss = [r['func_sim_gaussian'] for r in rows]

    r_standard[dist_name] = stats.pearsonr(ts_std, fs_train)[0] if len(ts_std) > 2 else 0
    r_generalized[dist_name] = stats.pearsonr(ts_gen, fs_train)[0] if len(ts_gen) > 2 else 0
    r_std_gauss[dist_name] = stats.pearsonr(ts_std, fs_gauss)[0] if len(ts_std) > 2 else 0

# Chart 1: Recovery curve — r vs epsilon
fig, ax = plt.subplots(figsize=(8, 5))
r_std_vals = [r_standard[d] for d in dist_names]
r_gen_vals = [r_generalized[d] for d in dist_names]
r_gauss_vals = [r_std_gauss[d] for d in dist_names]

ax.plot(EPSILONS, r_std_vals, 'o-', label='Standard TS vs func sim (train dist)', color='C0')
ax.plot(EPSILONS, r_gen_vals, 's--', label='Generalized TS vs func sim (train dist)', color='C1')
ax.plot(EPSILONS, r_gauss_vals, '^:', label='Standard TS vs func sim (gaussian)', color='C2')
ax.set_xlabel('ε (std of last dimension)')
ax.set_ylabel('Pearson r')
ax.set_title('Tensor Similarity vs Functional Similarity\nas Last-Dim Variance Increases (mean=-10)')
ax.legend()
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'recovery_curve.png'), dpi=150)
plt.close()
print("  Saved recovery_curve.png")

# Chart 2: Scatter grid — one panel per epsilon
fig, axes = plt.subplots(1, len(dist_names), figsize=(5 * len(dist_names), 5))
for ax, dist_name in zip(axes, dist_names):
    rows = [r for r in results if r['dist'] == dist_name]
    ts = [r['tensor_sim_standard'] for r in rows]
    fs = [r['func_sim_train'] for r in rows]
    same_step = [r['same_step'] for r in rows]

    ax.scatter([t for t, s in zip(ts, same_step) if not s],
               [f for f, s in zip(fs, same_step) if not s],
               alpha=0.4, s=12, color='C0', label='cross-step')
    ax.scatter([t for t, s in zip(ts, same_step) if s],
               [f for f, s in zip(fs, same_step) if s],
               alpha=0.4, s=12, color='C1', label='same-step')

    r_val = r_standard[dist_name]
    ax.set_title(f'{dist_name}\nr={r_val:.3f}')
    ax.set_xlabel('Tensor Sim (standard)')
    ax.set_ylabel('Func Sim (train dist)')
    ax.grid(True, alpha=0.3)
    if ax == axes[0]:
        ax.legend(fontsize=7)

plt.suptitle('Tensor Sim vs Functional Sim — Variance Sweep (no residual, 1 layer)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'scatter_grid.png'), dpi=150)
plt.close()
print("  Saved scatter_grid.png")

# Chart 3: Training curves
fig, axes = plt.subplots(1, len(dist_names), figsize=(5 * len(dist_names), 4))
for ax, dist_name in zip(axes, dist_names):
    for seed in range(NUM_SEEDS):
        accs = accuracies[(dist_name, seed)]
        steps = sorted(accs.keys())
        ax.plot(steps, [accs[s] for s in steps], alpha=0.6, marker='o', markersize=4)
    ax.set_title(dist_name)
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    ax.set_xscale('symlog', linthresh=1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

plt.suptitle('Training Curves — Variance Sweep', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'training_curves.png'), dpi=150)
plt.close()
print("  Saved training_curves.png")


# --- Summary ---

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"{'Distribution':<12} {'r(std,train)':>14} {'r(gen,train)':>14} {'r(std,gauss)':>14}")
print("-" * 56)
for dist_name in dist_names:
    print(f"{dist_name:<12} {r_standard[dist_name]:>14.3f} {r_generalized[dist_name]:>14.3f} {r_std_gauss[dist_name]:>14.3f}")

print("\nFinal accuracies:")
for dist_name in dist_names:
    accs = [accuracies[(dist_name, s)][TRAIN_STEPS] for s in range(NUM_SEEDS)]
    print(f"  {dist_name:<12} mean={np.mean(accs):.1%}  std={np.std(accs):.1%}")

print(f"\nDone! Results saved to {RESULTS_DIR}")
