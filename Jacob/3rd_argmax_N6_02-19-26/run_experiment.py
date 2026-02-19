"""
3rd Argmax (N=6): 1-Layer vs 2-Layer Tensor Network Experiment

Research questions:
  1. Can a single bilinear layer (rank=64) solve 3rd argmax on N=6?
  2. For 2-layer networks, does per-layer Sigma improve tensor similarity accuracy?

Architecture (equal parameter budget):
  1-layer: BilinearStack(n=6, num_layers=1, rank=64)  -- 18*64 = 1152 params
  2-layer: BilinearStack(n=6, num_layers=2, rank=32)  -- 2*18*32 = 1152 params

Tensor similarity variants:
  standard:    Sigma=I for all layers
  generalized: Sigma=Cov[x] for all layers (same for both model types)
  layer_sigma: (2-layer only) layer 0 uses Cov[x], layer 1 uses Cov[h1] (per-model, averaged)
"""

import os
import json
import itertools
import importlib.util
import torch
import torch.nn as nn
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Import model module ---
spec = importlib.util.spec_from_file_location("bilinear", "bilinear_3rd_argmax.py")
bilinear = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bilinear)

from metrics import (
    tensor_similarity, tensor_similarity_layer_sigma,
    estimate_layer_input_covariance, estimate_covariance,
    functional_similarity,
)

# --- Config ---
N            = bilinear.N
BATCH_SIZE   = bilinear.BATCH_SIZE
NUM_SEEDS    = 3
# Equal parameter budgets: 1-layer rank=64, 2-layer rank=32 both give 18*rank params
RANK_1LAYER  = 64
RANK_2LAYER  = 32
CHECKPOINT_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
TRAIN_STEPS  = CHECKPOINT_STEPS[-1]
FUNC_SAMPLES = 50000
COV_SAMPLES  = 100000

DISTRIBUTIONS = {
    'gaussian':            bilinear.gaussian,
    'half_gaussian':       bilinear.half_gaussian,
    'bimodal':             bilinear.bimodal,
    'uniform':             bilinear.uniform,
    'laplace':             bilinear.laplace,
    'sparse_spikes':       bilinear.sparse_spikes,
    'permutation':         bilinear.permutation,
    'correlated_gaussian': bilinear.correlated_gaussian,
}

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Phase 1: Train ──────────────────────────────────────────────────────────

def train_with_checkpoints(dist_fn, num_layers, rank, seed):
    torch.manual_seed(seed)
    model = bilinear.BilinearStack(N, num_layers=num_layers, rank=rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    checkpoints = {}

    def save_checkpoint(step):
        model.eval()
        with torch.no_grad():
            x_eval = dist_fn(FUNC_SAMPLES, N)
            targets = bilinear.task_3rd_argmax(x_eval)
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
        targets = bilinear.task_3rd_argmax(x)
        loss = nn.functional.cross_entropy(model(x), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in CHECKPOINT_STEPS:
            save_checkpoint(step)
    return checkpoints


def load_model(state_dict, num_layers, rank):
    model = bilinear.BilinearStack(N, num_layers=num_layers, rank=rank)
    model.load_state_dict(state_dict)
    model.eval()
    return model


print("=" * 60)
print("Phase 1: Training models")
print(f"  1-layer rank={RANK_1LAYER} and 2-layer rank={RANK_2LAYER}  (equal param budget)")
print(f"  {len(DISTRIBUTIONS)} dists × {NUM_SEEDS} seeds × {len(CHECKPOINT_STEPS)} checkpoints")
print("=" * 60)

models_1L = {}   # (dist_name, seed, step) -> model
models_2L = {}
accs_1L   = {}   # (dist_name, seed) -> {step: acc}
accs_2L   = {}

for dist_name, dist_fn in DISTRIBUTIONS.items():
    for seed in range(NUM_SEEDS):
        print(f"  {dist_name} seed={seed}  ", end="", flush=True)

        ckpts = train_with_checkpoints(dist_fn, num_layers=1, rank=RANK_1LAYER, seed=seed)
        accs = {}
        for step, ckpt in ckpts.items():
            models_1L[(dist_name, seed, step)] = load_model(ckpt['state_dict'], 1, RANK_1LAYER)
            accs[step] = ckpt['accuracy']
        accs_1L[(dist_name, seed)] = accs
        print(f"1L={accs[TRAIN_STEPS]:.1%}  ", end="", flush=True)

        ckpts = train_with_checkpoints(dist_fn, num_layers=2, rank=RANK_2LAYER, seed=seed)
        accs = {}
        for step, ckpt in ckpts.items():
            models_2L[(dist_name, seed, step)] = load_model(ckpt['state_dict'], 2, RANK_2LAYER)
            accs[step] = ckpt['accuracy']
        accs_2L[(dist_name, seed)] = accs
        print(f"2L={accs[TRAIN_STEPS]:.1%}")


# ─── Phase 2: Estimate covariances ───────────────────────────────────────────

print("\n" + "=" * 60)
print("Phase 2: Estimating covariances")
print("=" * 60)

# Input covariances — shared by all models of same distribution
input_covs = {}
for dist_name, dist_fn in DISTRIBUTIONS.items():
    cov = estimate_covariance(dist_fn, N, COV_SAMPLES)
    input_covs[dist_name] = cov
    print(f"  {dist_name}: trace={cov.trace().item():.3f}")

# Per-model layer-1 covariances for 2-layer models
# Cov[h1] = Cov[x + T0(x,x)] differs per model and per checkpoint
print("\n  Per-model Cov[h1] for 2-layer models...", flush=True)
layer1_covs = {}   # (dist_name, seed, step) -> 6×6 tensor
n_done = 0
n_total = len(DISTRIBUTIONS) * NUM_SEEDS * len(CHECKPOINT_STEPS)
for dist_name, dist_fn in DISTRIBUTIONS.items():
    for seed in range(NUM_SEEDS):
        for step in CHECKPOINT_STEPS:
            model = models_2L[(dist_name, seed, step)]
            layer1_covs[(dist_name, seed, step)] = estimate_layer_input_covariance(
                model, dist_fn, layer_idx=1, n=N, num_samples=COV_SAMPLES
            )
            n_done += 1
            if n_done % 30 == 0:
                print(f"    {n_done}/{n_total}", flush=True)
print(f"    {n_done}/{n_total}  done.")


# ─── Phase 3: Pairwise metrics ────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Phase 3: Computing pairwise metrics")
print("=" * 60)

within_1L = []
within_2L = []

for dist_name, dist_fn in DISTRIBUTIONS.items():
    keys = [(dist_name, s, st) for s in range(NUM_SEEDS) for st in CHECKPOINT_STEPS]
    pairs = list(itertools.combinations(keys, 2))
    Sigma_in = input_covs[dist_name]
    print(f"  {dist_name}: {len(pairs)} pairs...", flush=True)

    for idx, ((d1, s1, st1), (d2, s2, st2)) in enumerate(pairs):
        # --- 1-layer ---
        m1 = models_1L[(d1, s1, st1)]
        m2 = models_1L[(d2, s2, st2)]
        within_1L.append({
            'dist': dist_name,
            'seed1': s1, 'step1': st1, 'seed2': s2, 'step2': st2,
            'tensor_sim_standard':    tensor_similarity(m1, m2),
            'tensor_sim_generalized': tensor_similarity(m1, m2, Sigma=Sigma_in),
            'func_sim_train':         functional_similarity(m1, m2, dist_fn,          n=N, num_samples=FUNC_SAMPLES),
            'func_sim_gaussian':      functional_similarity(m1, m2, bilinear.gaussian, n=N, num_samples=FUNC_SAMPLES),
            'same_step': st1 == st2,
        })

        # --- 2-layer ---
        m1 = models_2L[(d1, s1, st1)]
        m2 = models_2L[(d2, s2, st2)]
        sigmas_m1 = [Sigma_in, layer1_covs[(d1, s1, st1)]]
        sigmas_m2 = [Sigma_in, layer1_covs[(d2, s2, st2)]]
        within_2L.append({
            'dist': dist_name,
            'seed1': s1, 'step1': st1, 'seed2': s2, 'step2': st2,
            'tensor_sim_standard':    tensor_similarity(m1, m2),
            'tensor_sim_generalized': tensor_similarity(m1, m2, Sigma=Sigma_in),
            'tensor_sim_layer_sigma': tensor_similarity_layer_sigma(m1, m2, sigmas_m1, sigmas_m2),
            'func_sim_train':         functional_similarity(m1, m2, dist_fn,          n=N, num_samples=FUNC_SAMPLES),
            'func_sim_gaussian':      functional_similarity(m1, m2, bilinear.gaussian, n=N, num_samples=FUNC_SAMPLES),
            'same_step': st1 == st2,
        })

        if (idx + 1) % 200 == 0:
            print(f"    {idx + 1}/{len(pairs)}", flush=True)


# ─── Phase 4: Save results ────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Phase 4: Saving results")
print("=" * 60)

def serialize_accs(accs_dict):
    return {f"{d}_seed{s}": {str(st): v for st, v in accs.items()}
            for (d, s), accs in accs_dict.items()}

results = {
    'config': {
        'n': N, 'rank_1layer': RANK_1LAYER, 'rank_2layer': RANK_2LAYER,
        'num_seeds': NUM_SEEDS, 'checkpoint_steps': CHECKPOINT_STEPS,
        'distributions': list(DISTRIBUTIONS.keys()),
    },
    'accuracies_1layer': serialize_accs(accs_1L),
    'accuracies_2layer': serialize_accs(accs_2L),
    'within_1layer': within_1L,
    'within_2layer': within_2L,
}

results_path = os.path.join(RESULTS_DIR, 'results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Saved {results_path}")


# ─── Phase 5: Charts ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Phase 5: Generating charts")
print("=" * 60)

dist_names = list(DISTRIBUTIONS.keys())
CHANCE = 1.0 / N   # 1/6

# ── Chart 1: Training curves, 1-layer vs 2-layer ──────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
for idx, dist_name in enumerate(dist_names):
    ax = axes[idx]
    for model_type, accs_dict, color, label in [
        ('1-layer', accs_1L, 'C0', f'1-layer (rank={RANK_1LAYER})'),
        ('2-layer', accs_2L, 'C1', f'2-layer (rank={RANK_2LAYER})'),
    ]:
        seed_accs = []
        for seed in range(NUM_SEEDS):
            accs = accs_dict[(dist_name, seed)]
            steps_sorted = sorted(accs.keys())
            seed_accs.append([accs[s] for s in steps_sorted])
        mean_a = np.mean(seed_accs, axis=0)
        std_a  = np.std(seed_accs, axis=0)
        ax.plot(steps_sorted, mean_a, color=color, linewidth=2, label=label)
        ax.fill_between(steps_sorted, mean_a - std_a, mean_a + std_a, alpha=0.2, color=color)
        for sa in seed_accs:
            ax.plot(steps_sorted, sa, alpha=0.2, color=color, linewidth=0.8)
    ax.axhline(CHANCE, color='gray', linestyle='--', linewidth=1, label='chance')
    ax.set_title(dist_name)
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    ax.set_xscale('symlog', linthresh=1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=8)

plt.suptitle('Training Curves: 1-Layer vs 2-Layer (3rd Argmax, N=6)', fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'training_curves.png'), dpi=150)
plt.close()
print("  Saved training_curves.png")

# ── Chart 2: Final accuracy bar chart ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(dist_names))
width = 0.35
for offset, accs_dict, color, label, rank in [
    (-width / 2, accs_1L, 'C0', f'1-layer (rank={RANK_1LAYER})', RANK_1LAYER),
    ( width / 2, accs_2L, 'C1', f'2-layer (rank={RANK_2LAYER})', RANK_2LAYER),
]:
    means = []
    errs  = []
    for dist_name in dist_names:
        vals = [accs_dict[(dist_name, s)][TRAIN_STEPS] for s in range(NUM_SEEDS)]
        means.append(np.mean(vals))
        errs.append(np.std(vals))
    ax.bar(x + offset, means, width, color=color, label=label,
           yerr=errs, capsize=4, error_kw={'linewidth': 1.2})
ax.axhline(CHANCE, color='gray', linestyle='--', linewidth=1.5, label=f'chance (1/6)')
ax.set_xticks(x)
ax.set_xticklabels(dist_names, rotation=30, ha='right')
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1.05)
ax.set_title(f'Final Accuracy at Step {TRAIN_STEPS} (3rd Argmax, N=6, equal param budget)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'final_accuracy.png'), dpi=150)
plt.close()
print("  Saved final_accuracy.png")

# ── Chart 3: Scatter grid, 2-layer layer-sigma metric vs func sim ─────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
for idx, dist_name in enumerate(dist_names):
    ax = axes[idx]
    rows = [r for r in within_2L if r['dist'] == dist_name]
    ts = [r['tensor_sim_layer_sigma'] for r in rows]
    fs = [r['func_sim_train'] for r in rows]
    same_step = [r['same_step'] for r in rows]

    ax.scatter([t for t, s in zip(ts, same_step) if not s],
               [f for f, s in zip(fs, same_step) if not s],
               alpha=0.25, s=6, color='C0', label='cross-step')
    ax.scatter([t for t, s in zip(ts, same_step) if s],
               [f for f, s in zip(fs, same_step) if s],
               alpha=0.35, s=8, color='C1', label='same-step')

    r_val = stats.pearsonr(ts, fs)[0] if len(ts) > 2 else 0.0
    ax.set_title(f'{dist_name}\nr={r_val:.3f}')
    ax.set_xlabel('Tensor Sim (layer-sigma)')
    ax.set_ylabel('Func Sim (train dist)')
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=7)

plt.suptitle('2-Layer: Layer-Sigma Tensor Sim vs Functional Sim', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'scatter_2layer_layer_sigma.png'), dpi=150)
plt.close()
print("  Saved scatter_2layer_layer_sigma.png")

# ── Chart 4: Metric correlation bars for 2-layer (standard vs gen vs layer-sigma) ──
fig, ax = plt.subplots(figsize=(14, 6))
metrics_to_compare = [
    ('tensor_sim_standard',    'Standard (Σ=I)',             'C0'),
    ('tensor_sim_generalized', 'Generalized (Σ=input cov)',  'C1'),
    ('tensor_sim_layer_sigma', 'Layer-Sigma (Σ_i=Cov[h_i])', 'C2'),
]
x = np.arange(len(dist_names))
width = 0.25
for i, (metric_key, label, color) in enumerate(metrics_to_compare):
    r_vals = []
    for dist_name in dist_names:
        rows = [r for r in within_2L if r['dist'] == dist_name]
        ts = [r[metric_key] for r in rows]
        fs = [r['func_sim_train'] for r in rows]
        r_val = stats.pearsonr(ts, fs)[0] if len(ts) > 2 else 0.0
        r_vals.append(r_val)
    offset = (i - 1) * width
    ax.bar(x + offset, r_vals, width, color=color, label=label)

ax.set_xticks(x)
ax.set_xticklabels(dist_names, rotation=30, ha='right')
ax.set_ylabel('Pearson r (tensor sim vs func sim)')
ax.set_ylim(0, 1.05)
ax.set_title('2-Layer Tensor Similarity Metric Comparison (correlation with func sim)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'metric_correlations_2layer.png'), dpi=150)
plt.close()
print("  Saved metric_correlations_2layer.png")

# ── Chart 5: Same plots for 1-layer (for reference) ─────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
for idx, dist_name in enumerate(dist_names):
    ax = axes[idx]
    rows = [r for r in within_1L if r['dist'] == dist_name]
    ts_std = [r['tensor_sim_standard']    for r in rows]
    ts_gen = [r['tensor_sim_generalized'] for r in rows]
    fs     = [r['func_sim_train']         for r in rows]
    same_step = [r['same_step'] for r in rows]

    ax.scatter([t for t, s in zip(ts_std, same_step) if not s],
               [f for f, s in zip(fs, same_step) if not s],
               alpha=0.25, s=6, color='C0', label='cross-step (standard)')
    ax.scatter([t for t, s in zip(ts_gen, same_step) if not s],
               [f for f, s in zip(fs, same_step) if not s],
               alpha=0.25, s=6, color='C1', label='cross-step (generalized)')

    r_std = stats.pearsonr(ts_std, fs)[0] if len(ts_std) > 2 else 0.0
    r_gen = stats.pearsonr(ts_gen, fs)[0] if len(ts_gen) > 2 else 0.0
    ax.set_title(f'{dist_name}\nstd r={r_std:.3f}  gen r={r_gen:.3f}')
    ax.set_xlabel('Tensor Sim')
    ax.set_ylabel('Func Sim (train dist)')
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=7)

plt.suptitle('1-Layer: Tensor Sim vs Functional Sim (reference)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'scatter_1layer.png'), dpi=150)
plt.close()
print("  Saved scatter_1layer.png")

print("\n" + "=" * 60)
print("Done! All results saved to", RESULTS_DIR)
print("=" * 60)
