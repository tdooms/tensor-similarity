#%%
"""
Cross-seed similarity experiment.

Trains the same progressive curriculum across multiple seeds and compares:
  - Thomas's functional similarity (gauge-invariant)
  - Weight cosine similarity (gauge-dependent)

Key claim: functional similarity should stay high across seeds (same function
learned), while weight cosine may drop due to permutation/scaling ambiguity.
"""
import sys
from pathlib import Path
import os

os.chdir(Path(__file__).parent)
project_root = Path.cwd()
sys.path.insert(0, str(Path.cwd().parent.parent))  # for src.*
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})
plt.rcParams['lines.linewidth'] = 2.5
from tqdm import tqdm
from copy import deepcopy
from kornia.augmentation import RandomGaussianNoise

from functions.model import Model
from functions.datasets import MNIST
from functions.tn_sim import tn_sim_thomas, get_interaction_matrix


def slice_sim(m1, m2, digit=9):
    """Gaussian functional similarity restricted to a single output class.

    Applies Isserlis theorem to f_d(x) = x^T M[d] x for x ~ N(0,I):
      E[f1·f2] = Tr(M1)Tr(M2) + 2·Tr(M1@M2)
      E[f²]    = Tr(M)² + 2·Tr(M²)
    """
    M1 = get_interaction_matrix(m1, include_embedding=True, symmetrize=True)[digit]
    M2 = get_interaction_matrix(m2, include_embedding=True, symmetrize=True)[digit]
    cross = M1.trace() * M2.trace() + 2 * (M1 @ M2).trace()
    norm1 = M1.trace() ** 2 + 2 * (M1 @ M1).trace()
    norm2 = M2.trace() ** 2 + 2 * (M2 @ M2).trace()
    return (cross / (norm1 * norm2).sqrt()).item()

device = "cpu"

#%% CONFIG
seeds = [42]
reference_seed = 42

epoch_setup = 20
d_hidden_setup = 256
d_embed_setup = 128
batch_size_setup = 248
record_every_n_batches_setup = 50

digit_curriculum = [
    {'name': 'base',     'digits': list(range(5)),  'epochs': epoch_setup, 'lr': 1e-3},
    {'name': 'add 5',    'digits': list(range(6)),  'epochs': epoch_setup, 'lr': 1e-3},
    {'name': 'add 6',    'digits': list(range(7)),  'epochs': epoch_setup, 'lr': 1e-3},
    {'name': 'add 7',    'digits': list(range(8)),  'epochs': epoch_setup, 'lr': 1e-3},
    {'name': 'add 8',    'digits': list(range(9)),  'epochs': epoch_setup, 'lr': 1e-3},
    {'name': 'add 9',    'digits': list(range(10)), 'epochs': epoch_setup, 'lr': 1e-3},
    {'name': 'repeat',   'digits': list(range(10)), 'epochs': epoch_setup, 'lr': 1e-3},
    {'name': 'remove 9', 'digits': list(range(9)),  'epochs': epoch_setup, 'lr': 1e-3},
    {'name': 're-add 9', 'digits': list(range(10)), 'epochs': epoch_setup, 'lr': 1e-3},
]

stage_display = {
    'base':     '{0,...,4}',
    'add 5':    '{0,...,5}',
    'add 6':    '{0,...,6}',
    'add 7':    '{0,...,7}',
    'add 8':    '{0,...,8}',
    'add 9':    '{0,...,9}',
    'repeat':   '{0,...,9}',
    'remove 9': '{0,...,8}',
    're-add 9': '{0,...,9}',
}

stage_colors = {
    'base': 'red', 'add 5': 'blue', 'add 6': 'green', 'add 7': 'orange',
    'add 8': 'purple', 'add 9': 'brown', 'repeat': 'black', 'remove 9': 'cyan', 're-add 9': 'magenta',
}

base_model_config = {
    'epochs': epoch_setup,
    'seed': reference_seed,
    'd_hidden': d_hidden_setup,
    'd_embed': d_embed_setup,
    'batch_size': batch_size_setup,
}

figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

#%% TRAIN ALL SEEDS
progressive_checkpoints = {seed: {} for seed in seeds}
progressive_histories   = {seed: {} for seed in seeds}

for seed in seeds:
    print(f"\n{'='*60}\nTraining seed {seed}\n{'='*60}")
    cfg = {**base_model_config, 'seed': seed}
    base_state = None

    for stage in digit_curriculum:
        name, digits = stage['name'], stage['digits']
        train_data = MNIST(train=True,  download=True, device=device, digits=digits)
        test_data  = MNIST(train=False, download=True, device=device, digits=digits)

        model = Model.from_config(**cfg).to(device)
        if base_state is not None:
            model.load_state_dict(base_state)

        history, checkpoints = model.fit(
            train_data, test_data,
            RandomGaussianNoise(std=0.0),
            record_every_n_batches=record_every_n_batches_setup,
            save_checkpoints=True,
        )
        progressive_checkpoints[seed][name] = checkpoints
        progressive_histories[seed][name]   = history
        base_state = deepcopy(model.state_dict())
        print(f"  {name}: val_acc={history['val/acc'].iloc[-1]:.4f}")

#%% COMPUTE SIMILARITIES
# Reference: seed 42's final checkpoint
ref_sd = progressive_checkpoints[reference_seed]['add 9'][-1]['state_dict']
reference_model = Model.from_config(**base_model_config).to(device)
reference_model.load_state_dict(ref_sd)
reference_model.eval()

results = {seed: {'functional': [], 'batch': [], 'stage': []} for seed in seeds}

for seed in seeds:
    print(f"\nComputing similarities for seed {seed}...")
    cfg = {**base_model_config, 'seed': seed}
    cumulative_batch = 0

    for stage in digit_curriculum:
        name = stage['name']
        checkpoints = progressive_checkpoints[seed][name]

        for cp in tqdm(checkpoints, desc=f"{seed} {name}"):
            model_temp = Model.from_config(**cfg).to(device)
            model_temp.load_state_dict(cp['state_dict'])
            model_temp.eval()

            results[seed]['functional'].append(slice_sim(model_temp, reference_model, digit=7))
            results[seed]['batch'].append(cumulative_batch + cp['batch'])
            results[seed]['stage'].append(name)

        if checkpoints:
            cumulative_batch += checkpoints[-1]['batch']

#%% Heatmap computation

# --- Heatmap (seed 42, functional similarity, subsampled) ---
ref_checkpoints = progressive_checkpoints[reference_seed]
all_cps = []
cum = 0
for stage in digit_curriculum:
    for cp in ref_checkpoints[stage['name']]:
        all_cps.append({'batch': cum + cp['batch'], 'state_dict': cp['state_dict'], 'stage': stage['name']})
    if ref_checkpoints[stage['name']]:
        cum += ref_checkpoints[stage['name']][-1]['batch']

N_HEATMAP = min(40, len(all_cps))
indices = np.linspace(0, len(all_cps) - 1, N_HEATMAP, dtype=int)
heatmap_cps = [all_cps[i] for i in indices]
heatmap_batches = [cp['batch'] for cp in heatmap_cps]

cfg42 = {**base_model_config, 'seed': reference_seed}
heatmap_models = []
for cp in tqdm(heatmap_cps, desc="Loading heatmap models"):
    m = Model.from_config(**cfg42).to(device)
    m.load_state_dict(cp['state_dict'])
    m.eval()
    heatmap_models.append(m)

print(f"Computing {N_HEATMAP}x{N_HEATMAP} heatmap...")
heatmap = np.zeros((N_HEATMAP, N_HEATMAP))
for i in tqdm(range(N_HEATMAP)):
    for j in range(i + 1):
        s = slice_sim(heatmap_models[i], heatmap_models[j], digit=7)
        heatmap[i, j] = s
        heatmap[j, i] = s

#%% COMBINED SUMMARY PLOT
# Build cumulative history (acc, loss) and stage boundary positions per seed
def build_cumulative_history(seed):
    cum_batch, train_acc, val_acc, train_loss, val_loss = [], [], [], [], []
    offset = 0
    for stage in digit_curriculum:
        h = progressive_histories[seed][stage['name']]
        # skip batch=0 init checkpoint except for first stage to avoid duplicates
        skip_zero = offset > 0
        for _, row in h.iterrows():
            if skip_zero and row['batch'] == 0:
                continue
            cum_batch.append(offset + row['batch'])
            train_acc.append(row['train/acc'])
            val_acc.append(row['val/acc'])
            train_loss.append(row['train/loss'])
            val_loss.append(row['val/loss'])
        if len(h) > 0:
            offset += h['batch'].iloc[-1]
    return np.array(cum_batch), np.array(train_acc), np.array(val_acc), np.array(train_loss), np.array(val_loss)

# Stage boundary x-positions (start of each stage except first) from seed 42
def stage_boundaries(seed=reference_seed):
    boundaries, offset = [], 0
    for i, stage in enumerate(digit_curriculum):
        if i > 0:
            boundaries.append((offset, stage['name']))
        h = progressive_histories[seed][stage['name']]
        if len(h) > 0:
            offset += h['batch'].iloc[-1]
    return boundaries

boundaries = stage_boundaries()

#%% PLOT
seed_linestyles = {42: '-', 123: '--', 456: ':'}

def add_stage_vlines(ax):
    from matplotlib.transforms import blended_transform_factory
    transform = blended_transform_factory(ax.transData, ax.transAxes)
    for x, name in boundaries:
        ax.axvline(x, color='black', linestyle=':', linewidth=2, alpha=0.6)
        ax.text(x, 0.7, name, fontsize=16,
                rotation=90, va='top', ha='right', color='black',
                fontstyle='italic', transform=transform)

seed_ls = {42: '-', 123: '--', 456: ':'}
FIGSIZE = (14, 5)

# --- Accuracy ---
fig, ax_acc = plt.subplots(figsize=FIGSIZE)
for seed in seeds:
    cb, tr_acc, vl_acc, _, _ = build_cumulative_history(seed)
    ax_acc.plot(cb, tr_acc, color='steelblue', linestyle=seed_ls[seed], alpha=0.8)
    ax_acc.plot(cb, vl_acc, color='tomato',    linestyle=seed_ls[seed], alpha=0.8)
add_stage_vlines(ax_acc)
ax_acc.set_xlabel("Cumulative batch")
ax_acc.set_ylabel("Accuracy")
ax_acc.set_ylim(0.9, 1.00)
ax_acc.set_xlim(0, cb[-1])
ax_acc.grid(True, alpha=0.3)
ax_acc.legend(handles=[
    plt.Line2D([0],[0], color='steelblue', label='Test'),
    plt.Line2D([0],[0], color='tomato',    label='Validation'),
], fontsize=16, ncol=2)
plt.tight_layout()
plt.savefig(figures_dir / "paper_slice_accuracy.png", dpi=150, bbox_inches='tight')
plt.show()

# --- Loss (log scale) ---
fig, ax_loss = plt.subplots(figsize=FIGSIZE)
for seed in seeds:
    cb, _, _, tr_loss, vl_loss = build_cumulative_history(seed)
    ax_loss.plot(cb, tr_loss, color='steelblue', linestyle=seed_ls[seed], alpha=0.8)
    ax_loss.plot(cb, vl_loss, color='tomato',    linestyle=seed_ls[seed], alpha=0.8)
add_stage_vlines(ax_loss)
ax_loss.set_xlabel("Cumulative batch")
ax_loss.set_yscale('log')
ax_loss.set_ylabel("Loss (log scale)")
ax_loss.set_xlim(0, cb[-1])
ax_loss.grid(True, alpha=0.3, which='both')
ax_loss.legend(handles=[
    plt.Line2D([0],[0], color='steelblue', label='Test'),
    plt.Line2D([0],[0], color='tomato',    label='Validation'),
], fontsize=16, ncol=2)
plt.tight_layout()
plt.savefig(figures_dir / "paper_slice_loss.png", dpi=150, bbox_inches='tight')
plt.show()

# --- Tensor similarity ---
fig, ax_sim = plt.subplots(figsize=FIGSIZE)
for seed in seeds:
    r = results[seed]
    ax_sim.plot(np.array(r['batch']), np.array(r['functional']),
                color='red', linestyle=seed_ls[seed], alpha=0.9)
add_stage_vlines(ax_sim)
ax_sim.set_xlabel("Cumulative batch")
ax_sim.set_ylabel("Tensor Slice Similarity")
ax_sim.set_ylim(-0.15, 1.05)
ax_sim.set_xlim(0, cb[-1])
ax_sim.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / "paper_slice_similarity.png", dpi=150, bbox_inches='tight')
plt.show()


fig, ax_heatmap = plt.subplots(figsize=FIGSIZE)
im = ax_heatmap.imshow(heatmap, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax_heatmap, orientation='horizontal', fraction=0.046, pad=0.15)
ax_heatmap.set_xticks(range(N_HEATMAP)[::4])
ax_heatmap.set_yticks(range(N_HEATMAP)[::4])
ax_heatmap.set_xticklabels([heatmap_cps[i]['stage'] for i in range(N_HEATMAP)[::4]], rotation=45, ha='right', fontsize=14,fontstyle='italic')
ax_heatmap.set_yticklabels([heatmap_cps[i]['stage'] for i in range(N_HEATMAP)[::4]], fontsize=14,fontstyle='italic')

heatmap_stage_boundaries = [i - 0.5 for i in range(1, N_HEATMAP)
                             if heatmap_cps[i]['stage'] != heatmap_cps[i-1]['stage']]
for b in heatmap_stage_boundaries:
    ax_heatmap.axvline(b, color='black', linestyle=':', linewidth=1.5)
    ax_heatmap.axhline(b, color='black', linestyle=':', linewidth=1.5)
plt.tight_layout()
plt.savefig(figures_dir / "paper_slice_heatmap.png", dpi=150, bbox_inches='tight')
plt.show()
print("Done.")

# %%
