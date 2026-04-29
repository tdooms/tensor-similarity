#%%
import sys
from pathlib import Path
import os

os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path.cwd().parent.parent))
sys.path.insert(0, str(Path.cwd()))

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
from functions.tn_sim import get_interaction_matrix

device = "cpu"

#%% HELPERS

def gaussian_sim_matrices(M1, M2):
    """Gaussian functional similarity between two symmetric bilinear forms.

    Applies Isserlis theorem: for x ~ N(0,I) and f(x) = x^T M x,
      E[f1·f2] = Tr(M1)Tr(M2) + 2·Tr(M1@M2)
      E[f²]    = Tr(M)² + 2·Tr(M²)
    M1 and M2 need not come from a model — works for any symmetric matrix.
    """
    cross = M1.trace() * M2.trace() + 2 * (M1 @ M2).trace()
    norm1 = M1.trace() ** 2 + 2 * (M1 @ M1).trace()
    norm2 = M2.trace() ** 2 + 2 * (M2 @ M2).trace()
    return (cross / (norm1 * norm2).sqrt()).item()


def diff_sim_slice(model, M_diff_digit, digit=7):
    """Similarity between a model's digit-d slice and a precomputed diff matrix slice."""
    M = get_interaction_matrix(model, include_embedding=True, symmetrize=True)[digit]
    return gaussian_sim_matrices(M, M_diff_digit)


def diff_sim_global(model, M_diff):
    """Global Gaussian functional similarity between a model and a diff (all classes).

    Sums Isserlis terms over all output classes:
      E[f_diff·f_prog] = Σ_o (Tr(D[o])Tr(P[o]) + 2·Tr(D[o]@P[o]))
    """
    M_prog = get_interaction_matrix(model, include_embedding=True, symmetrize=True)
    cross = sum(
        M_diff[o].trace() * M_prog[o].trace() + 2 * (M_diff[o] @ M_prog[o]).trace()
        for o in range(M_diff.shape[0])
    )
    norm_diff = sum(M_diff[o].trace()**2 + 2*(M_diff[o]@M_diff[o]).trace() for o in range(M_diff.shape[0]))
    norm_prog = sum(M_prog[o].trace()**2 + 2*(M_prog[o]@M_prog[o]).trace() for o in range(M_prog.shape[0]))
    return (cross / (norm_diff * norm_prog).sqrt()).item()


#%% CONFIG
epoch_setup = 20
d_hidden_setup = 256
d_embed_setup = 128
batch_size_setup = 248
record_every_n_batches_setup = 50

diff_seed = 123
prog_seed = 42
TARGET_DIGIT = 7

model_config = {
    'epochs': epoch_setup,
    'seed': diff_seed,
    'd_hidden': d_hidden_setup,
    'd_embed': d_embed_setup,
    'batch_size': batch_size_setup,
}

digit_curriculum = [
    {'name': 'base',     'digits': list(range(5)),  'epochs': epoch_setup},
    {'name': 'add 5',    'digits': list(range(6)),  'epochs': epoch_setup},
    {'name': 'add 6',    'digits': list(range(7)),  'epochs': epoch_setup},
    {'name': 'add 7',    'digits': list(range(8)),  'epochs': epoch_setup},
    {'name': 'add 8',    'digits': list(range(9)),  'epochs': epoch_setup},
    {'name': 'add 9',    'digits': list(range(10)), 'epochs': epoch_setup},
    {'name': 'repeat',   'digits': list(range(10)), 'epochs': epoch_setup},
    {'name': 'remove 9', 'digits': list(range(9)),  'epochs': epoch_setup},
    {'name': 're-add 9', 'digits': list(range(10)), 'epochs': epoch_setup},
]

stage_colors = {
    'base': 'red', 'add 5': 'blue', 'add 6': 'green', 'add 7': 'orange',
    'add 8': 'purple', 'add 9': 'brown', 'repeat': 'black',
    'remove 9': 'cyan', 're-add 9': 'magenta',
}

figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

#%% TRAIN DIFF MODELS (seed 123)
print("Training model A: all digits except 7 (seed 123)...")
digits_no7 = [d for d in range(10) if d != TARGET_DIGIT]
train_no7 = MNIST(train=True,  download=True, device=device, digits=digits_no7)
test_no7  = MNIST(train=False, download=True, device=device, digits=digits_no7)
model_A = Model.from_config(**model_config).to(device)
model_A.fit(train_no7, test_no7, RandomGaussianNoise(std=0.0))
model_A.eval()

print("Training model B: all digits (seed 456)...")
train_all = MNIST(train=True,  download=True, device=device)
test_all  = MNIST(train=False, download=True, device=device)
model_B = Model.from_config(**{**model_config, 'seed': 456}).to(device)
model_B.fit(train_all, test_all, RandomGaussianNoise(std=0.0))
model_B.eval()

#%% COMPUTE DIFF IN INTERACTION MATRIX SPACE
print("Computing interaction matrix diff...")
M_A = get_interaction_matrix(model_A, include_embedding=True, symmetrize=True)  # [10, 784, 784]
M_B = get_interaction_matrix(model_B, include_embedding=True, symmetrize=True)
M_diff = M_B - M_A  # captures what changed when digit 7 was added

M_diff_digit = M_diff[TARGET_DIGIT]  # [784, 784] — the digit-7 slice of the diff
# M_diff itself used for global sim

#%% TRAIN PROGRESSIVE MODEL (seed 42)
print(f"\nTraining progressive model (seed {prog_seed})...")
prog_config = {**model_config, 'seed': prog_seed}
progressive_checkpoints = {}
progressive_histories = {}
base_state = None

for stage in digit_curriculum:
    name, digits = stage['name'], stage['digits']
    train_data = MNIST(train=True,  download=True, device=device, digits=digits)
    test_data  = MNIST(train=False, download=True, device=device, digits=digits)

    model = Model.from_config(**prog_config).to(device)
    if base_state is not None:
        model.load_state_dict(base_state)

    history, checkpoints = model.fit(
        train_data, test_data,
        RandomGaussianNoise(std=0.0),
        record_every_n_batches=record_every_n_batches_setup,
        save_checkpoints=True,
    )
    progressive_checkpoints[name] = checkpoints
    progressive_histories[name] = history
    base_state = deepcopy(model.state_dict())
    print(f"  {name}: val_acc={history['val/acc'].iloc[-1]:.4f}")

#%% COMPUTE SIMILARITY TO DIFF
print("\nComputing similarity to diff...")
sim_batches = []
sim_slice   = []
sim_global  = []
sim_stages  = []
cumulative_batch = 0

for stage in digit_curriculum:
    name = stage['name']
    checkpoints = progressive_checkpoints[name]

    for cp in tqdm(checkpoints, desc=name):
        m = Model.from_config(**prog_config).to(device)
        m.load_state_dict(cp['state_dict'])
        m.eval()

        sim_slice.append(diff_sim_slice(m, M_diff_digit, digit=TARGET_DIGIT))
        sim_global.append(diff_sim_global(m, M_diff))
        sim_batches.append(cumulative_batch + cp['batch'])
        sim_stages.append(name)

    if checkpoints:
        cumulative_batch += checkpoints[-1]['batch']

sim_batches = np.array(sim_batches)
sim_slice   = np.array(sim_slice)
sim_global  = np.array(sim_global)

#%% BOUNDARIES
def stage_boundaries():
    boundaries, offset = [], 0
    for i, stage in enumerate(digit_curriculum):
        if i > 0:
            boundaries.append((offset, stage['name']))
        h = progressive_histories[stage['name']]
        if len(h) > 0:
            offset += h['batch'].iloc[-1]
    return boundaries

boundaries = stage_boundaries()

def add_stage_vlines(ax):
    from matplotlib.transforms import blended_transform_factory
    transform = blended_transform_factory(ax.transData, ax.transAxes)
    for x, name in boundaries:
        ax.axvline(x, color='black', linestyle=':', linewidth=2, alpha=0.6)
        ax.text(x, 0.7, name, fontsize=16,
                rotation=90, va='top', ha='right', color='black',
                fontstyle='italic', transform=transform)

FIGSIZE = (14, 5)

#%% PLOT — accuracy
fig, ax = plt.subplots(figsize=FIGSIZE)
offset = 0
for stage in digit_curriculum:
    h = progressive_histories[stage['name']]
    skip_zero = offset > 0
    cb, tr, vl = [], [], []
    for _, row in h.iterrows():
        if skip_zero and row['batch'] == 0:
            continue
        cb.append(offset + row['batch'])
        tr.append(row['train/acc'])
        vl.append(row['val/acc'])
    if len(h) > 0:
        offset += h['batch'].iloc[-1]
    ax.plot(cb, tr, color='steelblue', alpha=0.8)
    ax.plot(cb, vl, color='tomato', alpha=0.8)
add_stage_vlines(ax)
ax.set_xlabel("Cumulative batch")
ax.set_ylabel("Accuracy")
ax.set_ylim(0.9, 1.0)
ax.set_xlim(0, sim_batches[-1])
ax.grid(True, alpha=0.3)
ax.legend(handles=[
    plt.Line2D([0],[0], color='steelblue', label='Train'),
    plt.Line2D([0],[0], color='tomato',    label='Val'),
], fontsize=16, ncol=2)
plt.tight_layout()
plt.savefig(figures_dir / "paper_diffing_accuracy.png", dpi=150, bbox_inches='tight')
plt.show()

#%% PLOT — similarity to diff
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(sim_batches, sim_slice,  color='red',       alpha=0.9, label=f"Slice (digit {TARGET_DIGIT})")
ax.plot(sim_batches, sim_global, color='steelblue',  alpha=0.9, label="Global")
add_stage_vlines(ax)
ax.set_xlabel("Cumulative batch")
ax.set_ylabel("Similarity")
ax.set_ylim(-1, 1.05)
ax.set_xlim(0, sim_batches[-1])
ax.grid(True, alpha=0.3)
ax.legend(fontsize=16)
plt.tight_layout()
plt.savefig(figures_dir / "paper_diffing_similarity.png", dpi=150, bbox_inches='tight')
plt.show()

print("Done.")
# %%
