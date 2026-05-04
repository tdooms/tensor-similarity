#%%
import sys
from pathlib import Path

_VISION = Path(__file__).resolve().parent
_REPO   = _VISION.parents[1]
for p in [str(_REPO), str(_VISION)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from src.training import fit
from _common import (
    FIGURES_DIR, load_mnist, load_svhn, new_model,
    get_interaction_matrix,
    fit_progressive, build_cumulative_history, get_stage_boundaries, add_stage_vlines,
)

plt.rcParams.update({'font.size': 16})
plt.rcParams['lines.linewidth'] = 2.5

device = 'cpu'
TARGET_DIGIT = 7

#%% CONFIG
DATASET  = 'mnist'  # 'mnist' or 'svhn'
load_fn  = {'mnist': load_mnist, 'svhn': load_svhn}[DATASET]
D_INPUT  = {'mnist': 784, 'svhn': 1024}[DATASET]
EPOCH_SETUP   = 20
D_MODEL, D_HIDDEN = 128, 256
BATCH_SIZE    = 248
RECORD_EVERY  = 50

digit_curriculum = [
    {'name': 'base',     'digits': list(range(5)),  'epochs': EPOCH_SETUP},
    {'name': 'add 5',    'digits': list(range(6)),  'epochs': EPOCH_SETUP},
    {'name': 'add 6',    'digits': list(range(7)),  'epochs': EPOCH_SETUP},
    {'name': 'add 7',    'digits': list(range(8)),  'epochs': EPOCH_SETUP},
    {'name': 'add 8',    'digits': list(range(9)),  'epochs': EPOCH_SETUP},
    {'name': 'add 9',    'digits': list(range(10)), 'epochs': EPOCH_SETUP},
    {'name': 'repeat',   'digits': list(range(10)), 'epochs': EPOCH_SETUP},
    {'name': 'remove 9', 'digits': list(range(9)),  'epochs': EPOCH_SETUP},
    {'name': 're-add 9', 'digits': list(range(10)), 'epochs': EPOCH_SETUP},
]

#%% TRAIN DIFF MODELS
print("Training model A: all digits except 7 (seed 123)...")
digits_no7  = [d for d in range(10) if d != TARGET_DIGIT]
train_no7, val_no7 = load_fn(device=device, digits=digits_no7)
model_A = new_model(seed=123, d_input=D_INPUT, d_model=D_MODEL, d_hidden=D_HIDDEN, device=device)
fit(model_A, train_no7, val_no7, epochs=EPOCH_SETUP,
    batch_size=BATCH_SIZE, seed=123)
model_A.eval()

print("Training model B: all digits (seed 456)...")
train_all, val_all = load_fn(device=device)
model_B = new_model(seed=456, d_input=D_INPUT, d_model=D_MODEL, d_hidden=D_HIDDEN, device=device)
fit(model_B, train_all, val_all, epochs=EPOCH_SETUP,
    batch_size=BATCH_SIZE, seed=456)
model_B.eval()

#%% COMPUTE DIFF IN INTERACTION MATRIX SPACE
print("Computing interaction matrix diff...")
with torch.no_grad():
    M_A    = get_interaction_matrix(model_A)   # [10, 784, 784]
    M_B    = get_interaction_matrix(model_B)
M_diff       = M_B - M_A                       # what changed when digit 7 was added
M_diff_digit = M_diff[TARGET_DIGIT]            # [784, 784] — digit-7 slice


def _isserlis_cosine_1d(M1, M2):
    cross = M1.trace() * M2.trace() + 2 * (M1 @ M2).trace()
    norm1 = M1.trace() ** 2 + 2 * (M1 @ M1).trace()
    norm2 = M2.trace() ** 2 + 2 * (M2 @ M2).trace()
    return (cross / (norm1 * norm2).sqrt()).item()


def diff_sim_slice(model):
    M = get_interaction_matrix(model)[TARGET_DIGIT]
    return _isserlis_cosine_1d(M, M_diff_digit)


def diff_sim_global(model):
    M_prog = get_interaction_matrix(model)
    n = M_diff.shape[0]
    cross     = sum(M_diff[o].trace() * M_prog[o].trace() + 2 * (M_diff[o] @ M_prog[o]).trace()
                    for o in range(n))
    norm_diff = sum(M_diff[o].trace() ** 2 + 2 * (M_diff[o] @ M_diff[o]).trace() for o in range(n))
    norm_prog = sum(M_prog[o].trace() ** 2 + 2 * (M_prog[o] @ M_prog[o]).trace() for o in range(n))
    return (cross / (norm_diff * norm_prog).sqrt()).item()


#%% TRAIN PROGRESSIVE MODEL (seed 42)
print("\nTraining progressive model (seed 42)...")
prog_cps, prog_hist = fit_progressive(
    42, digit_curriculum, device=device,
    d_input=D_INPUT, d_model=D_MODEL, d_hidden=D_HIDDEN,
    batch_size=BATCH_SIZE, record_every_n_batches=RECORD_EVERY,
    load_fn=load_fn,
)

#%% COMPUTE SIMILARITY TO DIFF
print("\nComputing similarity to diff...")
sim_batches, sim_slice_vals, sim_global_vals = [], [], []
cumulative_batch = 0

m_temp = new_model(seed=42, d_input=D_INPUT, d_model=D_MODEL, d_hidden=D_HIDDEN, device=device)
for stage in digit_curriculum:
    name = stage['name']
    checkpoints = prog_cps[name]
    for cp in tqdm(checkpoints, desc=name):
        m_temp.load_state_dict(cp['state_dict'])
        m_temp.eval()
        sim_slice_vals.append(diff_sim_slice(m_temp))
        sim_global_vals.append(diff_sim_global(m_temp))
        sim_batches.append(cumulative_batch + cp['batch'])
    if checkpoints:
        cumulative_batch += checkpoints[-1]['batch']

sim_batches     = np.array(sim_batches)
sim_slice_vals  = np.array(sim_slice_vals)
sim_global_vals = np.array(sim_global_vals)

#%% BOUNDARIES
boundaries = get_stage_boundaries(prog_hist, digit_curriculum)
FIGSIZE    = (14, 5)

#%% PLOT — accuracy
fig, ax = plt.subplots(figsize=FIGSIZE)
offset = 0
for stage in digit_curriculum:
    h = prog_hist[stage['name']]
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
    ax.plot(cb, vl, color='tomato',    alpha=0.8)
add_stage_vlines(ax, boundaries)
ax.set_xlabel("Cumulative batch")
ax.set_ylabel("Accuracy")
ax.set_ylim(0.9, 1.0)
ax.set_xlim(0, sim_batches[-1])
ax.grid(True, alpha=0.3)
ax.legend(handles=[
    plt.Line2D([0], [0], color='steelblue', label='Train'),
    plt.Line2D([0], [0], color='tomato',    label='Val'),
], fontsize=16, ncol=2)
plt.tight_layout()
plt.savefig(FIGURES_DIR / f"vision_diffing_accuracy_{DATASET}.png", dpi=150, bbox_inches='tight')
plt.show()

#%% PLOT — similarity to diff
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(sim_batches, sim_slice_vals,  color='red',       alpha=0.9, label=f"Slice (digit {TARGET_DIGIT})")
ax.plot(sim_batches, sim_global_vals, color='steelblue', alpha=0.9, label="Global")
add_stage_vlines(ax, boundaries)
ax.set_xlabel("Cumulative batch")
ax.set_ylabel("Similarity to diff")
ax.set_ylim(-1, 1.05)
ax.set_xlim(0, sim_batches[-1])
ax.grid(True, alpha=0.3)
ax.legend(fontsize=16)
plt.tight_layout()
plt.savefig(FIGURES_DIR / f"vision_diffing_similarity_{DATASET}.png", dpi=150, bbox_inches='tight')
plt.show()

print("Done.")
