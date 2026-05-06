#%%
import sys
from pathlib import Path

_VISION = Path(__file__).resolve().parent
_REPO   = _VISION.parents[1]
for p in [str(_REPO), str(_VISION)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import copy
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from _common import (
    FIGURES_DIR, DATA_DIR, load_mnist, load_svhn, new_model, slice_sim, fit,
    get_interaction_matrix, fit_progressive, build_cumulative_history, get_stage_spans,
)
from _plots import line_plot, slice_heatmap_plot, per_digit_line_plot, save_show

plt.rcParams.update({'font.size': 16})
plt.rcParams['lines.linewidth'] = 2.5

device = 'cpu'
device = 'mps'
TARGET_DIGIT = 9

#%% CONFIG
DATASET  = 'svhn'  # 'mnist' or 'svhn'
load_fn  = {'mnist': load_mnist, 'svhn': load_svhn}[DATASET]
D_INPUT  = {'mnist': 784, 'svhn': 1024}[DATASET]
EPOCH_SETUP    = 20
D_MODEL, D_HIDDEN = 128, 256
RECORD_EVERY   = 50
SEED           = 123
SAVE_DATA    = True
SAVE_FIGURES = True

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

#%% BUILD DIFF MODEL (what changes in model B when digit 9 is fine-tuned in)
# Model A (progressive, seed 123) uses data offset 0.
# Model B (seed 456) uses a non-overlapping data subset (offset = n_train).
SEED_A, SEED_B = 123, 456
OFFSET_A, OFFSET_B = 0, 73257

digits_base = list(range(TARGET_DIGIT))      # 0–8
digits_all  = list(range(TARGET_DIGIT + 1))  # 0–9

print(f"Training model B base on digits 0–{TARGET_DIGIT - 1} (seed {SEED_B}, offset {OFFSET_B})...")
train_B_base, val_B = load_fn(device=device, digits=digits_base, offset=OFFSET_B)
model_B_base = new_model(seed=SEED_B, d_input=D_INPUT, d_model=D_MODEL, d_hidden=D_HIDDEN, device=device)
fit(model_B_base, train_B_base, val_B, epochs=EPOCH_SETUP, seed=SEED_B)
model_B_base.eval()

print(f"Fine-tuning model B on all digits 0–{TARGET_DIGIT} (seed {SEED_B}, offset {OFFSET_B})...")
import copy as _copy
model_B_ft = _copy.deepcopy(model_B_base)
train_B_ft, _ = load_fn(device=device, digits=digits_all, offset=OFFSET_B)
fit(model_B_ft, train_B_ft, val_B, epochs=EPOCH_SETUP, seed=SEED_B)
model_B_ft.eval()

with torch.no_grad():
    M_B_base = get_interaction_matrix(model_B_base)
    M_B_ft   = get_interaction_matrix(model_B_ft)
M_diff_full = M_B_ft - M_B_base            # [10, d_input, d_input]

#%% TRAIN PROGRESSIVE MODEL
print("\nTraining progressive model...")
prog_cps, prog_hist = fit_progressive(
    SEED_A, digit_curriculum, device=device,
    d_input=D_INPUT, d_model=D_MODEL, d_hidden=D_HIDDEN,
    record_every_n_batches=RECORD_EVERY,
    load_fn=partial(load_fn, offset=OFFSET_A),
)

#%% COMPUTE DIFF SIMILARITY OVER TRAINING (all digits)
def diff_sim_slice(model, digit):
    """Isserlis cosine between model's digit slice and M_diff_full[digit]."""
    M  = get_interaction_matrix(model)[digit]
    Md = M_diff_full[digit]
    cross = M.trace() * Md.trace() + 2 * (M @ Md).trace()
    norm1 = M.trace() ** 2 + 2 * (M @ M).trace()
    norm2 = Md.trace() ** 2 + 2 * (Md @ Md).trace()
    return (cross / (norm1 * norm2).sqrt()).item()


def diff_sim_global(model):
    """Isserlis cosine between model's full interaction matrix and M_diff_full."""
    M_prog = get_interaction_matrix(model)
    cross     = sum(M_diff_full[o].trace() * M_prog[o].trace() + 2 * (M_diff_full[o] @ M_prog[o]).trace()
                    for o in range(M_diff_full.shape[0]))
    norm_diff = sum(M_diff_full[o].trace() ** 2 + 2 * (M_diff_full[o] @ M_diff_full[o]).trace()
                    for o in range(M_diff_full.shape[0]))
    norm_prog = sum(M_prog[o].trace() ** 2      + 2 * (M_prog[o] @ M_prog[o]).trace()
                    for o in range(M_diff_full.shape[0]))
    return (cross / (norm_diff * norm_prog).sqrt()).item()


slice_vals_all = {d: [] for d in range(10)}
global_vals, diff_batches = [], []
cumulative_batch = 0
m_temp = new_model(seed=SEED, d_input=D_INPUT, d_model=D_MODEL, d_hidden=D_HIDDEN, device=device)

for stage in digit_curriculum:
    name = stage['name']
    checkpoints = prog_cps[name]
    for cp in tqdm(checkpoints, desc=name):
        m_temp.load_state_dict(cp['state_dict'])
        m_temp.eval()
        for d in range(10):
            slice_vals_all[d].append(diff_sim_slice(m_temp, d))
        global_vals.append(diff_sim_global(m_temp))
        diff_batches.append(cumulative_batch + cp['batch'])
    if checkpoints:
        cumulative_batch += checkpoints[-1]['batch']

slice_vals_all = {d: np.array(v) for d, v in slice_vals_all.items()}
global_vals    = np.array(global_vals)
diff_batches   = np.array(diff_batches)

#%% HEATMAP — load progressive checkpoints
all_cps, cum = [], 0
for stage in digit_curriculum:
    for cp in prog_cps[stage['name']]:
        all_cps.append({'batch': cum + cp['batch'],
                        'state_dict': cp['state_dict'],
                        'stage': stage['name']})
    if prog_cps[stage['name']]:
        cum += prog_cps[stage['name']][-1]['batch']

N_HEATMAP   = min(80, len(all_cps))
indices     = np.linspace(0, len(all_cps) - 1, N_HEATMAP, dtype=int)
heatmap_cps = [all_cps[i] for i in indices]

m_heat = new_model(seed=SEED, d_input=D_INPUT, d_model=D_MODEL, d_hidden=D_HIDDEN, device=device)
heatmap_models = []
for cp in tqdm(heatmap_cps, desc="Loading heatmap models"):
    m_heat.load_state_dict(cp['state_dict'])
    heatmap_models.append(copy.deepcopy(m_heat).eval())

#%% HEATMAP — compute slice similarity for all digits
print(f"Computing {N_HEATMAP}×{N_HEATMAP} slice similarity heatmaps (all digits)...")
slice_heatmaps = {d: np.zeros((N_HEATMAP, N_HEATMAP)) for d in range(10)}
for i in tqdm(range(N_HEATMAP)):
    for j in range(i + 1):
        for d in range(10):
            slice_heatmaps[d][i, j] = slice_heatmaps[d][j, i] = slice_sim(
                heatmap_models[i], heatmap_models[j], digit=d)

#%% SAVE DATA
if SAVE_DATA:
    import json
    cum_batch, tr_acc, vl_acc, tr_loss, vl_loss = build_cumulative_history(prog_hist, digit_curriculum)
    spans_tmp = get_stage_spans(prog_hist, digit_curriculum)
    np.savez_compressed(DATA_DIR / f"diffing_slicing_{DATASET}.npz",
        global_vals=global_vals,
        diff_batches=diff_batches,
        cum_batch=cum_batch,
        train_acc=tr_acc, val_acc=vl_acc,
        train_loss=tr_loss, val_loss=vl_loss,
        **{f'slice_heatmap_{d}': slice_heatmaps[d] for d in range(10)},
        **{f'slice_vals_{d}':    slice_vals_all[d]  for d in range(10)},
    )
    meta = {
        'heatmap_cps': [{'batch': int(cp['batch']), 'stage': cp['stage']} for cp in heatmap_cps],
        'spans': [[int(x0), int(x1), name] for x0, x1, name in spans_tmp],
        'xlim': [0, int(cum_batch[-1])],
        'diff_xlim': [int(diff_batches[0]), int(diff_batches[-1])],
        'dataset': DATASET,
        'target_digit': TARGET_DIGIT,
    }
    with open(DATA_DIR / f"diffing_slicing_{DATASET}_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Data saved to {DATA_DIR}")

#%% PLOT — accuracy
spans = get_stage_spans(prog_hist, digit_curriculum)
cum_batch, tr_acc, vl_acc, tr_loss, vl_loss = build_cumulative_history(prog_hist, digit_curriculum)
xlim = (0, cum_batch[-1])

acc_legend = [plt.Line2D([0], [0], color='steelblue', label='Train'),
              plt.Line2D([0], [0], color='tomato',    label='Val')]

fig, _ = line_plot(
    [(cum_batch, tr_acc, dict(color='steelblue', alpha=0.8)),
     (cum_batch, vl_acc, dict(color='tomato',    alpha=0.8))],
    spans, "Cumulative batch", "Accuracy", ylim=(0, 1), xlim=xlim, legend=acc_legend)
save_show(fig, FIGURES_DIR / f"vision_diffing_slicing_accuracy_{DATASET}.png", save=SAVE_FIGURES)

#%% PLOT — diff similarity over training (target digit + global)
fig, _ = line_plot(
    [(diff_batches, slice_vals_all[TARGET_DIGIT], dict(color='red',       label=f'Slice (digit {TARGET_DIGIT})', alpha=0.9)),
     (diff_batches, global_vals,                  dict(color='steelblue', label='Global',                        alpha=0.9))],
    spans, "Cumulative batch", "Similarity to diff",
    ylim=(-1.05, 1.05), xlim=xlim,
    legend=[plt.Line2D([0], [0], color='red',       label=f'Slice (digit {TARGET_DIGIT})'),
            plt.Line2D([0], [0], color='steelblue', label='Global')])
save_show(fig, FIGURES_DIR / f"vision_diffing_slicing_diff_sim_{DATASET}.png", save=SAVE_FIGURES)

#%% PLOT — per-digit diff similarity line plots
fig = per_digit_line_plot(
    slice_vals_all, diff_batches, spans, "Similarity to diff", ylim=(-1.05, 1.05))
save_show(fig, FIGURES_DIR / f"vision_diffing_slicing_diff_sim_digits_{DATASET}.png", save=SAVE_FIGURES)

#%% PLOT — 2×5 slice similarity heatmaps
fig = slice_heatmap_plot(slice_heatmaps, heatmap_cps, label='Tensor slice similarity')
save_show(fig, FIGURES_DIR / f"vision_diffing_slicing_heatmap_slice_{DATASET}.png", save=SAVE_FIGURES)

print("Done.")

# %%
