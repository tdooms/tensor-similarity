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
import torch
from tqdm import tqdm

from _common import (
    DATA_DIR, load_mnist, load_svhn, new_model, slice_sim, fit,
    get_interaction_matrix, fit_progressive, build_cumulative_history, get_stage_spans,
)

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
SAVE_DATA      = True

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
M_diff      = M_diff_full[TARGET_DIGIT]    # [d_input, d_input] — digit-9 slice

#%% TRAIN PROGRESSIVE MODEL
print("\nTraining progressive model...")
prog_cps, prog_hist = fit_progressive(
    SEED_A, digit_curriculum, device=device,
    d_input=D_INPUT, d_model=D_MODEL, d_hidden=D_HIDDEN,
    record_every_n_batches=RECORD_EVERY,
    load_fn=partial(load_fn, offset=OFFSET_A),
)

#%% COMPUTE DIFF SIMILARITY OVER TRAINING
def diff_sim_slice(model):
    """Isserlis cosine between model's digit-9 slice and M_diff."""
    M = get_interaction_matrix(model)[TARGET_DIGIT]
    cross = M.trace() * M_diff.trace() + 2 * (M @ M_diff).trace()
    norm1 = M.trace() ** 2      + 2 * (M @ M).trace()
    norm2 = M_diff.trace() ** 2 + 2 * (M_diff @ M_diff).trace()
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


slice_vals, global_vals, diff_batches = [], [], []
cumulative_batch = 0
m_temp = new_model(seed=SEED, d_input=D_INPUT, d_model=D_MODEL, d_hidden=D_HIDDEN, device=device)

for stage in digit_curriculum:
    name = stage['name']
    checkpoints = prog_cps[name]
    for cp in tqdm(checkpoints, desc=name):
        m_temp.load_state_dict(cp['state_dict'])
        m_temp.eval()
        slice_vals.append(diff_sim_slice(m_temp))
        global_vals.append(diff_sim_global(m_temp))
        diff_batches.append(cumulative_batch + cp['batch'])
    if checkpoints:
        cumulative_batch += checkpoints[-1]['batch']

slice_vals   = np.array(slice_vals)
global_vals  = np.array(global_vals)
diff_batches = np.array(diff_batches)

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

#%% HEATMAP — compute slice similarity for digit 9
print(f"Computing {N_HEATMAP}×{N_HEATMAP} slice similarity heatmap (digit {TARGET_DIGIT})...")
heatmap_slice = np.zeros((N_HEATMAP, N_HEATMAP))
for i in tqdm(range(N_HEATMAP)):
    for j in range(i + 1):
        heatmap_slice[i, j] = heatmap_slice[j, i] = slice_sim(
            heatmap_models[i], heatmap_models[j], digit=TARGET_DIGIT)

#%% SAVE DATA
if SAVE_DATA:
    import json
    cum_batch, tr_acc, vl_acc, tr_loss, vl_loss = build_cumulative_history(prog_hist, digit_curriculum)
    spans_tmp = get_stage_spans(prog_hist, digit_curriculum)
    np.savez_compressed(DATA_DIR / f"diffing_{DATASET}.npz",
        heatmap_slice=heatmap_slice,
        slice_vals=slice_vals,
        global_vals=global_vals,
        diff_batches=diff_batches,
        cum_batch=cum_batch,
        train_acc=tr_acc, val_acc=vl_acc,
        train_loss=tr_loss, val_loss=vl_loss,
    )
    meta = {
        'heatmap_cps': [{'batch': int(cp['batch']), 'stage': cp['stage']} for cp in heatmap_cps],
        'spans': [[int(x0), int(x1), name] for x0, x1, name in spans_tmp],
        'xlim': [0, int(cum_batch[-1])],
        'diff_xlim': [int(diff_batches[0]), int(diff_batches[-1])],
        'dataset': DATASET,
        'target_digit': TARGET_DIGIT,
    }
    with open(DATA_DIR / f"diffing_{DATASET}_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Data saved to {DATA_DIR}")

print("\nDone. Render figures with: uv run prepare svhn-diffing && uv run plot svhn-diffing")
