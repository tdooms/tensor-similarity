#%%
import sys
from pathlib import Path

_VISION = Path(__file__).resolve().parent
_REPO   = _VISION.parents[1]
for p in [str(_REPO), str(_VISION)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from _common import (
    FIGURES_DIR, load_mnist, load_svhn, new_model, slice_sim, linear_cka,
    fit_progressive, build_cumulative_history, get_stage_boundaries, add_stage_vlines,
)

plt.rcParams.update({'font.size': 16})
plt.rcParams['lines.linewidth'] = 2.5

device = 'cpu'
device = 'mps'
SLICE_DIGIT = 9
RECORD_EVERY = 50  # checkpoints every N batches

#%% CONFIG
DATASET  = 'svhn'  # 'mnist' or 'svhn'
load_fn  = {'mnist': load_mnist, 'svhn': load_svhn}[DATASET]
D_INPUT  = {'mnist': 784, 'svhn': 1024}[DATASET]
seeds          = [42]
reference_seed = 42
EPOCH_SETUP    = 20
D_MODEL, D_HIDDEN = 128, 256

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

#%% TRAIN ALL SEEDS
progressive_checkpoints = {}
progressive_histories   = {}

for seed in seeds:
    print(f"\n{'='*60}\nTraining seed {seed}\n{'='*60}")
    cps, hist = fit_progressive(seed, digit_curriculum, device=device,
                                d_input=D_INPUT, d_model=D_MODEL, d_hidden=D_HIDDEN,
                                record_every_n_batches=RECORD_EVERY,
                                load_fn=load_fn)
    progressive_checkpoints[seed] = cps
    progressive_histories[seed]   = hist

#%% REFERENCE MODEL (end of 'add 9' for reference seed)
ref_sd = progressive_checkpoints[reference_seed]['add 9'][-1]['state_dict']
ref_model = new_model(seed=reference_seed, d_input=D_INPUT, d_model=D_MODEL, d_hidden=D_HIDDEN, device=device)
ref_model.load_state_dict(ref_sd)
ref_model.eval()

#%% REFERENCE ACTIVATIONS FOR CKA
_, val_data = load_fn(device=device)
cka_x = val_data.x[:2000]
with torch.no_grad():
    ref_hidden = ref_model.body[0](ref_model.embed(ref_model.flatten(cka_x)))
    ref_logits = ref_model(cka_x)

#%% COMPUTE SIMILARITIES (slice_sim — no precompile needed)
results = {seed: {'functional': [], 'batch': [], 'stage': []} for seed in seeds}

for seed in seeds:
    print(f"\nComputing slice similarities for seed {seed}...")
    m_temp = new_model(seed=seed, d_input=D_INPUT, d_model=D_MODEL, d_hidden=D_HIDDEN, device=device)
    cumulative_batch = 0

    for stage in digit_curriculum:
        name = stage['name']
        checkpoints = progressive_checkpoints[seed][name]

        for cp in tqdm(checkpoints, desc=f"{seed} {name}"):
            m_temp.load_state_dict(cp['state_dict'])
            m_temp.eval()
            results[seed]['functional'].append(
                slice_sim(m_temp, ref_model, digit=SLICE_DIGIT))
            results[seed]['batch'].append(cumulative_batch + cp['batch'])
            results[seed]['stage'].append(name)

        if checkpoints:
            cumulative_batch += checkpoints[-1]['batch']

#%% HEATMAP (slice_sim)
ref_cps = progressive_checkpoints[reference_seed]
all_cps = []
cum = 0
for stage in digit_curriculum:
    for cp in ref_cps[stage['name']]:
        all_cps.append({'batch': cum + cp['batch'],
                        'state_dict': cp['state_dict'],
                        'stage': stage['name']})
    if ref_cps[stage['name']]:
        cum += ref_cps[stage['name']][-1]['batch']

N_HEATMAP = min(40, len(all_cps))
# N_HEATMAP =  200
indices   = np.linspace(0, len(all_cps) - 1, N_HEATMAP, dtype=int)
heatmap_cps = [all_cps[i] for i in indices]

m_heat = new_model(seed=reference_seed, d_input=D_INPUT, d_model=D_MODEL, d_hidden=D_HIDDEN, device=device)
heatmap_models = []
for cp in tqdm(heatmap_cps, desc="Loading heatmap models"):
    m_heat.load_state_dict(cp['state_dict'])
    heatmap_models.append(copy.deepcopy(m_heat).eval())

with torch.no_grad():
    heatmap_hidden = [m.body[0](m.embed(m.flatten(cka_x))) for m in heatmap_models]
    heatmap_logits = [m(cka_x) for m in heatmap_models]

print(f"Computing {N_HEATMAP}×{N_HEATMAP} heatmaps...")
heatmap       = np.zeros((N_HEATMAP, N_HEATMAP))
heatmap_cka_h = np.zeros((N_HEATMAP, N_HEATMAP))
heatmap_cka_l = np.zeros((N_HEATMAP, N_HEATMAP))
for i in tqdm(range(N_HEATMAP)):
    for j in range(i + 1):
        heatmap[i, j]       = heatmap[j, i]       = slice_sim(heatmap_models[i], heatmap_models[j], digit=SLICE_DIGIT)
        heatmap_cka_h[i, j] = heatmap_cka_h[j, i] = linear_cka(heatmap_hidden[i], heatmap_hidden[j])
        heatmap_cka_l[i, j] = heatmap_cka_l[j, i] = linear_cka(heatmap_logits[i], heatmap_logits[j])

#%% PLOT
seed_ls   = {42: '-', 123: '--', 456: ':'}
FIGSIZE   = (14, 5)
boundaries = get_stage_boundaries(progressive_histories[reference_seed], digit_curriculum)

# --- Accuracy ---
fig, ax = plt.subplots(figsize=FIGSIZE)
for seed in seeds:
    cb, tr_acc, vl_acc, _, _ = build_cumulative_history(progressive_histories[seed], digit_curriculum)
    ax.plot(cb, tr_acc, color='steelblue', linestyle=seed_ls[seed], alpha=0.8)
    ax.plot(cb, vl_acc, color='tomato',    linestyle=seed_ls[seed], alpha=0.8)
add_stage_vlines(ax, boundaries)
ax.set_xlabel("Cumulative batch")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.00)
ax.set_xlim(0, cb[-1])
ax.grid(True, alpha=0.3)
ax.legend(handles=[
    plt.Line2D([0], [0], color='steelblue', label='Train'),
    plt.Line2D([0], [0], color='tomato',    label='Val'),
], fontsize=16, ncol=2)
plt.tight_layout()
plt.savefig(FIGURES_DIR / f"vision_slice_accuracy_{DATASET}.png", dpi=150, bbox_inches='tight')
plt.show()

# --- Loss ---
fig, ax = plt.subplots(figsize=FIGSIZE)
for seed in seeds:
    cb, _, _, tr_loss, vl_loss = build_cumulative_history(progressive_histories[seed], digit_curriculum)
    ax.plot(cb, tr_loss, color='steelblue', linestyle=seed_ls[seed], alpha=0.8)
    ax.plot(cb, vl_loss, color='tomato',    linestyle=seed_ls[seed], alpha=0.8)
add_stage_vlines(ax, boundaries)
ax.set_xlabel("Cumulative batch")
ax.set_yscale('log')
ax.set_ylabel("Loss (log scale)")
ax.set_xlim(0, cb[-1])
ax.grid(True, alpha=0.3, which='both')
ax.legend(handles=[
    plt.Line2D([0], [0], color='steelblue', label='Train'),
    plt.Line2D([0], [0], color='tomato',    label='Val'),
], fontsize=16, ncol=2)
plt.tight_layout()
plt.savefig(FIGURES_DIR / f"vision_slice_loss_{DATASET}.png", dpi=150, bbox_inches='tight')
plt.show()

# --- Slice similarity ---
fig, ax = plt.subplots(figsize=FIGSIZE)
for seed in seeds:
    r = results[seed]
    ax.plot(np.array(r['batch']), np.array(r['functional']),
            color='red', linestyle=seed_ls[seed], alpha=0.9)
add_stage_vlines(ax, boundaries)
ax.set_xlabel("Cumulative batch")
ax.set_ylabel(f"Slice similarity (digit {SLICE_DIGIT})")
ax.set_ylim(-0.15, 1.05)
ax.set_xlim(0, cb[-1])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / f"vision_slice_similarity_{DATASET}.png", dpi=150, bbox_inches='tight')
plt.show()

heatmap_bounds = [i - 0.5 for i in range(1, N_HEATMAP)
                  if heatmap_cps[i]['stage'] != heatmap_cps[i - 1]['stage']]

stage_spans = {}
for i, cp in enumerate(heatmap_cps):
    stage_spans.setdefault(cp['stage'], []).append(i)
stage_ticks  = [idxs[len(idxs) // 2] for idxs in stage_spans.values()]
stage_labels = list(stage_spans.keys())

def _heatmap_plot(matrix, title, filename):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(matrix, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
    fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.15)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(stage_ticks)
    ax.set_yticks(stage_ticks)
    ax.set_xticklabels(stage_labels, rotation=45, ha='right', fontsize=14, fontstyle='italic')
    ax.set_yticklabels(stage_labels, fontsize=14, fontstyle='italic')
    for b in heatmap_bounds:
        ax.axvline(b, color='black', linestyle=':', linewidth=1.5)
        ax.axhline(b, color='black', linestyle=':', linewidth=1.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches='tight')
    plt.show()

# --- Heatmap (slice similarity) ---
_heatmap_plot(heatmap, f"Slice similarity (digit {SLICE_DIGIT})",
              f"vision_slice_heatmap_{DATASET}.png")

# # --- Heatmap CKA (hidden) ---
# _heatmap_plot(heatmap_cka_h, "CKA (hidden activations)",
#               f"vision_slice_heatmap_cka_hidden_{DATASET}.png")

# --- Heatmap CKA (logits) ---
_heatmap_plot(heatmap_cka_l, "CKA (logits)",
              f"vision_slice_heatmap_cka_logits_{DATASET}.png")

print("Done.")

# %%
