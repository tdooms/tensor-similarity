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

from src.components.similarity import precompile
from _common import (
    FIGURES_DIR, load_mnist, load_svhn, new_model, tn_sim, linear_cka,
    fit_progressive, build_cumulative_history, get_stage_spans,
)
from _plots import line_plot, heatmap_plot, save_show

plt.rcParams.update({'font.size': 16})
plt.rcParams['lines.linewidth'] = 2.5

device = 'cpu'
device = 'mps'

#%% CONFIG
DATASET  = 'svhn'  # 'mnist' or 'svhn'
load_fn  = {'mnist': load_mnist, 'svhn': load_svhn}[DATASET]
D_INPUT  = {'mnist': 784, 'svhn': 1024}[DATASET]
seeds          = [123]
reference_seed = 123
EPOCH_SETUP    = 20
D_MODEL, D_HIDDEN = 128, 256
RECORD_EVERY = 50  # checkpoints every N batches

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

#%% PRECOMPILE THOMAS SIMILARITY
dummy = new_model(seed=0, d_input=D_INPUT, d_model=D_MODEL, d_hidden=D_HIDDEN, device=device)
precompile(ref_model, dummy)

#%% REFERENCE ACTIVATIONS FOR CKA
_, val_data = load_fn(device=device)
cka_x = val_data.x[:2000]
with torch.no_grad():
    ref_hidden = ref_model.body[0](ref_model.embed(ref_model.flatten(cka_x)))
    ref_logits = ref_model(cka_x)

#%% COMPUTE SIMILARITIES
results = {seed: {'functional': [], 'cka_hidden': [], 'cka_logits': [], 'batch': [], 'stage': []} for seed in seeds}

for seed in seeds:
    print(f"\nComputing similarities for seed {seed}...")
    m_temp = new_model(seed=seed, d_input=D_INPUT, d_model=D_MODEL, d_hidden=D_HIDDEN, device=device)
    cumulative_batch = 0

    for stage in digit_curriculum:
        name = stage['name']
        checkpoints = progressive_checkpoints[seed][name]

        for cp in tqdm(checkpoints, desc=f"{seed} {name}"):
            m_temp.load_state_dict(cp['state_dict'])
            m_temp.eval()
            with torch.no_grad():
                hidden = m_temp.body[0](m_temp.embed(m_temp.flatten(cka_x)))
                logits = m_temp(cka_x)
            results[seed]['functional'].append(tn_sim(m_temp, ref_model))
            results[seed]['cka_hidden'].append(linear_cka(hidden, ref_hidden))
            results[seed]['cka_logits'].append(linear_cka(logits, ref_logits))
            results[seed]['batch'].append(cumulative_batch + cp['batch'])
            results[seed]['stage'].append(name)

        if checkpoints:
            cumulative_batch += checkpoints[-1]['batch']

#%% HEATMAP
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

N_HEATMAP = min(100, len(all_cps))
indices   = np.linspace(0, len(all_cps) - 1, N_HEATMAP, dtype=int)
heatmap_cps = [all_cps[i] for i in indices]

m_heat = new_model(seed=reference_seed, d_input=D_INPUT, d_model=D_MODEL, d_hidden=D_HIDDEN, device=device)
heatmap_models = []
for cp in tqdm(heatmap_cps, desc="Loading heatmap models"):
    m_heat.load_state_dict(cp['state_dict'])
    import copy
    heatmap_models.append(copy.deepcopy(m_heat).eval())

# with torch.no_grad():
#     heatmap_hidden = [m.body[0](m.embed(m.flatten(cka_x))) for m in heatmap_models]
#     heatmap_logits = [m(cka_x) for m in heatmap_models]

print(f"Computing {N_HEATMAP}×{N_HEATMAP} similarity heatmap...")
heatmap = np.zeros((N_HEATMAP, N_HEATMAP))
# heatmap_cka_h = np.zeros((N_HEATMAP, N_HEATMAP))
# heatmap_cka_l = np.zeros((N_HEATMAP, N_HEATMAP))
for i in tqdm(range(N_HEATMAP)):
    for j in range(i + 1):
        heatmap[i, j] = heatmap[j, i] = tn_sim(heatmap_models[i], heatmap_models[j])
        # heatmap_cka_h[i, j] = heatmap_cka_h[j, i] = linear_cka(heatmap_hidden[i], heatmap_hidden[j])
        # heatmap_cka_l[i, j] = heatmap_cka_l[j, i] = linear_cka(heatmap_logits[i], heatmap_logits[j])

#%% PLOT
seed_ls    = {42: '-', 123: '--', 456: ':'}
spans      = get_stage_spans(progressive_histories[reference_seed], digit_curriculum)
acc_legend = [plt.Line2D([0], [0], color='steelblue', label='Train'),
              plt.Line2D([0], [0], color='tomato',    label='Val')]

hists = {s: build_cumulative_history(progressive_histories[s], digit_curriculum) for s in seeds}
cb   = hists[reference_seed][0]
xlim = (0, cb[-1])

acc_series  = [(hists[s][0], hists[s][1], dict(color='steelblue', linestyle=seed_ls[s], alpha=0.8)) for s in seeds]
val_series  = [(hists[s][0], hists[s][2], dict(color='tomato',    linestyle=seed_ls[s], alpha=0.8)) for s in seeds]
loss_series = [(hists[s][0], hists[s][3], dict(color='steelblue', linestyle=seed_ls[s], alpha=0.8)) for s in seeds]
vl_series   = [(hists[s][0], hists[s][4], dict(color='tomato',    linestyle=seed_ls[s], alpha=0.8)) for s in seeds]

fig, _ = line_plot(acc_series + val_series, spans,
                   "Cumulative batch", "Accuracy", ylim=(0, 1), xlim=xlim, legend=acc_legend)
save_show(fig, FIGURES_DIR / f"vision_forgetting_accuracy_{DATASET}.png")

fig, _ = line_plot(loss_series + vl_series, spans,
                   "Cumulative batch", "Loss (log scale)", xlim=xlim, log_y=True, legend=acc_legend)
save_show(fig, FIGURES_DIR / f"vision_forgetting_loss_{DATASET}.png")

fig, _ = line_plot(
    [(np.array(results[s]['batch']), np.array(results[s]['functional']),
      dict(color='red', linestyle=seed_ls[s], alpha=0.9)) for s in seeds],
    spans, "Cumulative batch", "Tensor similarity", ylim=(-0.05, 1.05), xlim=xlim)
save_show(fig, FIGURES_DIR / f"vision_forgetting_similarity_{DATASET}.png")

fig, _ = line_plot(
    [(np.array(results[s]['batch']), np.array(results[s][m]),
      dict(color=c, linestyle=seed_ls[s], alpha=0.9, label=lbl if s == seeds[0] else None))
     for s in seeds for m, c, lbl in
     [('cka_hidden', 'orange', 'CKA (hidden)'), ('cka_logits', 'green', 'CKA (logits)')]],
    spans, "Cumulative batch", "CKA similarity", ylim=(-0.05, 1.05), xlim=xlim,
    legend=[plt.Line2D([0], [0], color='orange', label='CKA (hidden)'),
            plt.Line2D([0], [0], color='green',  label='CKA (logits)')])
save_show(fig, FIGURES_DIR / f"vision_forgetting_cka_{DATASET}.png")

fig, _ = heatmap_plot(heatmap,       heatmap_cps, "Tensor similarity")
save_show(fig, FIGURES_DIR / f"vision_forgetting_heatmap_{DATASET}.png")

# fig, _ = heatmap_plot(heatmap_cka_h, heatmap_cps, "CKA (hidden activations)")
# save_show(fig, FIGURES_DIR / f"vision_forgetting_heatmap_cka_hidden_{DATASET}.png")

fig, _ = heatmap_plot(heatmap_cka_l, heatmap_cps, "CKA (logits)")
save_show(fig, FIGURES_DIR / f"vision_forgetting_heatmap_cka_logits_{DATASET}.png")

print("Done.")

# %%
