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
import torch
from tqdm import tqdm

from src.components.similarity import precompile
from _common import (
    DATA_DIR, load_mnist, load_svhn, new_model, tn_sim, linear_cka,
    slice_sim, naive_weight_cosine, weight_slice_cosine,
    fit_progressive, build_cumulative_history, get_stage_spans,
)

device = 'cpu'
device = 'mps'
HeatMapSize = 80
SAVE_DATA = True

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
    ref_logits = ref_model(cka_x)

#%% COMPUTE SIMILARITIES & PER-DIGIT ACCURACY (vs reference model over training)
digit_val_x = {d: val_data.x[val_data.y == d][:500] for d in range(10)}
digit_val_y = {d: val_data.y[val_data.y == d][:500] for d in range(10)}

sim_results = {
    'functional': [], 'cka_logits': [], 'weight_cosine': [],
    **{f'slice_{d}': [] for d in range(10)},
    **{f'acc_{d}':   [] for d in range(10)},
    'batch': [],
}

m_temp = new_model(seed=reference_seed, d_input=D_INPUT, d_model=D_MODEL, d_hidden=D_HIDDEN, device=device)
cumulative_batch = 0
for stage in digit_curriculum:
    name = stage['name']
    checkpoints = progressive_checkpoints[reference_seed][name]
    for cp in tqdm(checkpoints, desc=name):
        m_temp.load_state_dict(cp['state_dict'])
        m_temp.eval()
        with torch.no_grad():
            logits       = m_temp(cka_x)
            digit_logits = {d: m_temp(digit_val_x[d]) for d in range(10)}
        sim_results['functional'].append(tn_sim(m_temp, ref_model))
        sim_results['cka_logits'].append(linear_cka(logits, ref_logits))
        sim_results['weight_cosine'].append(naive_weight_cosine(m_temp, ref_model))
        for d in range(10):
            sim_results[f'slice_{d}'].append(slice_sim(m_temp, ref_model, digit=d))
            sim_results[f'acc_{d}'].append(
                (digit_logits[d].argmax(-1) == digit_val_y[d]).float().mean().item())
        sim_results['batch'].append(cumulative_batch + cp['batch'])
    if checkpoints:
        cumulative_batch += checkpoints[-1]['batch']

#%% HEATMAP — load models
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

N_HEATMAP = min(HeatMapSize, len(all_cps))
indices     = np.linspace(0, len(all_cps) - 1, N_HEATMAP, dtype=int)
heatmap_cps = [all_cps[i] for i in indices]

m_heat = new_model(seed=reference_seed, d_input=D_INPUT, d_model=D_MODEL, d_hidden=D_HIDDEN, device=device)
heatmap_models = []
for cp in tqdm(heatmap_cps, desc="Loading heatmap models"):
    m_heat.load_state_dict(cp['state_dict'])
    heatmap_models.append(copy.deepcopy(m_heat).eval())

with torch.no_grad():
    heatmap_logits = [m(cka_x) for m in heatmap_models]

#%% HEATMAP — compute similarities
print(f"Computing {N_HEATMAP}×{N_HEATMAP} similarity matrices...")
heatmap_tn           = np.zeros((N_HEATMAP, N_HEATMAP))
heatmap_cka_l        = np.zeros((N_HEATMAP, N_HEATMAP))
heatmap_weight       = np.zeros((N_HEATMAP, N_HEATMAP))
slice_heatmaps       = {d: np.zeros((N_HEATMAP, N_HEATMAP)) for d in range(10)}
slice_heatmaps_weight = {d: np.zeros((N_HEATMAP, N_HEATMAP)) for d in range(10)}

for i in tqdm(range(N_HEATMAP)):
    for j in range(i + 1):
        heatmap_tn[i, j]     = heatmap_tn[j, i]     = tn_sim(heatmap_models[i], heatmap_models[j])
        heatmap_cka_l[i, j]  = heatmap_cka_l[j, i]  = linear_cka(heatmap_logits[i], heatmap_logits[j])
        heatmap_weight[i, j] = heatmap_weight[j, i]  = naive_weight_cosine(heatmap_models[i], heatmap_models[j])
        for d in range(10):
            slice_heatmaps[d][i, j]        = slice_heatmaps[d][j, i]        = slice_sim(heatmap_models[i], heatmap_models[j], digit=d)
            slice_heatmaps_weight[d][i, j] = slice_heatmaps_weight[d][j, i] = weight_slice_cosine(heatmap_models[i], heatmap_models[j], digit=d)

#%% SAVE DATA
if SAVE_DATA:
    import json
    hists_ref = build_cumulative_history(progressive_histories[reference_seed], digit_curriculum)
    np.savez_compressed(DATA_DIR / f"forgetting_{DATASET}.npz",
        heatmap_tn=heatmap_tn, heatmap_cka_l=heatmap_cka_l, heatmap_weight=heatmap_weight,
        cum_batch=hists_ref[0], train_acc=hists_ref[1], val_acc=hists_ref[2],
        train_loss=hists_ref[3], val_loss=hists_ref[4],
        sim_batch=np.array(sim_results['batch']),
        sim_functional=np.array(sim_results['functional']),
        sim_cka_logits=np.array(sim_results['cka_logits']),
        sim_weight_cosine=np.array(sim_results['weight_cosine']),
        **{f'slice_heatmap_{d}':        slice_heatmaps[d]        for d in range(10)},
        **{f'slice_heatmap_weight_{d}': slice_heatmaps_weight[d] for d in range(10)},
        **{f'sim_slice_{d}':  np.array(sim_results[f'slice_{d}']) for d in range(10)},
        **{f'sim_acc_{d}':    np.array(sim_results[f'acc_{d}'])   for d in range(10)},
    )
    spans_tmp = get_stage_spans(progressive_histories[reference_seed], digit_curriculum)
    sim_batch_tmp = np.array(sim_results['batch'])
    meta = {
        'heatmap_cps': [{'batch': int(cp['batch']), 'stage': cp['stage']} for cp in heatmap_cps],
        'spans': [[int(x0), int(x1), name] for x0, x1, name in spans_tmp],
        'xlim': [0, int(hists_ref[0][-1])],
        'sim_xlim': [int(sim_batch_tmp[0]), int(sim_batch_tmp[-1])],
        'dataset': DATASET,
    }
    with open(DATA_DIR / f"forgetting_{DATASET}_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Data saved to {DATA_DIR}")

#%% STAGE ACCURACY SUMMARY
print(f"\n{'stage':<12} {'train_start':>11} {'val_start':>9} {'train_end':>9} {'val_end':>7}")
print("-" * 52)
for stage in digit_curriculum:
    h = progressive_histories[reference_seed][stage['name']]
    if len(h) == 0:
        continue
    print(f"{stage['name']:<12} "
          f"{h['train/acc'].iloc[0]:>11.4f} "
          f"{h['val/acc'].iloc[0]:>9.4f} "
          f"{h['train/acc'].iloc[-1]:>9.4f} "
          f"{h['val/acc'].iloc[-1]:>7.4f}")

print("\nDone. Render figures with: uv run prepare svhn-forgetting && uv run plot svhn-forgetting")
