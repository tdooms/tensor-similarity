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
from src.training import fit
from _common import (
    FIGURES_DIR, load_mnist, load_svhn, new_model,
    tn_sim, naive_weight_cosine, linear_cka,
)

plt.rcParams.update({'font.size': 16})
plt.rcParams['lines.linewidth'] = 2.5

device = 'cpu'

#%% CONFIG
DATASET  = 'mnist'  # 'mnist' or 'svhn'
load_fn  = {'mnist': load_mnist, 'svhn': load_svhn}[DATASET]
D_INPUT  = {'mnist': 784, 'svhn': 1024}[DATASET]
seeds            = [42, 123]
reference_seed   = 42
EPOCHS           = 10
D_MODEL, D_HIDDEN = 128, 256
BATCH_SIZE       = 248
RECORD_EVERY     = 50

#%% LOAD DATA
train_data, val_data = load_fn(device=device)
cka_x = val_data.x[:2000]   # [2000, 1, 28, 28]

#%% TRAIN ALL SEEDS
histories   = {}
checkpoints = {}

for seed in seeds:
    print(f"\nTraining seed {seed}...")
    model = new_model(seed=seed, d_input=D_INPUT, d_model=D_MODEL, d_hidden=D_HIDDEN, device=device)
    h, cps = fit(model, train_data, val_data,
                 epochs=EPOCHS, batch_size=BATCH_SIZE,
                 record_every_n_batches=RECORD_EVERY,
                 save_checkpoints=True, seed=seed)
    histories[seed]   = h
    checkpoints[seed] = cps
    print(f"  final val_acc={h['val/acc'].iloc[-1]:.4f}")

#%% REFERENCE MODEL
ref_model = new_model(seed=reference_seed, d_input=D_INPUT, d_model=D_MODEL, d_hidden=D_HIDDEN, device=device)
ref_model.load_state_dict(checkpoints[reference_seed][-1]['state_dict'])
ref_model.eval()

#%% PRECOMPILE THOMAS SIMILARITY
other = new_model(seed=seeds[1], d_input=D_INPUT, d_model=D_MODEL, d_hidden=D_HIDDEN, device=device)
precompile(ref_model, other)

#%% REFERENCE ACTIVATIONS FOR CKA
with torch.no_grad():
    ref_hidden  = ref_model.body[0](ref_model.embed(ref_model.flatten(cka_x)))
    ref_logits  = ref_model(cka_x)

#%% COMPUTE SIMILARITIES
results = {seed: {'functional': [], 'naive': [], 'cka_hidden': [], 'cka_logits': [], 'batch': []}
           for seed in seeds}

for seed in seeds:
    print(f"\nComputing similarities for seed {seed}...")
    m_temp = new_model(seed=seed, d_input=D_INPUT, d_model=D_MODEL, d_hidden=D_HIDDEN, device=device)
    for cp in tqdm(checkpoints[seed]):
        m_temp.load_state_dict(cp['state_dict'])
        m_temp.eval()
        with torch.no_grad():
            hidden  = m_temp.body[0](m_temp.embed(m_temp.flatten(cka_x)))
            logits  = m_temp(cka_x)
        results[seed]['functional'].append(tn_sim(m_temp, ref_model))
        results[seed]['naive'].append(naive_weight_cosine(m_temp, ref_model))
        results[seed]['cka_hidden'].append(linear_cka(hidden, ref_hidden))
        results[seed]['cka_logits'].append(linear_cka(logits, ref_logits))
        results[seed]['batch'].append(cp['batch'])

#%% PLOT
seed_ls = {42: '-', 123: '--', 456: ':'}
sim_styles = {
    'naive':      ('crimson', "Weight Similarity"),
    'cka_hidden': ('orange',  "CKA (hidden)"),
    'cka_logits': ('green',   "CKA (logits)"),
    'functional': ('navy',    "Tensor Similarity"),
}

fig, ax = plt.subplots(figsize=(12, 5))
for seed in seeds:
    r = results[seed]
    batches = np.array(r['batch'])
    for metric, (color, label) in sim_styles.items():
        ax.plot(batches, np.array(r[metric]),
                color=color, linestyle=seed_ls[seed], alpha=0.8,
                label=label if seed == reference_seed else None)

ax.set_ylabel("Similarity", fontsize=24)
ax.set_xlabel("Batch step", fontsize=24)
ax.tick_params(axis='both', labelsize=20)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(handles=[
    plt.Line2D([0], [0], color=c, label=label)
    for _, (c, label) in sim_styles.items()
] + [
    plt.Line2D([0], [0], color='gray', linestyle=seed_ls[s], label=f"Seed {s}")
    for s in seeds
], fontsize=18, ncol=2)

plt.tight_layout()
plt.savefig(FIGURES_DIR / f"vision_seeds_summary_{DATASET}.png", dpi=150, bbox_inches='tight')
plt.show()
print("Done.")

# %%
