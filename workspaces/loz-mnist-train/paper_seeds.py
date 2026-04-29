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
from tqdm import tqdm
from copy import deepcopy
from kornia.augmentation import RandomGaussianNoise

from functions.model import Model
from functions.datasets import MNIST
from functions.tn_sim import model_similarity, tn_sim_thomas

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})

device = "cpu"

#%% CONFIG
seeds = [42, 123]#, 456]
reference_seed = 42

model_config = {
    'epochs': 10,
    'd_hidden': 256,
    'd_embed': 128,
    'batch_size': 248,
    'seed': reference_seed,
}

figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

#%% TRAIN ALL SEEDS
train_data = MNIST(train=True,  download=True, device=device)
test_data  = MNIST(train=False, download=True, device=device)

histories   = {}
checkpoints = {}

for seed in seeds:
    print(f"\nTraining seed {seed}...")
    cfg = {**model_config, 'seed': seed}
    model = Model.from_config(**cfg).to(device)
    history, cps = model.fit(
        train_data, test_data,
        RandomGaussianNoise(std=0.0),
        record_every_n_batches=50,
        save_checkpoints=True,
    )
    histories[seed]   = history
    checkpoints[seed] = cps
    print(f"  final val_acc={history['val/acc'].iloc[-1]:.4f}")

#%% COMPUTE SIMILARITIES
# Reference: seed 42's final checkpoint
ref_model = Model.from_config(**{**model_config, 'seed': reference_seed}).to(device)
ref_model.load_state_dict(checkpoints[reference_seed][-1]['state_dict'])
ref_model.eval()

cka_data = MNIST(train=False, download=True, device=device)
cka_x = cka_data.x[:2000].flatten(start_dim=1)

def naive_weight_cosine(m1, m2):
    w1 = torch.cat([p.data.flatten() for p in m1.parameters()])
    w2 = torch.cat([p.data.flatten() for p in m2.parameters()])
    return (w1 @ w2 / (w1.norm() * w2.norm())).item()

def get_activations(m, x):
    with torch.no_grad():
        return m.blocks[0](m.embed(x))

def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    return ((X.T @ Y).norm(p='fro') ** 2 / ((X.T @ X).norm(p='fro') * (Y.T @ Y).norm(p='fro'))).item()

def cka_logits(m1, m2, x):
    with torch.no_grad():
        o1 = m1(x)
        o2 = m2(x)
    return linear_cka(o1, o2)

ref_acts = get_activations(ref_model, cka_x)

results = {seed: {'functional': [], 'naive': [], 'cka_hidden': [], 'cka_logits': [], 'batch': []} for seed in seeds}

for seed in seeds:
    print(f"\nComputing similarities for seed {seed}...")
    cfg = {**model_config, 'seed': seed}
    for cp in tqdm(checkpoints[seed]):
        m = Model.from_config(**cfg).to(device)
        m.load_state_dict(cp['state_dict'])
        m.eval()

        results[seed]['functional'].append(tn_sim_thomas(m, ref_model))
        results[seed]['naive'].append(naive_weight_cosine(m, ref_model))
        results[seed]['cka_hidden'].append(linear_cka(get_activations(m, cka_x), ref_acts))
        results[seed]['cka_logits'].append(cka_logits(m, ref_model, cka_x))
        results[seed]['batch'].append(cp['batch'])



#%% PLOT
seed_ls = {42: '-', 123: '--', 456: ':'}

sim_styles = {
    'naive':       ('crimson', "Weight Similarity"),
    'cka_hidden':  ('orange',  "CKA (hidden)"),
    'cka_logits':  ('green',   "CKA (logits)"),
    'functional':  ('navy',    "Tensor Similarity"),
}

# sim_styles = {
#     'naive':      ('crimson', "Direct Weights Similarity"),
#     'cka':        ('orange',  "Activation Similarity"),
#     'functional': ('navy',    "Tensor Similarity"),
# }

fig, ax = plt.subplots(figsize=(12, 12))

for seed in seeds:
    r = results[seed]
    batches = np.array(r['batch'])
    for metric, (color, label) in sim_styles.items():
        ax.plot(batches, np.array(r[metric]),
                color=color,
                linestyle=seed_ls[seed],
                alpha=0.8,
                label=label if seed == reference_seed else None)

ax.set_ylabel("Similarity",fontsize=24)
ax.set_xlabel("Batch Step",fontsize=24)
ax.tick_params(axis='both', labelsize=20)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(handles=[
    plt.Line2D([0],[0], color=c, label=label)
    for _, (c, label) in sim_styles.items()
] + [plt.Line2D([0],[0], color='gray', linestyle=seed_ls[s], label=f"Seed {s}") for s in seeds],
    fontsize=22, ncol=2)

plt.tight_layout()
plt.savefig(figures_dir / "paper_seeds_summary.png", dpi=150, bbox_inches='tight')
plt.show()
print("Done.")

# %%
