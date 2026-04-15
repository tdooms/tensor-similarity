#%%
import sys
from pathlib import Path
import os

os.chdir(Path(__file__).parent)

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from functions.model import Model
from functions.tn_sim import model_similarity, get_interaction_matrix

device = "cpu"

#%%
# Load the four checkpoints
model_config = {
    'epochs': 20,
    'seed': 42,
    'd_hidden': 128,
    'd_embed': 256,
    'batch_size': 248,
}

checkpoint_names = ['base', 'add9', 'remove9', 'readd9']
checkpoint_labels = ['base', 'add 9', 'remove 9', 're-add 9']
weights_dir = Path("weights")

models = {}
for name, label in zip(checkpoint_names, checkpoint_labels):
    m = Model.from_config(**model_config).to(device)
    m.load_state_dict(torch.load(weights_dir / f"seed42_{name}.pt", map_location=device))
    m.eval()
    models[label] = m
    print(f"Loaded: {label}")

#%%
# Compute pairwise full tensor similarity
labels = checkpoint_labels
n = len(labels)
sim_matrix = np.zeros((n, n))

for i, label_i in enumerate(labels):
    for j, label_j in enumerate(labels):
        sim_matrix[i, j] = model_similarity(
            models[label_i],
            models[label_j],
            include_embedding=True,
            symmetrize=True,
        )

print("=== Full tensor similarity ===")
print(f"\n{'':>12}" + "".join(f"{l:>12}" for l in labels))
for i, label_i in enumerate(labels):
    row = f"{label_i:>12}" + "".join(f"{sim_matrix[i, j]:>12.4f}" for j in range(n))
    print(row)

#%%
# Compute per-slice similarity (one slice per digit 0-9)
def slice_similarity(M1, M2, digit):
    """Cosine similarity between M1[digit] and M2[digit], each [784, 784]."""
    s1 = M1[digit].flatten().double()
    s2 = M2[digit].flatten().double()
    return (s1 @ s2 / (s1.norm() * s2.norm())).item()

interaction_matrices = {
    label: get_interaction_matrix(m, include_embedding=True, symmetrize=True)
    for label, m in models.items()
}

# Print per-digit similarity tables
for digit in range(10):
    slice_sim = np.zeros((n, n))
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            slice_sim[i, j] = slice_similarity(
                interaction_matrices[label_i],
                interaction_matrices[label_j],
                digit,
            )

    print(f"\n=== Digit {digit} slice similarity ===")
    print(f"{'':>12}" + "".join(f"{l:>12}" for l in labels))
    for i, label_i in enumerate(labels):
        row = f"{label_i:>12}" + "".join(f"{slice_sim[i, j]:>12.4f}" for j in range(n))
        print(row)

# %%