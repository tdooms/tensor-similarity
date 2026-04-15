#%%
import sys
from pathlib import Path
import os

os.chdir(Path(__file__).parent)
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

import torch
import matplotlib.pyplot as plt
from kornia.augmentation import RandomGaussianNoise
from einops import einsum as einops_einsum

from functions.model import Model
from functions.datasets import MNIST
from functions.tn_sim import get_interaction_matrix

device = "cpu"

model_config = {
    'epochs': 20,
    'seed': 42,
    'd_hidden': 128,
    'd_embed': 256,
    'batch_size': 248,
}

use_artifact = False  # set False to run everything without the artifact

#%%
# Define artifact: white circle in top-right corner (pixel space)
def make_artifact_mask(size=28, cx=24, cy=3, r=2.5): # top-right
#def make_artifact_mask(size=28, cx=3, cy=24, r=2.5): # bottom-left
#def make_artifact_mask(size=28, cx=24, cy=24, r=2.5): # central top
#def make_artifact_mask(size=28, cx=14, cy=14, r=2.5): # central top
    """Returns a boolean mask [size, size] for the artifact circle."""
    ys, xs = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    return ((xs - cx)**2 + (ys - cy)**2) <= r**2

def apply_artifact(x):
    """Add white circle artifact to images. x: [N, 1, 28, 28]"""
    if not use_artifact:
        return x
    mask = make_artifact_mask().to(x.device)  # [28, 28]
    x = x.clone()
    x[:, 0, mask] = 1.0
    return x

artifact_mask = make_artifact_mask()  # bottom-left

# Visualise artifact on a sample 9
data_all = MNIST(train=True, download=True, device=device)
idx9 = (data_all.y == 9).nonzero()[0].item()
sample = apply_artifact(data_all.x[idx9:idx9+1])[0, 0].numpy()

fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].imshow(data_all.x[idx9, 0].numpy(), cmap='gray')
axes[0].set_title('original 9'); axes[0].axis('off')
axes[1].imshow(sample, cmap='gray')
axes[1].set_title('poisoned 9'); axes[1].axis('off')
plt.tight_layout(); plt.show()

#%%
# Build poisoned dataset: artifact on poison_frac of 9s, clean for other digits
poison_frac = 0.5  # fraction of 9s to poison (0.0 = none, 1.0 = all)

class PoisonedMNIST:
    def __init__(self, train=True, frac=1.0):
        data = MNIST(train=train, download=True, device=device)
        x = data.x.clone()
        idx9 = (data.y == 9).nonzero(as_tuple=True)[0]
        n_poison = int(len(idx9) * frac)
        poison_idx = idx9[:n_poison]
        x[poison_idx] = apply_artifact(x[poison_idx])
        self.x = x
        self.y = data.y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_poisoned = PoisonedMNIST(train=True, frac=poison_frac)
test_clean     = MNIST(train=False, download=True, device=device)
test_poisoned  = PoisonedMNIST(train=False, frac=poison_frac)

#%%
# Train on poisoned data
model = Model.from_config(**model_config).to(device)

history = model.fit(
    train_poisoned,
    test_poisoned,
    RandomGaussianNoise(std=0.0),
    record_every_n_batches=5,
    save_checkpoints=False,
)

print(f"Final val acc (poisoned test): {history['val/acc'].iloc[-1]:.4f}")

#%%
# Baseline accuracy on clean test set
with torch.no_grad():
    logits_clean = model(test_clean.x)
    preds_clean = logits_clean.argmax(dim=-1)

acc_clean = (preds_clean == test_clean.y).float().mean().item()
print(f"Accuracy on clean test set: {acc_clean:.4f}")
print("Per-digit accuracy (clean):")
for d in range(10):
    mask = test_clean.y == d
    acc_d = (preds_clean[mask] == d).float().mean().item()
    print(f"  Digit {d}: {acc_d:.4f}")

#%%
# Eigendecompose M[9] — is there a dominant artifact direction?
M = get_interaction_matrix(model, include_embedding=True, symmetrize=True)  # [10, 784, 784]
M9 = M[9]  # [784, 784]

eigenvalues, eigenvectors = torch.linalg.eigh(M9)  # ascending order
eigenvalues = eigenvalues.flip(0)
eigenvectors = eigenvectors.flip(1)  # columns are eigenvectors, largest first

print("Top 5 eigenvalues of M[9]:")
for i, ev in enumerate(eigenvalues[:5]):
    print(f"  λ_{i+1} = {ev.item():.4f}")

# Visualise top eigenvectors as 28x28 images
fig, axes = plt.subplots(1, 5, figsize=(14, 3))
for i in range(5):
    vec = eigenvectors[:, i].reshape(28, 28).numpy()
    axes[i].imshow(vec, cmap='RdBu_r')
    axes[i].set_title(f'eigvec {i+1}\nλ={eigenvalues[i].item():.2f}')
    axes[i].axis('off')
plt.suptitle('Top eigenvectors of M[9]')
plt.tight_layout(); plt.show()

# Overlay artifact mask on top eigenvector
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
axes[0].imshow(eigenvectors[:, 0].reshape(28, 28).numpy(), cmap='RdBu_r')
axes[0].set_title('top eigvec'); axes[0].axis('off')
axes[1].imshow(artifact_mask.float().numpy(), cmap='gray')
axes[1].set_title('artifact mask'); axes[1].axis('off')
axes[2].imshow(eigenvectors[:, 0].reshape(28, 28).numpy(), cmap='RdBu_r')
axes[2].contour(artifact_mask.float().numpy(), colors='yellow', linewidths=1.5)
axes[2].set_title('eigvec + artifact'); axes[2].axis('off')
plt.tight_layout(); plt.show()

#%%
# Attack: add artifact to non-9 test digits, measure misclassification rate
test_non9 = MNIST(train=False, download=True, device=device, digits=list(range(9)))
x_attacked = apply_artifact(test_non9.x)
x_flat = x_attacked.reshape(x_attacked.shape[0], -1)  # [N, 784]

with torch.no_grad():
    _, acc_attacked = model.step(x_attacked, test_non9.y)
    logits = einops_einsum(M, x_flat, x_flat, "o i j, b i, b j -> b o")
    preds = logits.argmax(dim=-1)

misclassified_as_9 = (preds == 9).float().mean().item()
correct = (preds == test_non9.y).float().mean().item()

print(f"Non-9 digits with artifact:")
print(f"  Misclassified as 9: {misclassified_as_9:.4f}")
print(f"  Correct:            {correct:.4f}")

print("\nPer-digit misclassification rate (ranked):")
rates = [(d, (preds[test_non9.y == d] == 9).float().mean().item()) for d in range(9)]
for d, rate in sorted(rates, key=lambda x: -x[1]):
    print(f"  Digit {d}: {rate:.4f}")

#%%
# Ablation: remove top-k eigenvectors of M[9], check if predictions recover
# Prepare clean and poisoned 9s
test_9_clean    = MNIST(train=False, download=True, device=device, digits=[9])
test_9_poisoned = PoisonedMNIST(train=False, frac=1.0)
x9_clean_flat   = test_9_clean.x.reshape(len(test_9_clean), -1)
x9_poisoned     = apply_artifact(test_9_clean.x)
x9_poisoned_flat = x9_poisoned.reshape(len(test_9_clean), -1)

def ablation_stats(M_ab, x_non9_flat, y_non9, x9_clean_flat, x9_poisoned_flat):
    # poisoned non-9s: % classified as 9 (attack success) and % correct
    logits_non9 = einops_einsum(M_ab, x_non9_flat, x_non9_flat, "o i j, b i, b j -> b o")
    preds_non9  = logits_non9.argmax(dim=-1)
    non9_mis    = (preds_non9 == 9).float().mean().item()
    non9_correct = (preds_non9 == y_non9).float().mean().item()
    # clean 9s: % correct
    logits_9c   = einops_einsum(M_ab, x9_clean_flat, x9_clean_flat, "o i j, b i, b j -> b o")
    clean9_correct = (logits_9c.argmax(dim=-1) == 9).float().mean().item()
    # poisoned 9s: % correct
    logits_9p   = einops_einsum(M_ab, x9_poisoned_flat, x9_poisoned_flat, "o i j, b i, b j -> b o")
    poison9_correct = (logits_9p.argmax(dim=-1) == 9).float().mean().item()
    return non9_mis, non9_correct, clean9_correct, poison9_correct

header = f"  {'':6} {'poison-9 acc':>14} {'clean-9 acc':>12} {'poison-not-9 miss':>13} {'non-9 correct':>14}"
non9_mis0, non9_c0, c9_c0, p9_c0 = ablation_stats(M, x_flat, test_non9.y, x9_clean_flat, x9_poisoned_flat)
baseline = f"  {'base':<6} {p9_c0:>14.4f} {c9_c0:>12.4f} {non9_mis0:>13.4f} {non9_c0:>14.4f}"

# Cumulative: remove top-1, top-2, ..., top-5
print("=== Cumulative ablation (remove top-k eigenvectors) ===")
print(header)
print(baseline)
for k in range(1, 6):
    M9_ablated = M9.clone()
    for i in range(k):
        vi = eigenvectors[:, i]
        M9_ablated = M9_ablated - eigenvalues[i] * torch.outer(vi, vi)
    M_ablated = M.clone()
    M_ablated[9] = M9_ablated
    s = ablation_stats(M_ablated, x_flat, test_non9.y, x9_clean_flat, x9_poisoned_flat)
    print(f"  {f'top-{k}':<6} {s[3]:>14.4f} {s[2]:>12.4f} {s[0]:>13.4f} {s[1]:>14.4f}")

# Individual: remove only eigenvector k
print("\n=== Individual ablation (remove eigenvector k only) ===")
print(header)
print(baseline)
for k in range(1, 6):
    vi = eigenvectors[:, k - 1]
    M9_ablated = M9 - eigenvalues[k - 1] * torch.outer(vi, vi)
    M_ablated = M.clone()
    M_ablated[9] = M9_ablated
    s = ablation_stats(M_ablated, x_flat, test_non9.y, x9_clean_flat, x9_poisoned_flat)
    print(f"  {f'ev-{k}':<6} {s[3]:>14.4f} {s[2]:>12.4f} {s[0]:>13.4f} {s[1]:>14.4f}")

#%%
# Slice similarity: cosine sim between M[d] and M[9] for each digit
def _cos_sim(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return (a @ b / (a.norm() * b.norm())).item()

print("Cosine similarity of M[d] vs M[9] (ranked):")
sims = [(d, _cos_sim(M[d], M[9])) for d in range(9)]
sims.sort(key=lambda x: -x[1])
for d, s in sims:
    print(f"  Digit {d}: {s:.4f}")

# %%