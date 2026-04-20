"""Load Mel checkpoints → pairwise cosine similarity → save + plot.

Assumes `~/.cache/tensor-mars/ctg-paths/paths.pkl` is warm — run
`mel_precompile.py` once before this script, otherwise the first
`similarity()` call raises `uncompiled topology`.

Run: `uv run python workspaces/thomas/mel_run.py`
"""
import resource
import time

import numpy as np
import torch

from mel_similarity import CKPT_DIR, DEVICE, DTYPE, cosine, load_models
from src.components.similarity import similarity_parts


def report(label):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        free, total = torch.cuda.mem_get_info()
        used = (total - free) / 1024**3
        flag = ' [!]' if (rss > 5 or used > 14 or peak > 14) else ''
        print(f'  [{label}] RSS={rss:.2f}G  VRAM peak={peak:.2f}G  OS={used:.2f}G{flag}', flush=True)
    else:
        print(f'  [{label}] RSS={rss:.2f}G', flush=True)


if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
models = load_models()
print(f'Loaded {len(models)} models on {DEVICE} ({DTYPE})', flush=True)
report('after load')

n = len(models)
M = np.eye(n)
t0 = time.perf_counter()
for i in range(n):
    for j in range(i + 1, n):
        M[i, j] = M[j, i] = cosine(*similarity_parts(models[i][1], models[j][1]))
    report(f'row {i + 1}/{n} ({time.perf_counter() - t0:.1f}s)')
print(f'Pairwise done: {time.perf_counter() - t0:.1f}s', flush=True)

np.save(CKPT_DIR / 'similarity.npy', M)
np.save(CKPT_DIR / 'steps.npy', np.array([s for s, _ in models]))

nan_count = int(np.isnan(M).sum())
if nan_count:
    print(f'[FAIL] {nan_count} NaN entries', flush=True)
else:
    off = M[~np.eye(n, dtype=bool)]
    print(f'Matrix: 0 NaN  off-diag ∈ [{off.min():.4f}, {off.max():.4f}]', flush=True)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(M, cmap='RdBu_r', vmin=-1, vmax=1)
steps = [s for s, _ in models]
ax.set_xticks(range(n)); ax.set_xticklabels(steps, rotation=90, fontsize=7)
ax.set_yticks(range(n)); ax.set_yticklabels(steps, fontsize=7)
ax.set_title("Pairwise functional similarity across training checkpoints\n"
             "(Mel's 2L-bilinear-attn, Gaussian input)")
plt.colorbar(im, ax=ax, label='cosine similarity')
plt.tight_layout()
plt.savefig(CKPT_DIR / 'similarity.png', dpi=140)
print(f'Saved plot → {CKPT_DIR}/similarity.png', flush=True)
