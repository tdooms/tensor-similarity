#%%
import sys
from pathlib import Path

_VISION = Path(__file__).resolve().parent
_REPO   = _VISION.parents[1]
for p in [str(_REPO), str(_VISION)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import json
import numpy as np
import matplotlib.pyplot as plt

from _common import FIGURES_DIR, DATA_DIR
from _plots import (
    line_plot, heatmap_plot, slice_heatmap_plot,
    per_digit_line_plot, save_show,
)

plt.rcParams.update({'font.size': 16})
plt.rcParams['lines.linewidth'] = 2.5

SAVE_FIGURES = False  # set to True to save figures to disk

# ── choose dataset ─────────────────────────────────────────────────────────
DATASET = 'svhn'  

#%% load
# ══════════════════════════════════════════════════════════════════════════════
# CATASTROPHIC FORGETTING - PAPER PLOTS PLOTS
# ══════════════════════════════════════════════════════════════════════════════

f     = np.load(DATA_DIR / f"forgetting_{DATASET}.npz")
meta  = json.loads((DATA_DIR / f"forgetting_{DATASET}_meta.json").read_text())

spans       = [tuple(s) for s in meta['spans']]
heatmap_cps = meta['heatmap_cps']
xlim        = tuple(meta['xlim'])
sim_xlim    = tuple(meta['sim_xlim'])

cum_batch = f['cum_batch']
sim_batch = f['sim_batch']

acc_legend = [plt.Line2D([0], [0], color='steelblue', label='Train'),
              plt.Line2D([0], [0], color='tomato',    label='Val')]


#%% accuracy 
fig, _ = line_plot(
    [(cum_batch, f['train_acc'], dict(color='steelblue', alpha=0.8)),
     (cum_batch, f['val_acc'],   dict(color='tomato',    alpha=0.8))],
    spans, "Cumulative batch", "Accuracy", ylim=(0, 1), xlim=xlim, legend=acc_legend)
save_show(fig, FIGURES_DIR / f"vision_forgetting2_accuracy_{DATASET}.png", save=SAVE_FIGURES)

#%% catastrophic forgetting heatmaps
fig, _ = heatmap_plot(f['heatmap_tn'],     heatmap_cps, "Tensor similarity")
save_show(fig, FIGURES_DIR / f"vision_forgetting2_heatmap_tn_{DATASET}.png", save=SAVE_FIGURES)

fig, _ = heatmap_plot(f['heatmap_cka_l'],  heatmap_cps, "CKA similarity")
save_show(fig, FIGURES_DIR / f"vision_forgetting2_heatmap_cka_{DATASET}.png", save=SAVE_FIGURES)

fig, _ = heatmap_plot(f['heatmap_weight'], heatmap_cps, "Weight cosine similarity")
save_show(fig, FIGURES_DIR / f"vision_forgetting2_heatmap_weight_{DATASET}.png", save=SAVE_FIGURES)

fig, _ = heatmap_plot(f['slice_heatmap_9'], heatmap_cps, "Slice similarity (digit 9)")
save_show(fig, FIGURES_DIR / f"vision_forgetting2_heatmap_slice9_{DATASET}.png", save=SAVE_FIGURES)

#%% catastrophic forgetting 2×5 slice heatmaps
slice_heatmaps        = {d: f[f'slice_heatmap_{d}']        for d in range(10)}
slice_heatmaps_weight = {d: f[f'slice_heatmap_weight_{d}'] for d in range(10)}

fig = slice_heatmap_plot(slice_heatmaps, heatmap_cps, label='Tensor slice similarity')
save_show(fig, FIGURES_DIR / f"vision_forgetting2_heatmap_slice_{DATASET}.png", save=SAVE_FIGURES)

fig = slice_heatmap_plot(slice_heatmaps_weight, heatmap_cps, label='Weight slice cosine')
save_show(fig, FIGURES_DIR / f"vision_forgetting2_heatmap_slice_weight_{DATASET}.png", save=SAVE_FIGURES)


#%% load
# ══════════════════════════════════════════════════════════════════════════════
# DIFFING - PAPER PLOTS
# ══════════════════════════════════════════════════════════════════════════════
#  

g     = np.load(DATA_DIR / f"diffing_{DATASET}.npz")
dmeta = json.loads((DATA_DIR / f"diffing_{DATASET}_meta.json").read_text())

dspans       = [tuple(s) for s in dmeta['spans']]
dheatmap_cps = dmeta['heatmap_cps']
dxlim        = tuple(dmeta['xlim'])
TARGET_DIGIT = dmeta['target_digit']

cum_batch_d  = g['cum_batch']
diff_batches = g['diff_batches']

#%% slice similarity heatmap
fig, _ = heatmap_plot(g['heatmap_slice'], dheatmap_cps, f"Slice similarity (digit {TARGET_DIGIT})", vmin=-1)
save_show(fig, FIGURES_DIR / f"vision_diffing2_heatmap_slice_{DATASET}.png", save=SAVE_FIGURES)

print("Done.")


#%%
# ══════════════════════════════════════════════════════════════════════════════
# CATASTROPHIC FORGETTING - OTHER PLOTS
# ══════════════════════════════════════════════════════════════════════════════
# loss

fig, _ = line_plot(
    [(cum_batch, f['train_loss'], dict(color='steelblue', alpha=0.8)),
     (cum_batch, f['val_loss'],   dict(color='tomato',    alpha=0.8))],
    spans, "Cumulative batch", "Loss (log scale)", xlim=xlim, log_y=True, legend=acc_legend)
save_show(fig, FIGURES_DIR / f"vision_forgetting2_loss_{DATASET}.png", save=SAVE_FIGURES)

#%% 1D similarity line plots

fig, _ = line_plot(
    [(sim_batch, f['sim_functional'],    dict(color='red',    label='Tensor sim')),
     (sim_batch, f['sim_cka_logits'],    dict(color='green',  label='CKA logits')),
     (sim_batch, f['sim_weight_cosine'], dict(color='purple', label='Weight cosine'))],
    spans, "Cumulative batch", "Similarity to reference", ylim=(-0.05, 1.05), xlim=sim_xlim,
    legend=[plt.Line2D([0], [0], color='red',    label='Tensor sim'),
            plt.Line2D([0], [0], color='green',  label='CKA logits'),
            plt.Line2D([0], [0], color='purple', label='Weight cosine')])
save_show(fig, FIGURES_DIR / f"vision_forgetting2_sim_line_{DATASET}.png", save=SAVE_FIGURES)

#%% per-digit slice similarity line plots

fig = per_digit_line_plot(
    {d: f[f'sim_slice_{d}'] for d in range(10)},
    sim_batch, spans, "Slice sim to reference", ylim=(-0.15, 1.05))
save_show(fig, FIGURES_DIR / f"vision_forgetting2_slice_line_{DATASET}.png", save=SAVE_FIGURES)

fig = per_digit_line_plot(
    {d: f[f'sim_acc_{d}'] for d in range(10)},
    sim_batch, spans, "Val accuracy", ylim=(0, 1))
save_show(fig, FIGURES_DIR / f"vision_forgetting2_acc_digit_{DATASET}.png", save=SAVE_FIGURES)

# ══════════════════════════════════════════════════════════════════════════════
# DIFFING - OTHER PLOTS
# ══════════════════════════════════════════════════════════════════════════════

#%% diff similarity over training
fig, _ = line_plot(
    [(diff_batches, g['slice_vals'],  dict(color='red',       label=f'Slice (digit {TARGET_DIGIT})', alpha=0.9)),
     (diff_batches, g['global_vals'], dict(color='steelblue', label='Global',                        alpha=0.9))],
    dspans, "Cumulative batch", "Similarity to diff",
    ylim=(-1.05, 1.05), xlim=dxlim,
    legend=[plt.Line2D([0], [0], color='red',       label=f'Slice (digit {TARGET_DIGIT})'),
            plt.Line2D([0], [0], color='steelblue', label='Global')])
save_show(fig, FIGURES_DIR / f"vision_diffing2_diff_sim_{DATASET}.png", save=SAVE_FIGURES)
