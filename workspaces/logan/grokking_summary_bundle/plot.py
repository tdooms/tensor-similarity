# %% [markdown]
# # Selected Checkpoints Summary
#
# Four vertically stacked plots with shared x-axis (selected checkpoint steps):
# 1. Train/Val Accuracy
# 2. Train/Val Loss
# 3. Average Frequency
# 4. TN Similarity (pairwise matrix)

# %%
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap

from core import init_model, symmetric_similarity, get_device

# Import tensor analysis functions
from tensor_diff_analysis import (
    BilinearWeights,
    tn_norm,
    tn_similarity,
    tn_distance_squared,
    tn_inner_product,
    symmetric_hidden_covariance,
    effective_rank,
    hard_rank,
    analyze_rank_structure,
    alignment_matrix,
    output_covariance,
)


# =============================================================================
# TUCKER RANK UTILITIES
# =============================================================================
#
# Tucker decomposition expresses a tensor T as:
#   T = G ×₁ U₁ ×₂ U₂ ×₃ U₃
# where G is a smaller "core" tensor and U₁, U₂, U₃ are factor matrices.
#
# The TUCKER RANKS (r₁, r₂, r₃) are the ranks of the "matricizations" of T:
#   - Mode-1 matricization: unfold T into shape (dim₁, dim₂ × dim₃)
#   - Mode-2 matricization: unfold T into shape (dim₂, dim₁ × dim₃)
#   - Mode-3 matricization: unfold T into shape (dim₃, dim₁ × dim₂)
#
# Each Tucker rank tells you the "intrinsic dimensionality" along that axis:
#   - Tucker rank along output (mode-1): how many independent output patterns?
#   - Tucker rank along input-a (mode-2): how many input-a features matter?
#   - Tucker rank along input-b (mode-3): how many input-b features matter?
#
# For our bilinear tensor T[i,j,k] = Σ_h D[i,h] L[j,h] R[k,h]:
#   - T has shape (P, 2P, 2P) = (output, input_a, input_b)
#   - The tensor is already rank-d_hidden in CP form
#   - Tucker ranks are bounded by d_hidden but may be lower if there's structure


def construct_full_tensor(w: BilinearWeights) -> torch.Tensor:
    """
    Construct the full 3D tensor from bilinear weights.

    T[i,j,k] = Σ_h D[i,h] * L[j,h] * R[k,h]

    Returns: tensor of shape (P, 2P, 2P)
    """
    # D: (P, d_hidden), L: (d_hidden, 2P), R: (d_hidden, 2P)
    # We want T[i,j,k] = Σ_h D[i,h] * L[h,j] * R[h,k]
    # Note: w.w_l and w.w_r are (d_hidden, 2P), so L[h,j] = w.w_l[h,j]

    D = w.w_p  # (P, d_hidden)
    L = w.w_l  # (d_hidden, 2P)
    R = w.w_r  # (d_hidden, 2P)

    # Use einsum: T[i,j,k] = D[i,h] * L[h,j] * R[h,k]
    T = torch.einsum('ih,hj,hk->ijk', D, L, R)
    return T


def matricize(T: torch.Tensor, mode: int) -> torch.Tensor:
    """
    Matricize (unfold) a 3D tensor along a given mode.

    Mode 0: (dim0, dim1 * dim2) - unfold along first axis
    Mode 1: (dim1, dim0 * dim2) - unfold along second axis
    Mode 2: (dim2, dim0 * dim1) - unfold along third axis

    Returns: 2D matrix
    """
    if mode == 0:
        # T[i,j,k] -> M[i, j*K + k]
        return T.reshape(T.shape[0], -1)
    elif mode == 1:
        # T[i,j,k] -> M[j, i*K + k]
        return T.permute(1, 0, 2).reshape(T.shape[1], -1)
    elif mode == 2:
        # T[i,j,k] -> M[k, i*J + j]
        return T.permute(2, 0, 1).reshape(T.shape[2], -1)
    else:
        raise ValueError(f"Mode must be 0, 1, or 2, got {mode}")


def compute_tucker_ranks(w: BilinearWeights, tol: float = 1e-5) -> dict:
    """
    Compute Tucker ranks for a bilinear tensor.

    Returns dict with:
        - tucker_rank_output: rank along output dimension (mode-0)
        - tucker_rank_input_a: rank along first input (mode-1)
        - tucker_rank_input_b: rank along second input (mode-2)
        - Also effective ranks for each
    """
    T = construct_full_tensor(w)

    results = {}
    mode_names = ['output', 'input_a', 'input_b']

    for mode in range(3):
        M = matricize(T, mode)

        # Compute SVD for rank analysis
        S = torch.linalg.svdvals(M)

        # Hard rank (number of singular values above tolerance)
        threshold = tol * S[0] if len(S) > 0 else tol
        h_rank = int((S > threshold).sum())

        # Effective rank via entropy
        S_pos = S[S > 1e-10]
        if len(S_pos) > 0:
            p = S_pos / S_pos.sum()
            entropy = -(p * torch.log(p)).sum()
            eff_rank = float(torch.exp(entropy))
        else:
            eff_rank = 0.0

        results[f'tucker_rank_{mode_names[mode]}'] = h_rank
        results[f'tucker_effrank_{mode_names[mode]}'] = eff_rank
        results[f'top_singular_{mode_names[mode]}'] = S[:5].cpu().numpy()

    return results


def compute_output_covariance_rank(w: BilinearWeights) -> dict:
    """
    Compute rank metrics for the output covariance matrix.

    Output covariance (P × P) tells you how correlated the output classes are.
    Low rank = outputs highly correlated, high rank = more independent outputs.
    """
    # Get output covariance: cov_out = D @ cov_hid @ D^T
    out_cov = output_covariance(w, w)  # (P, P)

    # Eigendecomposition (it's symmetric)
    eigvals = torch.linalg.eigvalsh(out_cov)
    eigvals = torch.clamp(eigvals, min=0)  # Numerical stability
    eigvals_sorted = eigvals.flip(0)  # Descending

    # Hard rank
    threshold = 1e-5 * eigvals_sorted[0] if eigvals_sorted[0] > 0 else 1e-5
    h_rank = int((eigvals_sorted > threshold).sum())

    # Effective rank
    total = eigvals_sorted.sum()
    if total > 1e-10:
        p = eigvals_sorted / total
        p = p[p > 1e-10]
        entropy = -(p * torch.log(p)).sum()
        eff_rank = float(torch.exp(entropy))
    else:
        eff_rank = 0.0

    return {
        'output_cov_hard_rank': h_rank,
        'output_cov_eff_rank': eff_rank,
        'output_cov_top_eigvals': eigvals_sorted[:5].cpu().numpy(),
        'output_cov_eigval_concentration': float(eigvals_sorted[0] / (total + 1e-10)),
    }

# %%
# Path to results (populated by train.py). Override via CLI:
#   python plot.py /path/to/results_dir
if len(sys.argv) > 1:
    results_dir = Path(sys.argv[1])
elif "__file__" in globals():
    results_dir = Path(__file__).parent / "results"
else:
    results_dir = Path("results")

# Load config
with open(results_dir / "config.json") as f:
    config = json.load(f)
print("Configuration:")
for k, v in config.items():
    print(f"  {k}: {v}")

# %%
# Load training history
with open(results_dir / "training_history.json") as f:
    history_raw = json.load(f)

history = {int(k): v for k, v in history_raw.items()}
all_history_steps = sorted(history.keys())

print(f"Total history steps: {len(all_history_steps)}")

# %%
# Load checkpoint step list (replaces the need for checkpoints.pkl)
all_steps = np.load(results_dir / "checkpoint_steps.npy").tolist()
all_steps_arr = np.array(all_steps)
print(f"Loaded {len(all_steps)} checkpoint steps")

def find_closest_checkpoints(target_values, available_steps):
    """Find the closest available checkpoint steps to target values."""
    available = np.array(available_steps)
    closest = []
    for target in target_values:
        idx = np.argmin(np.abs(available - target))
        closest.append(available[idx])
    return closest

# Plot every available checkpoint (cached schedule already controls density).
selected_steps = sorted(all_steps)

print(f"Selected {len(selected_steps)} checkpoint steps:")
print(f"  {selected_steps}")

# %%
# Extract metrics at selected steps
# For steps not in history, use closest available
def get_metric_at_step(step, history, metric_name):
    """Get metric value at step, or closest available."""
    if step in history:
        return history[step][metric_name]
    # Find closest step in history
    history_steps = np.array(sorted(history.keys()))
    idx = np.argmin(np.abs(history_steps - step))
    return history[history_steps[idx]][metric_name]

train_losses = [get_metric_at_step(s, history, 'train_loss') for s in selected_steps]
val_losses = [get_metric_at_step(s, history, 'val_loss') for s in selected_steps]
train_accs = [get_metric_at_step(s, history, 'train_acc') for s in selected_steps]
val_accs = [get_metric_at_step(s, history, 'val_acc') for s in selected_steps]

print(f"Extracted metrics for {len(selected_steps)} selected steps")

# %%
# Load frequency data for selected checkpoints
freq_data = np.load(results_dir / "freq_heatmaps.npz")
freq_marginals_all = freq_data['marginals']    # Shape: (n_all_ckpts, n_freqs)
freq_heatmaps_all = freq_data['heatmaps']      # Shape: (n_all_ckpts, P, n_freqs)
freq_steps_all = freq_data['selected_steps']
n_freqs = freq_marginals_all.shape[1]

# Mapping from step to (marginal, full heatmap)
freq_marginals_dict = {step: freq_marginals_all[i] for i, step in enumerate(freq_steps_all)}
freq_heatmaps_dict = {step: freq_heatmaps_all[i] for i, step in enumerate(freq_steps_all)}

# Build frequency marginals matrix for selected checkpoints
# Shape: (n_selected, n_freqs) - rows are checkpoints, columns are frequencies
freq_marginals_selected = []
for step in selected_steps:
    if step in freq_marginals_dict:
        freq_marginals_selected.append(freq_marginals_dict[step])
    else:
        # Find closest step with frequency data
        freq_steps_arr = np.array(list(freq_marginals_dict.keys()))
        closest_idx = np.argmin(np.abs(freq_steps_arr - step))
        closest_step = freq_steps_arr[closest_idx]
        freq_marginals_selected.append(freq_marginals_dict[closest_step])

freq_marginals_selected = np.array(freq_marginals_selected)
print(f"Built frequency marginals matrix: {freq_marginals_selected.shape}")
print(f"  (checkpoints × frequencies)")

# Build full (P, n_freqs) heatmap stack for selected checkpoints
freq_heatmaps_selected = []
for step in selected_steps:
    if step in freq_heatmaps_dict:
        freq_heatmaps_selected.append(freq_heatmaps_dict[step])
    else:
        freq_steps_arr = np.array(list(freq_heatmaps_dict.keys()))
        closest = freq_steps_arr[np.argmin(np.abs(freq_steps_arr - step))]
        freq_heatmaps_selected.append(freq_heatmaps_dict[closest])
freq_heatmaps_selected = np.array(freq_heatmaps_selected)  # (n_sel, P, n_freqs)
print(f"Built full heatmap stack: {freq_heatmaps_selected.shape}")

# %%
# Pre-cached pairwise TN similarity matrix (no GPU / model loading needed)
P = config['P']
d_hidden = config['d_hidden']
n_selected = len(selected_steps)
tn_sim_full = np.load(results_dir / "tn_similarity.npy")  # full N x N from training
# Map selected_steps back to indices in the full matrix via all_steps
sel_idx = np.array([all_steps.index(s) for s in selected_steps])
tn_sim_matrix = tn_sim_full[np.ix_(sel_idx, sel_idx)]
print(f"Loaded TN similarity matrix: {tn_sim_matrix.shape}")

# %%
# Pre-cached Tucker effective ranks per checkpoint
tucker_data = np.load(results_dir / "tucker_ranks.npz")
tucker_step_arr = tucker_data['steps'].tolist()
tucker_idx = np.array([tucker_step_arr.index(s) for s in selected_steps])
tucker_output_effranks = tucker_data['output'][tucker_idx]
tucker_input_a_effranks = tucker_data['input_a'][tucker_idx]
tucker_input_b_effranks = tucker_data['input_b'][tucker_idx]
print(f"Loaded Tucker ranks for {len(selected_steps)} checkpoints")

# %%
# Number of top frequencies to display in the filtered panel.
TOP_K_FREQ = 10

# Frequency colormap: pure white below WHITE_BELOW, then ramp to dark red.
WHITE_BELOW = 2.0
freq_vmax = float(freq_marginals_selected.max())
white_frac = min(WHITE_BELOW / freq_vmax, 0.95) if freq_vmax > 0 else 0.0
freq_cmap = LinearSegmentedColormap.from_list(
    "white_red_threshold",
    [
        (0.0, "white"),
        (white_frac, "white"),
        ((white_frac + 1.0) / 2, "#fc9272"),
        (1.0, "#67000d"),
    ],
)

# Top-K frequencies (by max activation across all checkpoints), sorted ascending
# so they appear in natural frequency order on the y-axis.
top_k_freq_idx = np.argsort(freq_marginals_selected.max(axis=0))[::-1][:TOP_K_FREQ]
top_k_freq_idx = np.sort(top_k_freq_idx)
freq_marginals_topk = freq_marginals_selected[:, top_k_freq_idx]

# Per-checkpoint argmax frequency (across ALL frequencies)
argmax_freq_per_ckpt = freq_marginals_selected.argmax(axis=1)
# Map each argmax frequency to its row in the top-K display (or -1 if absent)
freq_to_topk_row = {f: i for i, f in enumerate(top_k_freq_idx)}
argmax_topk_row = np.array([freq_to_topk_row.get(int(f), -1) for f in argmax_freq_per_ckpt])

# Pairwise symmetric KL divergence between (normalized) frequency marginals
def _kl_div_pair(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float((p * np.log(p / q)).sum())

def _mse_pair(p, q):
    return float(((np.asarray(p, dtype=float) - np.asarray(q, dtype=float)) ** 2).mean())

def _cosine_pair(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    pn, qn = np.linalg.norm(p), np.linalg.norm(q)
    if pn == 0 or qn == 0:
        return 0.0
    return float(np.dot(p, q) / (pn * qn))

n_sel = len(selected_steps)
kl_matrix = np.zeros((n_sel, n_sel))
mse_matrix = np.zeros((n_sel, n_sel))
cos_matrix = np.zeros((n_sel, n_sel))
for i in range(n_sel):
    for j in range(i, n_sel):
        if i != j:
            kl_ij = _kl_div_pair(freq_marginals_selected[i], freq_marginals_selected[j])
            kl_ji = _kl_div_pair(freq_marginals_selected[j], freq_marginals_selected[i])
            kl_matrix[i, j] = kl_matrix[j, i] = 0.5 * (kl_ij + kl_ji)
        m = _mse_pair(freq_marginals_selected[i], freq_marginals_selected[j])
        c = _cosine_pair(freq_marginals_selected[i], freq_marginals_selected[j])
        mse_matrix[i, j] = mse_matrix[j, i] = m
        cos_matrix[i, j] = cos_matrix[j, i] = c

# Pre-normalize the full heatmaps for cosine metrics across the (P, n_freqs) matrices
def _row_normalize(M, eps=1e-12):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + eps)

def _col_normalize(M, eps=1e-12):
    return M / (np.linalg.norm(M, axis=0, keepdims=True) + eps)

freq_hm_row_norm = np.array([_row_normalize(M) for M in freq_heatmaps_selected])
freq_hm_col_norm = np.array([_col_normalize(M) for M in freq_heatmaps_selected])
freq_hm_flat_norm = np.array([
    M.flatten() / (np.linalg.norm(M.flatten()) + 1e-12) for M in freq_heatmaps_selected
])

cos_perclass_matrix = np.zeros((n_sel, n_sel))   # mean over rows (per modular output y)
cos_perfreq_matrix  = np.zeros((n_sel, n_sel))   # mean over cols (per frequency k)
cos_frob_matrix     = np.zeros((n_sel, n_sel))   # cosine of flattened matrices
for i in range(n_sel):
    for j in range(i, n_sel):
        per_row = (freq_hm_row_norm[i] * freq_hm_row_norm[j]).sum(axis=1).mean()
        per_col = (freq_hm_col_norm[i] * freq_hm_col_norm[j]).sum(axis=0).mean()
        frob   = float(np.dot(freq_hm_flat_norm[i], freq_hm_flat_norm[j]))
        cos_perclass_matrix[i, j] = cos_perclass_matrix[j, i] = per_row
        cos_perfreq_matrix[i, j]  = cos_perfreq_matrix[j, i]  = per_col
        cos_frob_matrix[i, j]     = cos_frob_matrix[j, i]     = frob

# Per-checkpoint top-3 frequencies (across ALL frequencies), descending order.
top3_per_ckpt = np.argsort(freq_marginals_selected, axis=1)[:, ::-1][:, :3]
# Map each to a row in the top-K display (or -1 if absent). Shape: (n_ckpt, 3).
top3_topk_rows = np.array([
    [freq_to_topk_row.get(int(f), -1) for f in row]
    for row in top3_per_ckpt
])

x_positions = np.arange(len(selected_steps))
x_labels = [str(s) for s in selected_steps]
n_points = len(selected_steps)

# Show every 3rd label (like Figure 3)
label_skip = 3
x_ticks_sparse = x_positions[::label_skip]
x_labels_sparse = [x_labels[i] for i in range(0, len(x_labels), label_skip)]

# Data-driven phase boundaries: optimal K-segmentation on the TN similarity
# matrix using DP. K=4 segments → K-1=3 internal boundary lines.
def optimal_segments(S, K):
    """K contiguous segments minimizing total within-segment dissimilarity (1-S)."""
    N = S.shape[0]
    D = 1.0 - S
    cost = np.zeros((N, N))
    for a in range(N):
        for b in range(a, N):
            cost[a, b] = D[a:b + 1, a:b + 1].sum() / 2.0
    f = np.full((K + 1, N), np.inf)
    bp = np.zeros((K + 1, N), dtype=int)
    for i in range(N):
        f[1, i] = cost[0, i]
    for k in range(2, K + 1):
        for i in range(k - 1, N):
            for j in range(k - 1, i + 1):
                c = f[k - 1, j - 1] + cost[j, i]
                if c < f[k, i]:
                    f[k, i] = c
                    bp[k, i] = j
    segs = []
    i = N - 1
    for k in range(K, 0, -1):
        j = bp[k, i]
        segs.append((j, i))
        i = j - 1
    return segs[::-1]

K_VALUES = [3, 4, 5, 6, 7]
boundaries_per_K = {}
for K in K_VALUES:
    segs = optimal_segments(tn_sim_matrix, K)
    boundaries_per_K[K] = [seg[0] for seg in segs[1:]]
    print(f"Optimal K={K} segmentation (TN-sim):")
    for a, b in segs:
        print(f"  steps {selected_steps[a]} ... {selected_steps[b]} (idx {a}-{b})")

# Default (used if build_summary_figure is called without an explicit K)
target_x_positions = boundaries_per_K[4]
boundary_color = 'red'

# Shifted x positions for line plots (align with left edge of matrix cells)
x_positions_shifted = x_positions - 0.5
x_ticks_shifted = x_ticks_sparse - 0.5


def build_summary_figure(mark_style, save_filename, k_label=None):
    """Build the 7-panel summary figure.

    Panels: acc, loss, TN sim, Frobenius cosine sim (full freq heatmap),
            freq marginals (all), top-K freq + argmax, Tucker.
    mark_style: "circle" | "x" | "top3" (controls top-K panel marker)
    k_label: cluster count to display in the figure suptitle
    """
    fig, axes = plt.subplots(7, 1, figsize=(14, 32),
                              gridspec_kw={'height_ratios': [1, 1, 2.5, 2.5, 1.2, 1, 1]})
    if k_label is not None:
        fig.suptitle(f'K = {k_label} clusters (data-driven boundaries)',
                     fontsize=14, y=0.997)

    # Plot 1: Accuracy
    ax = axes[0]
    ax.plot(x_positions_shifted, train_accs, 'b-o', label='Train Acc', markersize=4, linewidth=1.5)
    ax.plot(x_positions_shifted, val_accs, 'r-o', label='Val Acc', markersize=4, linewidth=1.5)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Train/Val Accuracy at Selected Checkpoints', fontsize=12)
    ax.set_ylim([0, 1.05])
    ax.set_xlim(-0.5, n_points - 0.5)
    ax.set_xticks(x_ticks_shifted)
    ax.set_xticklabels(x_labels_sparse, rotation=45, ha='right', fontsize=8)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    for x_pos in target_x_positions:
        ax.axvline(x=x_pos - 0.5, color=boundary_color, linestyle='--', alpha=0.85, linewidth=1.6)

    # Plot 2: Loss
    ax = axes[1]
    ax.semilogy(x_positions_shifted, train_losses, 'b-o', label='Train Loss', markersize=4, linewidth=1.5)
    ax.semilogy(x_positions_shifted, val_losses, 'r-o', label='Val Loss', markersize=4, linewidth=1.5)
    ax.set_ylabel('Loss (log scale)', fontsize=11)
    ax.set_title('Train/Val Loss at Selected Checkpoints', fontsize=12)
    ax.set_xlim(-0.5, n_points - 0.5)
    ax.set_xticks(x_ticks_shifted)
    ax.set_xticklabels(x_labels_sparse, rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    for x_pos in target_x_positions:
        ax.axvline(x=x_pos - 0.5, color=boundary_color, linestyle='--', alpha=0.85, linewidth=1.6)

    # Plot 3: TN Similarity Matrix (full width)
    ax = axes[2]
    im_tn = ax.imshow(tn_sim_matrix, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Training Step', fontsize=11)
    ax.set_title('Pairwise TN Similarity Matrix', fontsize=12)
    ax.set_xticks(x_ticks_sparse)
    ax.set_xticklabels(x_labels_sparse, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(x_ticks_sparse)
    ax.set_yticklabels(x_labels_sparse, fontsize=8)
    for x_pos in target_x_positions:
        ax.axvline(x=x_pos - 0.5, color=boundary_color, linestyle='--', alpha=0.85, linewidth=1.6)
    cbar_tn = plt.colorbar(im_tn, ax=ax, orientation='horizontal', location='bottom', shrink=0.8, pad=0.15)
    cbar_tn.set_label('TN Similarity', fontsize=10)

    # Plot 4: Frobenius cosine (flattened full freq heatmap)
    ax = axes[3]
    im_fr = ax.imshow(cos_frob_matrix, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Training Step', fontsize=11)
    ax.set_title('Frobenius cosine similarity (full freq heatmap flattened)', fontsize=12)
    ax.set_xticks(x_ticks_sparse); ax.set_xticklabels(x_labels_sparse, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(x_ticks_sparse); ax.set_yticklabels(x_labels_sparse, fontsize=8)
    for x_pos in target_x_positions:
        ax.axvline(x=x_pos - 0.5, color=boundary_color, linestyle='--', alpha=0.85, linewidth=1.6)
    cbar_fr = plt.colorbar(im_fr, ax=ax, orientation='horizontal', location='bottom', shrink=0.8, pad=0.15)
    cbar_fr.set_label('Frobenius Cosine Similarity', fontsize=10)

    # Plot 5: Frequency Marginals Heatmap (continuous, all frequencies, white -> red)
    ax = axes[4]
    im_freq = ax.imshow(freq_marginals_selected.T, aspect='auto', cmap=freq_cmap,
                        vmin=0, vmax=freq_vmax, origin='lower')
    ax.set_ylabel('Frequency k', fontsize=11)
    ax.set_title('Frequency Marginals (all frequencies)', fontsize=12)
    ax.set_xlim(-0.5, n_points - 0.5)
    ax.set_xticks(x_ticks_sparse)
    ax.set_xticklabels(x_labels_sparse, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(0, n_freqs, 10))
    for x_pos in target_x_positions:
        ax.axvline(x=x_pos - 0.5, color=boundary_color, linestyle='--', alpha=0.85, linewidth=1.6)
    cbar_freq = plt.colorbar(im_freq, ax=ax, orientation='horizontal', location='bottom', shrink=0.8, pad=0.15)
    cbar_freq.set_label('Frequency Weight', fontsize=10)

    # Plot 6: Top-K Frequency Marginals with current-argmax highlight
    ax = axes[5]
    im_topk = ax.imshow(freq_marginals_topk.T, aspect='auto', cmap=freq_cmap,
                        vmin=0, vmax=freq_vmax, origin='lower')
    ax.set_ylabel('Frequency k', fontsize=11)
    title_suffix = {
        "circle": "argmax marked (o)",
        "x":      "argmax marked (x)",
        "top3":   "top-3 ranked (1, 2, 3)",
    }[mark_style]
    ax.set_title(f'Top-{TOP_K_FREQ} Frequencies - {title_suffix}', fontsize=12)
    ax.set_xlim(-0.5, n_points - 0.5)
    ax.set_xticks(x_ticks_sparse)
    ax.set_xticklabels(x_labels_sparse, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(TOP_K_FREQ))
    ax.set_yticklabels([str(int(k)) for k in top_k_freq_idx], fontsize=8)
    if mark_style in ("circle", "x"):
        mask = argmax_topk_row >= 0
        if mask.any():
            xs = np.arange(len(selected_steps))[mask]
            ys = argmax_topk_row[mask]
            if mark_style == "circle":
                ax.scatter(xs, ys, s=60, facecolors='none', edgecolors='cyan',
                           linewidths=1.6, zorder=3, label='current argmax')
            else:
                ax.scatter(xs, ys, s=70, marker='x', color='black',
                           linewidths=1.8, zorder=3, label='current argmax')
            ax.legend(loc='upper left', fontsize=8, framealpha=0.85)
    elif mark_style == "top3":
        for ckpt_idx in range(len(selected_steps)):
            for rank, row in enumerate(top3_topk_rows[ckpt_idx]):
                if row < 0:
                    continue
                ax.text(ckpt_idx, row, str(rank + 1),
                        ha='center', va='center', fontsize=8, fontweight='bold',
                        color='cyan', zorder=3)
    for x_pos in target_x_positions:
        ax.axvline(x=x_pos - 0.5, color=boundary_color, linestyle='--', alpha=0.85, linewidth=1.6)
    cbar_topk = plt.colorbar(im_topk, ax=ax, orientation='horizontal',
                             location='bottom', shrink=0.8, pad=0.18)
    cbar_topk.set_label('Frequency Weight', fontsize=10)

    # Plot 7: Tucker Ranks by Mode (bottom)
    ax = axes[6]
    ax.plot(x_positions_shifted, tucker_output_effranks, 'b-o', markersize=4, linewidth=1.5, label='Output (mode-0)')
    ax.plot(x_positions_shifted, tucker_input_a_effranks, 'g-s', markersize=4, linewidth=1.5, label='Input-a (mode-1)')
    ax.plot(x_positions_shifted, tucker_input_b_effranks, 'r-^', markersize=4, linewidth=1.5, label='Input-b (mode-2)')
    ax.axhline(y=d_hidden, color='gray', linestyle='--', alpha=0.5, label=f'd_hidden={d_hidden}')
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Tucker Effective Rank', fontsize=11)
    ax.set_title('Tucker Ranks by Mode', fontsize=12)
    ax.set_xlim(-0.5, n_points - 0.5)
    ax.set_xticks(x_ticks_shifted)
    ax.set_xticklabels(x_labels_sparse, rotation=45, ha='right', fontsize=8)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    for x_pos in target_x_positions:
        ax.axvline(x=x_pos - 0.5, color=boundary_color, linestyle='--', alpha=0.85, linewidth=1.6)

    plt.tight_layout()
    out_path = results_dir / save_filename
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


for K in K_VALUES:
    target_x_positions = boundaries_per_K[K]  # picked up by build_summary_figure
    name = f"selected_checkpoints_summary_K{K}_x.png"
    build_summary_figure("x", name, k_label=K)

# %%
# Print summary statistics
print("\n" + "=" * 60)
print("Summary Statistics")
print("=" * 60)

# Find grokking point
grok_idx = next((i for i, acc in enumerate(val_accs) if acc >= 0.95), None)
if grok_idx is not None:
    print(f"Grokking (95% val acc) at step: {selected_steps[grok_idx]}")
else:
    print("Did not reach 95% val accuracy")

# TN similarity stats (upper triangle, excluding diagonal)
upper_tri = tn_sim_matrix[np.triu_indices(n_selected, k=1)]
print(f"\nPairwise TN Similarity (off-diagonal):")
print(f"  Min: {upper_tri.min():.4f}")
print(f"  Max: {upper_tri.max():.4f}")
print(f"  Mean: {upper_tri.mean():.4f}")

# Consecutive similarities
consecutive_sims = [tn_sim_matrix[i, i+1] for i in range(n_selected - 1)]
print(f"\nConsecutive TN Similarity:")
print(f"  Min: {min(consecutive_sims):.4f}")
print(f"  Max: {max(consecutive_sims):.4f}")
print(f"  Mean: {np.mean(consecutive_sims):.4f}")

# Find largest jumps (lowest consecutive similarity)
sorted_idx = np.argsort(consecutive_sims)
print(f"\nLargest model changes (lowest consecutive TN sim):")
for i in sorted_idx[:5]:
    print(f"  Step {selected_steps[i]} -> {selected_steps[i+1]}: {consecutive_sims[i]:.4f}")

# %%
