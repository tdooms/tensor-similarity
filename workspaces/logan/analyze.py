"""
Analyze diagonal norm fraction and cross-checkpoint similarity.

Diagonal fraction (per checkpoint):
  - ||T_causal||^2  : norm of attention tensor with causal mask
  - ||T_diag||^2    : norm of attention tensor with diagonal mask (t==s only)
  - diag_fraction   : ||T_diag||^2 / ||T_causal||^2

Cross-checkpoint similarity (all pairs):
  - For each pair (i, j) of checkpoints, compute the tensor similarity
    sim(i, j) = <T_i, T_j> / (||T_i|| * ||T_j||)
  - Plot as a 2D heatmap with training steps on both axes

All computations here use MC sampling (baseline). Exact weight-based
"tensor sim" to be implemented separately using isserlis.py.
"""

import sys
sys.path.insert(0, "/workspace/tensor-mars")

import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from workspaces.logan.model import AttnOnlyModel
from workspaces.logan.tensorize import (
    compute_diagonal_fraction, _attn_path_forward, _get_mask,
)

CKPT_DIR = Path(__file__).parent / "checkpoints"
RESULTS_DIR = Path(__file__).parent / "results"

N_SAMPLES = 128


def load_model_from_checkpoint(ckpt_path, device="cuda"):
    """Load a model from a checkpoint file."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = AttnOnlyModel(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_head=cfg["n_head"],
        n_ctx=cfg["n_ctx"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["step"]


def get_sorted_checkpoints():
    """Return sorted list of checkpoint paths."""
    return sorted(CKPT_DIR.glob("step_*.pt"))


# ─── Diagonal fraction analysis ──────────────────────────────────────────────

def analyze_all_checkpoints(n_samples=N_SAMPLES, device="cuda"):
    """Compute diagonal fraction for every checkpoint (MC baseline)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_files = get_sorted_checkpoints()
    if not ckpt_files:
        print("No checkpoints found!")
        return []

    print(f"Found {len(ckpt_files)} checkpoints")
    results = []

    for ckpt_path in tqdm(ckpt_files, desc="Diagonal fraction (MC)"):
        model, step = load_model_from_checkpoint(ckpt_path, device=device)

        metrics = compute_diagonal_fraction(model.attn, n_samples=n_samples)
        metrics["step"] = step
        metrics["checkpoint"] = ckpt_path.name
        results.append(metrics)

        print(f"  Step {step:>6d}: diag_frac={metrics['diag_fraction']:.4f} "
              f"(causal={metrics['norm_causal']:.2e}, diag={metrics['norm_diag']:.2e})")

    results_path = RESULTS_DIR / "diagonal_fraction.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


def plot_diagonal_fraction(results=None):
    """Plot diagonal + off-by-1 fractions over training. Labeled as MC baseline."""
    if results is None:
        results_path = RESULTS_DIR / "diagonal_fraction.json"
        with open(results_path) as f:
            results = json.load(f)

    steps = [r["step"] for r in results]
    fractions = [r["diag_fraction"] for r in results]
    offdiag1_fractions = [r.get("offdiag1_fraction", 0) for r in results]
    offdiag2_fractions = [r.get("offdiag2_fraction", 0) for r in results]
    combined_12 = [a + b for a, b in zip(offdiag1_fractions, offdiag2_fractions)]
    norms_causal = [r["norm_causal"] for r in results]
    norms_diag = [r["norm_diag"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("MC Baseline — Norm Fractions by Diagonal", fontsize=14, fontweight='bold')

    # Plot 1: All fractions on same axes
    axes[0].plot(steps, fractions, 'b.-', markersize=3, linewidth=1, label="Diag (t=s, copy)")
    axes[0].plot(steps, offdiag1_fractions, 'r.-', markersize=3, linewidth=1, label="Off-1 (t=s+1, bigram)")
    axes[0].plot(steps, offdiag2_fractions, 'g.-', markersize=3, linewidth=1, label="Off-2 (t=s+2, trigram)")
    axes[0].plot(steps, combined_12, 'm.-', markersize=3, linewidth=1, label="Off-1 + Off-2")
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Norm Fraction")
    axes[0].set_title("||T_mask||² / ||T_causal||² (MC)")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Both norms
    axes[1].plot(steps, norms_causal, 'r.-', markersize=3, label="Causal (full)")
    axes[1].plot(steps, norms_diag, 'g.-', markersize=3, label="Diagonal only")
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("||T||² (MC baseline)")
    axes[1].set_title("Attention Tensor Norms (MC)")
    axes[1].legend()
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Off-diagonal norm
    off_diag = [c - d for c, d in zip(norms_causal, norms_diag)]
    axes[2].plot(steps, off_diag, 'm.-', markersize=3)
    axes[2].set_xlabel("Training Step")
    axes[2].set_ylabel("||T_offdiag||² (MC baseline)")
    axes[2].set_title("Off-Diagonal Norm (MC)")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = RESULTS_DIR / "diagonal_fraction.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Plot saved to {fig_path}")
    plt.close()


# ─── Cross-checkpoint similarity heatmap ─────────────────────────────────────

@torch.no_grad()
def compute_similarity_matrix(n_samples=N_SAMPLES, batch_size=32, device="cuda"):
    """
    Compute pairwise MC similarity between all checkpoint pairs.

    Efficient approach: for shared random inputs, compute outputs from every
    checkpoint's attention path, then build the Gram matrix from dot products.

    sim(i, j) = G_ij / sqrt(G_ii * G_jj)
    where G_ij = E_x[ attn_i(x) . attn_j(x) ]
    """
    ckpt_files = get_sorted_checkpoints()
    n_ckpts = len(ckpt_files)
    if n_ckpts == 0:
        print("No checkpoints found!")
        return None, []

    # Load all models (small — ~1MB each, 76 total ~76MB)
    print(f"Loading {n_ckpts} checkpoints...")
    models = []
    steps = []
    for ckpt_path in tqdm(ckpt_files, desc="Loading models"):
        model, step = load_model_from_checkpoint(ckpt_path, device=device)
        models.append(model)
        steps.append(step)

    # Get mask and dimensions from first model
    attn0 = models[0].attn
    n_ctx = attn0.n_ctx
    d_model = attn0.d_model
    mask = _get_mask(n_ctx, "causal").to(device=device, dtype=attn0.q1.weight.dtype)

    # Accumulate Gram matrix: G_ij = E_x[ out_i . out_j ]
    G = np.zeros((n_ckpts, n_ckpts))

    n_batches = (n_samples + batch_size - 1) // batch_size
    for b in tqdm(range(n_batches), desc="Similarity (MC)"):
        bs = min(batch_size, n_samples - b * batch_size)
        x = torch.randn(bs, n_ctx, d_model, device=device, dtype=mask.dtype)

        # Compute outputs for all models on this shared input
        outs = []
        for model in models:
            out = _attn_path_forward(model.attn, x, mask)  # (bs, n_ctx, d_model)
            # Flatten to (bs, n_ctx*d_model), move to CPU
            outs.append(out.reshape(bs, -1).cpu().float().numpy())

        # Stack: (n_ckpts, bs, flat_dim)
        outs = np.stack(outs, axis=0)

        # Accumulate: for each sample, add outer product of output vectors
        # G_ij += sum_sample out_i . out_j / n_samples
        for s in range(bs):
            vecs = outs[:, s, :]  # (n_ckpts, flat_dim)
            G += vecs @ vecs.T / n_samples

    # Normalize to cosine similarity
    diag = np.sqrt(np.diag(G))
    diag[diag == 0] = 1.0
    sim_matrix = G / np.outer(diag, diag)

    return sim_matrix, steps


def plot_similarity_heatmap(sim_matrix=None, steps=None):
    """Plot cross-checkpoint similarity as a 2D heatmap. Labeled as MC baseline."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if sim_matrix is None:
        data = np.load(RESULTS_DIR / "similarity_matrix.npz")
        sim_matrix = data["sim_matrix"]
        steps = data["steps"]

    n = len(steps)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    im = ax.imshow(sim_matrix, cmap='Reds', vmin=0, vmax=1,
                   origin='lower', aspect='equal')

    # Tick labels: show every ~10 checkpoints
    tick_stride = max(1, n // 10)
    tick_idx = list(range(0, n, tick_stride))
    tick_labels = [str(steps[i]) for i in tick_idx]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax.set_yticks(tick_idx)
    ax.set_yticklabels(tick_labels)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Training Step")
    ax.set_title("MC Baseline — Cross-Checkpoint Tensor Similarity", fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cosine Similarity (MC)")

    plt.tight_layout()
    fig_path = RESULTS_DIR / "similarity_heatmap.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Heatmap saved to {fig_path}")
    plt.close()


def plot_combined(diag_results=None, sim_matrix=None, steps_sim=None):
    """Vertically stacked: CE loss, diagonal fraction, similarity heatmap.

    The x-axes are aligned so you can visually compare all three panels.
    """
    import matplotlib.gridspec as gridspec

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if diag_results is None:
        with open(RESULTS_DIR / "diagonal_fraction.json") as f:
            diag_results = json.load(f)
    if sim_matrix is None:
        data = np.load(RESULTS_DIR / "similarity_matrix.npz")
        sim_matrix = data["sim_matrix"]
        steps_sim = data["steps"]

    # Load training log for CE loss curve
    train_log_path = CKPT_DIR / "train_log.json"
    train_log = None
    if train_log_path.exists():
        with open(train_log_path) as f:
            train_log = json.load(f)

    steps_diag = [r["step"] for r in diag_results]
    fractions = [r["diag_fraction"] for r in diag_results]
    n = len(steps_sim)

    # Layout: top=CE loss, middle=norm fractions, bottom=heatmap, colorbar
    fig = plt.figure(figsize=(10, 14))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 3], hspace=0.2)

    ax_loss = fig.add_subplot(gs[0])
    ax_diag = fig.add_subplot(gs[1], sharex=ax_loss)
    ax_heat = fig.add_subplot(gs[2], sharex=ax_loss)

    # Build title from config.json in results dir (or checkpoint fallback)
    config_path = RESULTS_DIR / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
    else:
        first_ckpt = sorted(CKPT_DIR.glob("step_*.pt"))[0]
        cfg = torch.load(first_ckpt, map_location="cpu", weights_only=False)["config"]
    d_model = cfg.get("d_model", "?")
    n_head = cfg.get("n_head", "?")
    optim = cfg.get("optimizer", "?")
    init = cfg.get("init", "?")
    title = (f"MC Baseline — d={d_model}, h={n_head}, "
             f"opt={optim}, init={init}")
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.96)

    # Tick positions and labels for shared x-axis
    tick_stride = max(1, n // 10)
    tick_idx = list(range(0, n, tick_stride))
    tick_labels = [str(steps_sim[i]) for i in tick_idx]

    # Map steps to checkpoint indices for aligned x-axis
    step_to_idx = {s: i for i, s in enumerate(steps_sim)}

    # ── Top: CE loss ──────────────────────────────────────────────────────
    if train_log:
        log_steps = [r["step"] for r in train_log]
        log_loss = [r["loss"] for r in train_log]
        # Subsample for plotting (every ~50 steps to avoid overcrowding)
        stride = max(1, len(log_steps) // 500)
        log_steps_sub = log_steps[::stride]
        log_loss_sub = log_loss[::stride]
        # Map log steps to fractional checkpoint indices for alignment
        # Interpolate: find where each step falls between checkpoints
        ckpt_steps_arr = np.array(list(steps_sim))
        log_idx = np.interp(log_steps_sub, ckpt_steps_arr, np.arange(n))
        ax_loss.plot(log_idx, log_loss_sub, 'k-', linewidth=0.5, alpha=0.7)
    ax_loss.set_ylabel("CE Loss")
    ax_loss.set_yscale("log")
    ax_loss.set_xlim(-0.5, n - 0.5)
    ax_loss.set_xticks(tick_idx)
    plt.setp(ax_loss.get_xticklabels(), visible=False)
    ax_loss.grid(True, alpha=0.3)

    # ── Middle: norm fractions ────────────────────────────────────────────
    offdiag1_fractions = [r.get("offdiag1_fraction", 0) for r in diag_results]
    offdiag2_fractions = [r.get("offdiag2_fraction", 0) for r in diag_results]
    combined_12 = [a + b for a, b in zip(offdiag1_fractions, offdiag2_fractions)]

    diag_idx = [step_to_idx[s] for s in steps_diag if s in step_to_idx]
    diag_frac = [f for s, f in zip(steps_diag, fractions) if s in step_to_idx]
    off1_frac = [f for s, f in zip(steps_diag, offdiag1_fractions) if s in step_to_idx]
    off2_frac = [f for s, f in zip(steps_diag, offdiag2_fractions) if s in step_to_idx]
    comb_frac = [f for s, f in zip(steps_diag, combined_12) if s in step_to_idx]

    ax_diag.plot(diag_idx, diag_frac, 'b.-', markersize=3, linewidth=1, label="Diag (t=s, copy)")
    ax_diag.plot(diag_idx, off1_frac, 'r.-', markersize=3, linewidth=1, label="Off-1 (t=s+1, bigram)")
    ax_diag.plot(diag_idx, off2_frac, 'g.-', markersize=3, linewidth=1, label="Off-2 (t=s+2, trigram)")
    ax_diag.plot(diag_idx, comb_frac, 'm.-', markersize=3, linewidth=1, label="Off-1 + Off-2")
    ax_diag.set_ylabel("Norm Fraction")
    ax_diag.legend(loc='upper right', fontsize=7)
    ax_diag.set_xticks(tick_idx)
    plt.setp(ax_diag.get_xticklabels(), visible=False)
    ax_diag.grid(True, alpha=0.3)

    # ── Bottom: similarity heatmap ────────────────────────────────────────
    im = ax_heat.imshow(sim_matrix, cmap='Reds', vmin=0, vmax=1,
                        origin='lower', aspect='auto',
                        extent=[-0.5, n - 0.5, -0.5, n - 0.5])
    ax_heat.set_xticks(tick_idx)
    ax_heat.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax_heat.set_yticks(tick_idx)
    ax_heat.set_yticklabels(tick_labels)
    ax_heat.set_xlabel("Training Step")
    ax_heat.set_ylabel("Training Step")

    # Horizontal colorbar below heatmap with extra padding
    cbar = plt.colorbar(im, ax=ax_heat, orientation='horizontal', pad=0.12,
                        shrink=0.8, aspect=40)
    cbar.set_label("Cosine Similarity (MC)")

    fig_path = RESULTS_DIR / "combined_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Combined plot saved to {fig_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--diag", action="store_true", help="Run diagonal fraction analysis")
    parser.add_argument("--sim", action="store_true", help="Run cross-checkpoint similarity")
    parser.add_argument("--plot", action="store_true", help="Plot from saved results")
    parser.add_argument("--all", action="store_true", help="Run everything")
    args = parser.parse_args()

    if args.all or (not args.diag and not args.sim and not args.plot):
        args.diag = args.sim = args.plot = True

    diag_results = None
    sim_matrix = None
    steps_sim = None

    if args.diag:
        diag_results = analyze_all_checkpoints()
        if diag_results:
            plot_diagonal_fraction(diag_results)

    if args.sim:
        sim_matrix, steps_sim = compute_similarity_matrix()
        if sim_matrix is not None:
            np.savez(RESULTS_DIR / "similarity_matrix.npz",
                     sim_matrix=sim_matrix, steps=steps_sim)
            print(f"Similarity matrix saved to {RESULTS_DIR / 'similarity_matrix.npz'}")
            plot_similarity_heatmap(sim_matrix, steps_sim)

    if args.plot:
        plot_diagonal_fraction()
        try:
            data = np.load(RESULTS_DIR / "similarity_matrix.npz")
            plot_similarity_heatmap(data["sim_matrix"], data["steps"])
            plot_combined()
        except FileNotFoundError:
            print("No similarity data found, skipping heatmap plots")
