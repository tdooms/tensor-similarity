#!/usr/bin/env python3
"""Compare TN similarity vs MC similarity across checkpoints.

Loads all checkpoints from a training run, computes pairwise similarities
using both methods, and plots heatmaps.

Usage (from bilinear_attn directory):
    python -m tn_sim.compare --run-dir tn_sim/runs/<timestamp>_tiny
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from models import AttentionLM
from tn_sim.tn_similarity import compute_tn_similarity
from tn_sim.mc_similarity import mc_similarity_gaussian


def load_model(cfg, ckpt_path, device):
    """Create model from config and load checkpoint weights."""
    model = AttentionLM.from_config(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, ckpt.get("step", -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mc-samples", type=int, default=4000,
                        help="Number of random-token samples for MC similarity")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    ckpt_dir = run_dir / "checkpoints"
    ckpt_files = sorted(ckpt_dir.glob("step_*.pt"))
    print(f"Found {len(ckpt_files)} checkpoints in {ckpt_dir}")

    # Load all models
    models = []
    steps = []
    for cp in ckpt_files:
        m, s = load_model(cfg, cp, device)
        models.append(m)
        steps.append(s)
        print(f"  Loaded step {s}")

    n = len(models)
    vocab_size = cfg["model"]["vocab_size"]
    n_ctx = cfg["model"]["n_ctx"]

    # ── TN similarity ─────────────────────────────────────────────────────
    print("\nComputing TN similarities ...")
    tn_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            sim = compute_tn_similarity(models[i], models[j], device=device)
            tn_sim[i, j] = sim
            tn_sim[j, i] = sim
            print(f"  TN sim({steps[i]}, {steps[j]}) = {sim:.4f}")

    # ── MC similarity ─────────────────────────────────────────────────────
    print("\nComputing MC similarities ...")
    mc_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            sim = mc_similarity_gaussian(
                models[i], models[j],
                vocab_size=vocab_size, n_ctx=n_ctx,
                device=device, n_samples=args.mc_samples,
            )
            mc_sim[i, j] = sim
            mc_sim[j, i] = sim
            print(f"  MC sim({steps[i]}, {steps[j]}) = {sim:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────
    step_labels = [str(s) for s in steps]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im0 = axes[0].imshow(tn_sim, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
    axes[0].set_title("TN Similarity (Exact / Gaussian approx)")
    axes[0].set_xticks(range(n))
    axes[0].set_xticklabels(step_labels, rotation=45, ha="right")
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels(step_labels)
    axes[0].set_xlabel("Checkpoint step")
    axes[0].set_ylabel("Checkpoint step")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(mc_sim, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
    axes[1].set_title("MC Similarity (Uniform tokens)")
    axes[1].set_xticks(range(n))
    axes[1].set_xticklabels(step_labels, rotation=45, ha="right")
    axes[1].set_yticks(range(n))
    axes[1].set_yticklabels(step_labels)
    axes[1].set_xlabel("Checkpoint step")
    axes[1].set_ylabel("Checkpoint step")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = run_dir / "similarity_heatmaps.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved heatmap to {out_path}")
    plt.show()

    # Save raw data
    np.savez(
        run_dir / "similarity_data.npz",
        tn_sim=tn_sim, mc_sim=mc_sim, steps=np.array(steps),
    )
    print(f"Saved raw data to {run_dir / 'similarity_data.npz'}")


if __name__ == "__main__":
    main()
