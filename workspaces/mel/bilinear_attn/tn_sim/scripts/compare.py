#!/usr/bin/env python3
"""Compare TN similarity vs MC similarity across checkpoints.

Loads all checkpoints from a training run, computes pairwise similarities
using both methods, and plots heatmaps.

Usage (from bilinear_attn directory):
    python -m tn_sim.compare --run-dir runs/<timestamp>

Note: TN similarity requires models trained with norm_type='none' and norm_places=[].
For models with normalization, only MC similarity will be computed.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from models import AttentionLM
from tn_sim.similarity import cosine_similarity as compute_tn_similarity
from tn_sim.mc_similarity import mc_similarity


def load_model(cfg, ckpt_path, device):
    """Create model from config and load checkpoint weights."""
    model = AttentionLM.from_config(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, ckpt.get("step", -1)


def is_tn_compatible(cfg):
    """Check if model config is compatible with TN similarity."""
    model_cfg = cfg.get("model", {})
    norm_type = model_cfg.get("norm_type", "none")
    norm_places = model_cfg.get("norm_places", [])
    use_rmsnorm_qk = model_cfg.get("use_rmsnorm_qk", False)
    attn_type = model_cfg.get("attn_type", "quadratic")
    
    if norm_type != "none":
        return False, f"norm_type={norm_type!r} (must be 'none')"
    if norm_places:
        return False, f"norm_places={norm_places} (must be empty)"
    if use_rmsnorm_qk:
        return False, "use_rmsnorm_qk=True (must be False)"
    if attn_type not in ("bilinear", "quadratic"):
        return False, f"attn_type={attn_type!r} (must be 'bilinear' or 'quadratic')"
    
    return True, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mc-samples", type=int, default=4000,
                        help="Number of Gaussian residual-stream samples for MC similarity")
    parser.add_argument("--mc-only", action="store_true",
                        help="Skip TN similarity (useful for models with normalization)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Check TN compatibility
    tn_compatible, tn_reason = is_tn_compatible(cfg)
    if not tn_compatible and not args.mc_only:
        print(f"Warning: Model not compatible with TN similarity: {tn_reason}")
        print("         Use --mc-only to skip TN similarity, or train with norm_type='none'")
        args.mc_only = True

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
    # ── TN similarity ─────────────────────────────────────────────────────
    tn_sim = None
    if not args.mc_only:
        print("\nComputing TN similarities (using main codebase exact algorithm)...")
        print("  Note: This may take several minutes for larger models.")
        tn_sim = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                try:
                    sim = compute_tn_similarity(models[i], models[j], device=device)
                    tn_sim[i, j] = sim
                    tn_sim[j, i] = sim
                    print(f"  TN sim({steps[i]}, {steps[j]}) = {sim:.4f}")
                except Exception as e:
                    print(f"  TN sim({steps[i]}, {steps[j]}) FAILED: {e}")
                    tn_sim[i, j] = np.nan
                    tn_sim[j, i] = np.nan

    # ── MC similarity ─────────────────────────────────────────────────────
    print("\nComputing MC similarities ...")
    mc_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            sim = mc_similarity(
                models[i], models[j],
                device=device, n_samples=args.mc_samples,
            )
            mc_sim[i, j] = sim
            mc_sim[j, i] = sim
            print(f"  MC sim({steps[i]}, {steps[j]}) = {sim:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────
    step_labels = [str(s) for s in steps]

    if tn_sim is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        im0 = axes[0].imshow(tn_sim, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
        axes[0].set_title("TN Similarity (Exact / Main Codebase)")
        axes[0].set_xticks(range(n))
        axes[0].set_xticklabels(step_labels, rotation=45, ha="right")
        axes[0].set_yticks(range(n))
        axes[0].set_yticklabels(step_labels)
        axes[0].set_xlabel("Checkpoint step")
        axes[0].set_ylabel("Checkpoint step")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(mc_sim, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
        axes[1].set_title("MC Similarity (Gaussian residual stream)")
        axes[1].set_xticks(range(n))
        axes[1].set_xticklabels(step_labels, rotation=45, ha="right")
        axes[1].set_yticks(range(n))
        axes[1].set_yticklabels(step_labels)
        axes[1].set_xlabel("Checkpoint step")
        axes[1].set_ylabel("Checkpoint step")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        # MC only mode
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        im = ax.imshow(mc_sim, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
        ax.set_title("MC Similarity (Gaussian residual stream)")
        ax.set_xticks(range(n))
        ax.set_xticklabels(step_labels, rotation=45, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(step_labels)
        ax.set_xlabel("Checkpoint step")
        ax.set_ylabel("Checkpoint step")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = run_dir / "similarity_heatmaps.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved heatmap to {out_path}")
    plt.show()

    # Save raw data
    save_data = {"mc_sim": mc_sim, "steps": np.array(steps)}
    if tn_sim is not None:
        save_data["tn_sim"] = tn_sim
    np.savez(run_dir / "similarity_data.npz", **save_data)
    print(f"Saved raw data to {run_dir / 'similarity_data.npz'}")


if __name__ == "__main__":
    main()
