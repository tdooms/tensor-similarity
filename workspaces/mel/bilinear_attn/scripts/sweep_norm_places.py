#!/usr/bin/env python3
"""Sweep over norm_places configurations using wandb.

Runs 4 training jobs:
  1. norm_places: []               (no norm)
  2. norm_places: [pre_unembed]
  3. norm_places: [post_embed]
  4. norm_places: [post_embed, pre_unembed]

Usage:
    python -m scripts.sweep_norm_places --config configs/sweep_base.yaml
"""
import argparse
import copy
import yaml
import torch
from models import AttentionLM
from data.cached import create_dataloaders
from train.trainer import Trainer


NORM_PLACES_COMBOS = [
    [],
    ["pre_unembed"],
    ["post_embed"],
    ["post_embed", "pre_unembed"],
]


def _combo_name(norm_places: list[str]) -> str:
    """Human-readable name for a norm_places combo."""
    if not norm_places:
        return "none"
    return "+".join(norm_places)


def run_single(cfg: dict, norm_places: list[str], device: str):
    """Run a single training job with the given norm_places."""
    import wandb

    cfg = copy.deepcopy(cfg)
    cfg["model"]["norm_places"] = norm_places

    combo = _combo_name(norm_places)
    cfg["name"] = f"norm_{combo}"

    wandb_run = wandb.init(
        entity="melwina-albuquerque-flame-university",
        project="bilinear-attn",
        group="sweep_norm_places",
        name=f"norm_{combo}",
        config=cfg,
        reinit=True,
    )

    torch.manual_seed(cfg.get("seed", 0))

    model_cfg = cfg["model"]
    train_cfg = cfg.get("train", {})

    print(f"\n{'='*60}")
    print(f"  Sweep run: norm_places={norm_places}")
    print(f"{'='*60}")

    print("Creating dataloaders...")
    train_dataloader, val_dataloader = create_dataloaders(
        n_ctx=model_cfg["n_ctx"],
        batch_size=train_cfg.get("batch_size", 16),
        max_train_samples=train_cfg.get("max_train_samples"),
        max_val_samples=train_cfg.get("max_val_samples", 1000),
    )

    print("Building model...")
    model = AttentionLM.from_config(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    print("Starting training...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        cfg=cfg,
        device=device,
        wandb_run=wandb_run,
    )

    trainer.train(
        eval_every=train_cfg.get("eval_every", 500),
        save_every=train_cfg.get("save_every", 1000),
    )

    print(f"Run complete. Run dir: {trainer.run_dir}")
    wandb_run.finish()


def main():
    parser = argparse.ArgumentParser(description="Sweep over norm_places")
    parser.add_argument("--config", type=str, default="configs/sweep_base.yaml",
                        help="Path to base config YAML")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device == "cuda":
        torch.cuda.init()
        torch.cuda.empty_cache()

    for norm_places in NORM_PLACES_COMBOS:
        run_single(cfg, norm_places, device)

    print("\nSweep complete! All 4 runs finished.")


if __name__ == "__main__":
    main()
