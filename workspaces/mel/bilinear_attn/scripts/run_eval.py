#!/usr/bin/env python3
"""Evaluation entrypoint for Quadratic Attention LM."""
import argparse
import yaml
import torch
from models import AttentionLM
from data.cached import create_dataloaders
from train.eval import evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate Quadratic Attention LM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--max_batches", type=int, default=None, help="Max batches to evaluate")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_cfg = cfg["model"]
    
    print("Creating dataloader...")
    _, val_dataloader = create_dataloaders(
        n_ctx=model_cfg["n_ctx"],
        batch_size=cfg.get("train", {}).get("batch_size", 16),
        max_val_samples=None,
    )
    
    print("Building model...")
    model = AttentionLM.from_config(cfg)
    
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    print("Evaluating...")
    val_loss = evaluate(model, val_dataloader, device, max_batches=args.max_batches)
    print(f"Validation loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()
