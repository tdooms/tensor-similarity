#!/usr/bin/env python3
"""Training entrypoint for Quadratic Attention LM."""
import argparse
import random

import numpy as np
import yaml
import torch
from models import AttentionLM
from data.cached import create_dataloaders as create_cached_simplestories_dataloaders
from data.pile import create_pile_dataloaders
from train.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train Quadratic Attention LM")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--track-behaviour", action="store_true", help="Enable behaviour tracking")
    parser.add_argument("--behaviour-every", type=int, default=100, help="Compute behaviour metrics every N steps")
    parser.add_argument("--no-bigram", action="store_true", help="Disable bigram tracking")
    parser.add_argument("--no-ngram", action="store_true", help="Disable n-gram tracking")
    parser.add_argument("--no-ablation", action="store_true", help="Disable position-ablated loss tracking")
    parser.add_argument("--behaviour-cache-dir", type=str, default="cache/behaviour", help="Directory for cached bigram/ngram distributions")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize CUDA before any operations to avoid CUBLAS errors
    if device == "cuda":
        torch.cuda.init()
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
    
    model_cfg = cfg["model"]
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    dataset_name = data_cfg.get("name", "simplestories_cached")
    
    print("Creating dataloaders...")
    if dataset_name == "pile_dsir":
        train_dataloader, val_dataloader = create_pile_dataloaders(
            n_ctx=model_cfg["n_ctx"],
            batch_size=train_cfg.get("batch_size", 16),
            vocab_size=model_cfg["vocab_size"],
            max_val_samples=train_cfg.get("max_val_samples", 500),
            val_cache_size=data_cfg.get("val_cache_size", 500),
            cache_dir=data_cfg.get("cache_dir"),
            tokenizer_name=data_cfg.get("tokenizer", "gpt2"),
            dataset_repo=data_cfg.get("repo", "stanford-crfm/DSIR-filtered-pile-50M"),
            text_field=data_cfg.get("text_field", "contents"),
            train_split=data_cfg.get("train_split", "train"),
            val_split=data_cfg.get("val_split", "validation"),
            dataset_revision=data_cfg.get("revision"),
            train_shuffle=data_cfg.get("train_shuffle", True),
            train_shuffle_seed=data_cfg.get("train_shuffle_seed", seed),
            train_shuffle_buffer_size=data_cfg.get("train_shuffle_buffer_size", 10_000),
            insert_eos_between_documents=data_cfg.get("insert_eos_between_documents", True),
            holdout_fraction=data_cfg.get("holdout_fraction", 0.01),
            holdout_seed=data_cfg.get("holdout_seed", seed),
        )
    elif dataset_name == "simplestories_cached":
        train_dataloader, val_dataloader = create_cached_simplestories_dataloaders(
            n_ctx=model_cfg["n_ctx"],
            batch_size=train_cfg.get("batch_size", 16),
            max_train_samples=train_cfg.get("max_train_samples"),
            max_val_samples=train_cfg.get("max_val_samples", 1000),
        )
    else:
        raise ValueError(
            f"Unknown data.name={dataset_name!r}. "
            "Expected one of: 'simplestories_cached', 'pile_dsir'."
        )
    
    print("Building model...")
    model = AttentionLM.from_config(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Set up behaviour tracker if enabled
    behaviour_tracker = None
    if args.track_behaviour:
        from analysis.behaviour import BehaviourTracker, TrackerConfig
        
        tracker_config = TrackerConfig(
            bigram_enabled=not args.no_bigram,
            bigram_compute_every=args.behaviour_every,
            bigram_n_samples=500,
            ngram_enabled=not args.no_ngram,
            ngram_compute_every=args.behaviour_every,
            ngram_max_n=4,
            ngram_max_val_batches=20,
            ablation_enabled=not args.no_ablation,
            ablation_compute_every=args.behaviour_every,
        )
        
        behaviour_tracker = BehaviourTracker(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            vocab_size=model_cfg["vocab_size"],
            device=device,
            config=tracker_config,
            cache_dir=args.behaviour_cache_dir,
        )
        
        print("Fitting behaviour analyzers...")
        behaviour_tracker.fit(max_fit_samples=5000)
    
    # Initialize wandb if enabled
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            entity="melwina-albuquerque-flame-university",
            project="bilinear-attn",
            config=cfg,
        )
    
    print("Starting training...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        cfg=cfg,
        device=device,
        behaviour_tracker=behaviour_tracker,
        wandb_run=wandb_run,
    )
    
    trainer.train(
        eval_every=train_cfg.get("eval_every", 500),
        save_every=train_cfg.get("save_every", train_cfg.get("checkpoint_every", 1000)),
    )
    
    # Save behaviour metrics and generate plots if tracking was enabled
    if behaviour_tracker is not None:
        behaviour_tracker.run_dir = trainer.run_dir
        behaviour_tracker._metrics_file = trainer.run_dir / "behaviour_metrics.jsonl"
        behaviour_tracker.save_history(trainer.run_dir / "behaviour_metrics_full.json")
        
        from analysis.behaviour import plot_all_metrics, load_metrics
        metrics = behaviour_tracker.get_history()
        if metrics:
            print("Generating behaviour plots...")
            plot_all_metrics(metrics, save_path=trainer.run_dir / "behaviour_plots.png")
    
    if wandb_run is not None:
        wandb_run.finish()
    
    print(f"Training complete. Run dir: {trainer.run_dir}")


if __name__ == "__main__":
    main()
