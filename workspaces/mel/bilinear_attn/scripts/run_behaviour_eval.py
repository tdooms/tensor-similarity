#!/usr/bin/env python3
"""Evaluate behaviour metrics on saved checkpoints.

Usage (from the bilinear_attn directory):
    # Evaluate all checkpoints in a run directory (config auto-detected):
    python -m scripts.run_behaviour_eval --run-dir runs/2025-01-01_120000_main256

    # Evaluate with an explicit config (e.g. config not saved in run dir):
    python -m scripts.run_behaviour_eval --run-dir runs/my_run --config configs/main256.yaml

    # Evaluate a single checkpoint:
    python -m scripts.run_behaviour_eval --checkpoint runs/my_run/checkpoints/step_5000.pt --config configs/main256.yaml

    # Disable n-gram tracking for faster evaluation:
    python -m scripts.run_behaviour_eval --run-dir runs/my_run --config configs/main256.yaml --no-ngram
"""
import argparse
import torch
from pathlib import Path

from analysis.behaviour import BehaviourTracker, TrackerConfig, plot_all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate behaviour metrics on saved checkpoints"
    )
    
    # Source: either a run directory or explicit checkpoint(s)
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Run directory containing checkpoints/ folder")
    parser.add_argument("--checkpoint", type=str, nargs="+", default=None,
                        help="Explicit checkpoint path(s) to evaluate")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML (auto-detected from run-dir if omitted)")
    
    # Tracker settings
    parser.add_argument("--no-bigram", action="store_true",
                        help="Disable bigram tracking")
    parser.add_argument("--no-ngram", action="store_true",
                        help="Disable n-gram tracking")
    parser.add_argument("--no-ablation", action="store_true",
                        help="Disable position-ablated loss tracking")
    parser.add_argument("--bigram-n-samples", type=int, default=500,
                        help="Number of samples for bigram scoring")
    parser.add_argument("--ngram-max-n", type=int, default=4,
                        help="Maximum n for n-gram scoring")
    parser.add_argument("--ngram-max-val-batches", type=int, default=20,
                        help="Max validation batches for n-gram position loss")
    parser.add_argument("--ablation-max-val-batches", type=int, default=50,
                        help="Max validation batches for ablated loss")
    parser.add_argument("--max-fit-samples", type=int, default=5000,
                        help="Max samples for fitting analyzers")
    
    # Infrastructure
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Cache directory for analyzers (default: <run-dir>/behaviour_cache)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate behaviour plots after evaluation")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results (default: run-dir)")
    
    args = parser.parse_args()
    
    if args.run_dir is None and args.checkpoint is None:
        parser.error("Provide --run-dir or --checkpoint")
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device == "cuda":
        torch.cuda.init()
        torch.cuda.empty_cache()
    
    # Build tracker config
    tracker_config = TrackerConfig(
        bigram_enabled=not args.no_bigram,
        bigram_compute_every=1,  # irrelevant for checkpoint eval
        bigram_n_samples=args.bigram_n_samples,
        ngram_enabled=not args.no_ngram,
        ngram_compute_every=1,
        ngram_max_n=args.ngram_max_n,
        ngram_max_val_batches=args.ngram_max_val_batches,
        ablation_enabled=not args.no_ablation,
        ablation_compute_every=1,
        ablation_max_val_batches=args.ablation_max_val_batches,
    )
    
    # Determine output directory
    output_dir = args.output_dir or args.run_dir
    
    # Build tracker
    if args.run_dir is not None:
        print(f"Building tracker from run directory: {args.run_dir}")
        tracker = BehaviourTracker.from_run_dir(
            run_dir=args.run_dir,
            config=tracker_config,
            device=device,
            cache_dir=args.cache_dir,
            cfg_path=args.config,
        )
    else:
        # Explicit checkpoint(s) require a config
        if args.config is None:
            parser.error("--config is required when using --checkpoint without --run-dir")
        
        import yaml
        from models import AttentionLM
        from data import create_dataloaders
        
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        
        model_cfg = cfg["model"]
        train_cfg = cfg.get("train", {})
        
        model = AttentionLM.from_config(cfg)
        model.to(device)
        
        train_dataloader, val_dataloader = create_dataloaders(
            n_ctx=model_cfg["n_ctx"],
            batch_size=train_cfg.get("batch_size", 16),
            max_train_samples=train_cfg.get("max_train_samples"),
            max_val_samples=train_cfg.get("max_val_samples", 1000),
        )
        
        tracker = BehaviourTracker(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            vocab_size=model_cfg["vocab_size"],
            device=device,
            config=tracker_config,
            run_dir=output_dir,
            cache_dir=args.cache_dir or "cache/behaviour",
        )
    
    # Fit analyzers (loads from cache if available)
    print("Fitting behaviour analyzers...")
    tracker.fit(max_fit_samples=args.max_fit_samples)
    
    # Evaluate
    print("\nEvaluating checkpoints...")
    checkpoint_paths = [Path(p) for p in args.checkpoint] if args.checkpoint else None
    all_metrics = tracker.evaluate_checkpoints(
        checkpoint_paths=checkpoint_paths,
        run_dir=output_dir,
    )
    
    # Plot
    if args.plot and all_metrics:
        print("\nGenerating behaviour plots...")
        save_dir = Path(output_dir) if output_dir else tracker.run_dir
        if save_dir is not None:
            plot_all_metrics(
                all_metrics,
                save_path=save_dir / "behaviour_checkpoint_plots.png",
                max_n=args.ngram_max_n,
            )
            print(f"Plots saved to {save_dir / 'behaviour_checkpoint_plots.png'}")
    
    print(f"\nDone. Evaluated {len(all_metrics)} checkpoint(s).")


if __name__ == "__main__":
    main()
