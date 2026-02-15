#!/usr/bin/env python3
"""Compare results from different Q/K normalization experiments.

Usage (from the bilinear_attn directory):
    python -m experiments.norms.compare --run-dirs experiments/norms/runs/*
"""

import argparse
import json
from pathlib import Path
import pandas as pd


def load_metrics(run_dir: Path) -> dict:
    """Load final metrics from a run directory."""
    metrics_file = run_dir / "metrics.jsonl"
    if not metrics_file.exists():
        return None
    
    final_metrics = {}
    with open(metrics_file) as f:
        for line in f:
            data = json.loads(line)
            if "final_val_loss" in data:
                final_metrics = data
                break
    
    if not final_metrics:
        lines = list(open(metrics_file))
        if lines:
            final_metrics = json.loads(lines[-1])
    
    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Compare Q/K normalization experiments")
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Run directories to compare")
    args = parser.parse_args()
    
    results = []
    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str)
        if not run_dir.exists():
            print(f"Warning: {run_dir} does not exist, skipping")
            continue
        
        metrics = load_metrics(run_dir)
        if metrics is None:
            print(f"Warning: No metrics found in {run_dir}, skipping")
            continue
        
        config_file = run_dir / "config.yaml"
        if config_file.exists():
            import yaml
            with open(config_file) as f:
                cfg = yaml.safe_load(f)
            qk_norm_type = cfg.get("model", {}).get("qk_norm_type", "unknown")
            attn_type = cfg.get("model", {}).get("attn_type", "unknown")
        else:
            qk_norm_type = "unknown"
            attn_type = "unknown"
        
        results.append({
            "run_name": run_dir.name,
            "qk_norm_type": qk_norm_type,
            "attn_type": attn_type,
            "final_val_loss": metrics.get("final_val_loss", metrics.get("val_loss", float("nan"))),
            "final_val_acc": metrics.get("final_val_acc", metrics.get("val_acc", float("nan"))),
        })
    
    if not results:
        print("No results to compare")
        return
    
    df = pd.DataFrame(results)
    df = df.sort_values("final_val_loss")
    
    print("\n" + "=" * 80)
    print("Q/K NORMALIZATION COMPARISON")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    
    print("\nBest by validation loss:")
    best = df.iloc[0]
    print(f"  {best['qk_norm_type']} ({best['attn_type']}): "
          f"val_loss={best['final_val_loss']:.4f}, val_acc={best['final_val_acc']:.4f}")
    
    print("\nBest by validation accuracy:")
    best_acc = df.loc[df["final_val_acc"].idxmax()]
    print(f"  {best_acc['qk_norm_type']} ({best_acc['attn_type']}): "
          f"val_loss={best_acc['final_val_loss']:.4f}, val_acc={best_acc['final_val_acc']:.4f}")


if __name__ == "__main__":
    main()
