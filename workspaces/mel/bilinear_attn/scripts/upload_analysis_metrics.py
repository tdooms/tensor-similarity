#!/usr/bin/env python
"""Upload analysis_metrics.jsonl to HuggingFace repo using metadata.

Usage:
    python upload_analysis_metrics.py --run-dir experiments/pile_metrics/runs/melephant_2l-bilinear-attn-normalised-v2/checkpoints
"""

import argparse
import yaml
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="Upload analysis_metrics.jsonl to HuggingFace")
    parser.add_argument("--run-dir", required=True, help="Run directory containing metadata/ and metrics/")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    metadata_dir = run_dir / "metadata"
    metrics_dir = run_dir / "metrics"
    
    # Load run settings to get HF repo info
    run_settings_path = metadata_dir / "run_settings.yaml"
    if not run_settings_path.exists():
        print(f"Error: run_settings.yaml not found at {run_settings_path}")
        return
    
    with open(run_settings_path) as f:
        run_settings = yaml.safe_load(f)
    
    hf_repo = run_settings.get("hf_repo")
    if not hf_repo:
        print("Error: hf_repo not found in run_settings.yaml")
        return
    
    # Path to analysis_metrics.jsonl
    metrics_path = metrics_dir / "analysis_metrics.jsonl"
    if not metrics_path.exists():
        print(f"Error: analysis_metrics.jsonl not found at {metrics_path}")
        return
    
    api = HfApi()
    repo_type = "dataset"  # Checkpoints are stored as datasets
    
    try:
        api.upload_file(
            path_or_fileobj=str(metrics_path),
            path_in_repo="metrics/analysis_metrics.jsonl",
            repo_id=hf_repo,
            repo_type=repo_type,
            commit_message="Upload analysis_metrics.jsonl",
        )
        print(f"Uploaded analysis_metrics.jsonl to hf://{repo_type}/{hf_repo}/metrics/analysis_metrics.jsonl")
    except Exception as e:
        print(f"Error uploading file: {e}")
        return


if __name__ == "__main__":
    main()
