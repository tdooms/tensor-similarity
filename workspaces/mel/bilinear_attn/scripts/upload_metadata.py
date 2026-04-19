#!/usr/bin/env python
"""Upload model metadata (config.json and model code) to Hugging Face repo.

Usage:
    python upload_metadata.py --repo-id melephant/2l-bilinear-attention --repo-type dataset
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="Upload model metadata to Hugging Face")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo ID")
    parser.add_argument("--repo-type", default="dataset", help="Repo type (model or dataset)")
    parser.add_argument("--config", default="configs/pile_dsir.yaml", help="Path to config file")
    parser.add_argument("--models-src", default="models", help="Path to models directory")
    parser.add_argument("--private", action="store_true", help="Create private repo if it doesn't exist")
    
    args = parser.parse_args()
    
    api = HfApi()
    
    # Create repo if it doesn't exist
    try:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            private=args.private,
            exist_ok=True,
        )
        print(f"Ensured repo exists: hf://{args.repo_type}/{args.repo_id}")
    except Exception as e:
        print(f"Warning: Could not create repo: {e}")
    
    # Upload config.json
    config_path = Path(args.config)
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        config_json = json.dumps(config, indent=2)
        api.upload_file(
            path_or_fileobj=config_json.encode(),
            path_in_repo="config.json",
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            commit_message="Upload model config.json",
        )
        print(f"Uploaded config.json to hf://{args.repo_type}/{args.repo_id}")
    else:
        print(f"Config file not found: {config_path}")
    
    # Upload model code
    models_src = Path(args.models_src)
    if not models_src.exists():
        print(f"Models directory not found: {models_src}")
        return
    
    required_files = ("__init__.py", "transformer.py")
    for name in required_files:
        src_path = models_src / name
        if src_path.is_file():
            api.upload_file(
                path_or_fileobj=str(src_path),
                path_in_repo=f"models/{name}",
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                commit_message=f"Upload models/{name}",
            )
            print(f"Uploaded models/{name} to hf://{args.repo_type}/{args.repo_id}")
    
    # Upload attention_kernels directory
    attn_src = models_src / "attention_kernels"
    if attn_src.is_dir():
        for py_file in attn_src.glob("*.py"):
            if not py_file.name.startswith("_"):
                api.upload_file(
                    path_or_fileobj=str(py_file),
                    path_in_repo=f"models/attention_kernels/{py_file.name}",
                    repo_id=args.repo_id,
                    repo_type=args.repo_type,
                    commit_message=f"Upload models/attention_kernels/{py_file.name}",
                )
                print(f"Uploaded models/attention_kernels/{py_file.name} to hf://{args.repo_type}/{args.repo_id}")
    
    print("Done!")


if __name__ == "__main__":
    main()
