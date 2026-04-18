"""Run a grid sweep over n_ctx, d_model, and n_layers for induction head experiments.

This script runs all combinations of:
- n_ctx: [8, 10, 12, 16]
- d_model: [8, 12, 16]
- n_layers: [1, 2]

Total: 24 runs with wandb logging
"""

import itertools
import subprocess
import sys
from pathlib import Path

import yaml

# Grid parameters
n_ctx_values = [8, 16]
d_model_values = [8, 16]
n_layers_values = [1, 2]

# Base config path
config_path = Path("configs/mini.yaml")

# Load base config
with open(config_path, 'r') as f:
    base_config = yaml.safe_load(f)

# Create temporary config directory
temp_config_dir = Path("configs/sweep_temp")
temp_config_dir.mkdir(exist_ok=True)

print("=" * 80)
print("INDUCTION HEAD SWEEP")
print("=" * 80)
print(f"\nGrid:")
print(f"  n_ctx: {n_ctx_values}")
print(f"  d_model: {d_model_values}")
print(f"  n_layers: {n_layers_values}")
print(f"\nTotal runs: {len(n_ctx_values) * len(d_model_values) * len(n_layers_values)}")
print("=" * 80)

run_count = 0
total_runs = len(n_ctx_values) * len(d_model_values) * len(n_layers_values)

for n_ctx, d_model, n_layers in itertools.product(n_ctx_values, d_model_values, n_layers_values):
    run_count += 1
    
    print(f"\n[{run_count}/{total_runs}] Running: n_ctx={n_ctx}, d_model={d_model}, n_layers={n_layers}")
    
    # Create modified config
    config = base_config.copy()
    config['model']['n_ctx'] = n_ctx
    config['model']['vocab_size'] = n_ctx  # Set vocab_size = n_ctx
    config['model']['d_model'] = d_model
    config['model']['n_layers'] = n_layers
    
    # Update name to reflect parameters
    config['name'] = f"sweep_ctx{n_ctx}_d{d_model}_L{n_layers}"
    
    # Save temporary config
    temp_config_path = temp_config_dir / f"config_ctx{n_ctx}_d{d_model}_L{n_layers}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    cmd = [
        sys.executable,
        "-m",
        "experiments.induction_heads.run",
        "--config",
        str(temp_config_path),
        "--wandb",
    ]

    print(f"  Running with wandb...")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"  ✓ Completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed with exit code {e.returncode}")
        print(f"  Continuing with next run...")
    except KeyboardInterrupt:
        print(f"\n\nSweep interrupted by user")
        break

print("\n" + "=" * 80)
print("SWEEP COMPLETE")
print("=" * 80)
print(f"Completed {run_count}/{total_runs} runs")
print(f"\nView results at: https://wandb.ai/melwina-albuquerque-flame-university/bilinear-induction-heads")
