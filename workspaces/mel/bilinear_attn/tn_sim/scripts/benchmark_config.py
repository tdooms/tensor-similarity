#!/usr/bin/env python3
"""Benchmark different configs to find the largest that runs in <2 minutes.

Non-negotiable constraints:
- n_layers = 2
- n_head >= 2
- d_model = 12 (preferred)
- n_ctx >= 4

We'll test increasing n_ctx values with fixed d_model=12, n_layers=2, n_head=2.
"""

import sys
from pathlib import Path
import time
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from models import AttentionLM
from models.components import AttentionLMComponent
from src.components.similarity import similarity

def cosine_sim(state):
    tr = lambda S: torch.einsum('ijij->', S[:, 1:, :, 1:])
    return (tr(state.S_ab) / (tr(state.S_aa) * tr(state.S_bb)) ** 0.5).item()

def benchmark_config(n_ctx, d_model=12, n_layers=2, n_head=2, timeout=120):
    """Benchmark a single config and return time taken."""
    cfg = {
        "model": {
            "vocab_size": 32,
            "n_ctx": n_ctx,
            "d_model": d_model,
            "n_head": n_head,
            "n_layers": n_layers,
            "attn_scale": 0.5,
            "attn_type": "bilinear",
            "use_bias_qk": False,
            "use_rmsnorm_qk": False,
            "norm_type": "none",
            "norm_places": [],
            "rope_base": 10000,
        },
        "init": {
            "std_embed": 0.02,
            "std_qkv": 0.02,
            "std_o": 0.01,
        }
    }
    
    print(f"\nTesting: n_ctx={n_ctx}, d_model={d_model}, n_layers={n_layers}, n_head={n_head}")
    print("-" * 60)
    
    try:
        # Create model
        torch.manual_seed(42)
        model = AttentionLM.from_config(cfg).double()
        comp = AttentionLMComponent.from_trained_model(model)
        
        # Benchmark
        start = time.time()
        state = similarity(comp, comp)
        elapsed = time.time() - start
        
        cos = cosine_sim(state)
        
        print(f"✓ Success!")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Cosine similarity: {cos:.10f}")
        print(f"  Within timeout: {'YES' if elapsed < timeout else 'NO'}")
        
        return elapsed, True
        
    except Exception as e:
        print(f"✗ Failed: {type(e).__name__}: {e}")
        return None, False

def find_optimal_config():
    """Find the largest config that completes in <2 minutes."""
    print("=" * 60)
    print("FINDING OPTIMAL TN SIMILARITY CONFIG")
    print("=" * 60)
    print("\nConstraints:")
    print("  - n_layers = 2 (non-negotiable)")
    print("  - n_head >= 2 (non-negotiable)")
    print("  - d_model = 12  (preferred)")
    print("  - n_ctx >= 4 (minimum)")
    print("  - Time limit: 120 seconds")
    print()
    
    timeout = 120  # 2 minutes
    d_model = 12
    n_layers = 2
    n_head = 2
    
    # Test different n_ctx values
    test_configs = [
        4,   # Minimum
        5,
        6,
        7,
        8,
    ]
    
    results = []
    
    for n_ctx in test_configs:
        elapsed, success = benchmark_config(n_ctx, d_model, n_layers, n_head, timeout)
        
        if not success:
            print(f"\nStopping: Config failed")
            break
        
        results.append((n_ctx, elapsed))
        
        if elapsed > timeout:
            print(f"\nStopping: Exceeded timeout ({elapsed:.2f}s > {timeout}s)")
            break
    
    # Find best config
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    valid_configs = [(n_ctx, t) for n_ctx, t in results if t < timeout]
    
    if not valid_configs:
        print("\nNo configs completed within timeout!")
        print("Falling back to smallest config...")
        best_n_ctx = 4
        best_time = results[0][1] if results else None
    else:
        best_n_ctx, best_time = max(valid_configs, key=lambda x: x[0])
        
        print("\nValid configs (within 120s timeout):")
        for n_ctx, t in valid_configs:
            marker = " ← BEST" if n_ctx == best_n_ctx else ""
            print(f"  n_ctx={n_ctx}: {t:.2f}s{marker}")
    
    print(f"\n{'=' * 60}")
    print("OPTIMAL CONFIG")
    print("=" * 60)
    print(f"  vocab_size: 32")
    print(f"  n_ctx: {best_n_ctx}")
    print(f"  d_model: {d_model}")
    print(f"  n_head: {n_head}")
    print(f"  n_layers: {n_layers}")
    print(f"  attn_type: bilinear")
    print(f"  attn_scale: 0.5")
    if best_time:
        print(f"\n  Expected time: ~{best_time:.1f}s")
    print()
    
    # Write to YAML
    optimal_cfg = {
        "name": "minimal_tn_compatible",
        "seed": 42,
        "model": {
            "vocab_size": 32,
            "n_ctx": best_n_ctx,
            "d_model": d_model,
            "n_head": n_head,
            "n_layers": n_layers,
            "attn_type": "bilinear",
            "attn_scale": 0.5,
            "rope_base": 10000,
            "norm_type": "none",
            "norm_places": [],
            "use_rmsnorm_qk": False,
            "use_bias_qk": False,
        },
        "init": {
            "init_type": "normal",
            "std_embed": 0.02,
            "std_qkv": 0.02,
            "std_o": 0.01,
        }
    }
    
    config_path = Path(__file__).parent / "config_minimal_tn.yaml"
    with open(config_path, "w") as f:
        yaml.dump(optimal_cfg, f, default_flow_style=False, sort_keys=False)
    
    print(f"Saved optimal config to: {config_path}")
    
    return optimal_cfg

if __name__ == "__main__":
    find_optimal_config()
