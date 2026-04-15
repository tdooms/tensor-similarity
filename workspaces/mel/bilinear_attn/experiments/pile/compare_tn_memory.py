#!/usr/bin/env python3
"""Compare memory allocation between tn_sim wrapper and src implementation."""

import torch
import sys
import os
import psutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import AttentionLM
from models.components.model import AttentionLMComponent
from tn_sim import cosine_similarity as tn_cosine_similarity_wrapper
from src.components.similarity import similarity as tn_similarity_direct
from tn_sim.similarity import _cosine_from_state


def get_max_memory_mb():
    """Get max memory used in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


# Use small dimensions for testing
vocab_size = 100
n_ctx = 8
d_model = 8
n_head = 2
n_layers = 2

print(f"Creating small model for memory comparison:")
print(f"  vocab_size={vocab_size}, n_ctx={n_ctx}, d_model={d_model}, n_head={n_head}, n_layers={n_layers}")

# Create model with norm_type='none' for TN compatibility
cfg = {
    "model": {
        "vocab_size": vocab_size,
        "n_ctx": n_ctx,
        "d_model": d_model,
        "n_head": n_head,
        "n_layers": n_layers,
        "attn_type": "bilinear",
        "norm_type": "none",
        "norm_places": [],
    },
    "init": {}
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create two random models
model1 = AttentionLM.from_config(cfg)
model2 = AttentionLM.from_config(cfg)

print("Models created successfully")

# Convert to AttentionLMComponent
print("Converting to AttentionLMComponent...")
comp1 = AttentionLMComponent.from_trained_model(model1)
comp2 = AttentionLMComponent.from_trained_model(model2)

comp1 = comp1.to(device)
comp2 = comp2.to(device)

# Test 1: tn_sim wrapper
print("\n" + "="*60)
print("TEST 1: tn_sim wrapper")
print("="*60)
torch.cuda.empty_cache() if torch.cuda.is_available() else None
initial_mem = get_max_memory_mb()
print(f"Initial memory: {initial_mem:.2f} MB")

try:
    with torch.no_grad():
        sim_wrapper = tn_cosine_similarity_wrapper(comp1, comp2, device=device, dtype=torch.float64)
    max_mem_wrapper = get_max_memory_mb()
    mem_used_wrapper = max_mem_wrapper - initial_mem
    print(f"Similarity: {sim_wrapper:.6f}")
    print(f"Max memory used: {mem_used_wrapper:.2f} MB")
    print(f"Total max memory: {max_mem_wrapper:.2f} MB")
except Exception as e:
    print(f"Error: {e}")
    mem_used_wrapper = None
    max_mem_wrapper = None

# Test 2: src implementation direct
print("\n" + "="*60)
print("TEST 2: src implementation direct")
print("="*60)
torch.cuda.empty_cache() if torch.cuda.is_available() else None
initial_mem = get_max_memory_mb()
print(f"Initial memory: {initial_mem:.2f} MB")

try:
    with torch.no_grad():
        state_direct = tn_similarity_direct(comp1, comp2)
        sim_direct = _cosine_from_state(state_direct)
    max_mem_direct = get_max_memory_mb()
    mem_used_direct = max_mem_direct - initial_mem
    print(f"Similarity: {sim_direct:.6f}")
    print(f"Max memory used: {mem_used_direct:.2f} MB")
    print(f"Total max memory: {max_mem_direct:.2f} MB")
except Exception as e:
    print(f"Error: {e}")
    mem_used_direct = None
    max_mem_direct = None

# Test 3: src implementation with dtype conversion (like wrapper does)
print("\n" + "="*60)
print("TEST 3: src implementation with dtype=float64")
print("="*60)
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Recreate components to test fresh
comp1_test = AttentionLMComponent.from_trained_model(model1)
comp2_test = AttentionLMComponent.from_trained_model(model2)
comp1_test = comp1_test.to(device=device, dtype=torch.float64)
comp2_test = comp2_test.to(device=device, dtype=torch.float64)

initial_mem = get_max_memory_mb()
print(f"Initial memory: {initial_mem:.2f} MB")

try:
    with torch.no_grad():
        state_dtype = tn_similarity_direct(comp1_test, comp2_test)
        sim_dtype = _cosine_from_state(state_dtype)
    max_mem_dtype = get_max_memory_mb()
    mem_used_dtype = max_mem_dtype - initial_mem
    print(f"Similarity: {sim_dtype:.6f}")
    print(f"Max memory used: {mem_used_dtype:.2f} MB")
    print(f"Total max memory: {max_mem_dtype:.2f} MB")
except Exception as e:
    print(f"Error: {e}")
    mem_used_dtype = None
    max_mem_dtype = None

# Test 4: Wrapper but skip conversion (pass components directly)
print("\n" + "="*60)
print("TEST 4: tn_sim wrapper with components (skip conversion)")
print("="*60)
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Use the same components from test 2 (already on device)
initial_mem = get_max_memory_mb()
print(f"Initial memory: {initial_mem:.2f} MB")

try:
    # Call the internal compute_tn_similarity directly with components
    from tn_sim.similarity import compute_tn_similarity
    with torch.no_grad():
        state_skip = compute_tn_similarity(comp1, comp2, device=device, dtype=torch.float64)
        sim_skip = _cosine_from_state(state_skip)
    max_mem_skip = get_max_memory_mb()
    mem_used_skip = max_mem_skip - initial_mem
    print(f"Similarity: {sim_skip:.6f}")
    print(f"Max memory used: {mem_used_skip:.2f} MB")
    print(f"Total max memory: {max_mem_skip:.2f} MB")
except Exception as e:
    print(f"Error: {e}")
    mem_used_skip = None
    max_mem_skip = None

# Comparison
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
if mem_used_wrapper is not None and mem_used_direct is not None:
    ratio = mem_used_wrapper / mem_used_direct if mem_used_direct > 0 else float('inf')
    print(f"tn_sim wrapper:    {mem_used_wrapper:.2f} MB")
    print(f"src direct:        {mem_used_direct:.2f} MB")
    print(f"src with float64:  {mem_used_dtype:.2f} MB" if mem_used_dtype else "src with float64:  N/A")
    print(f"wrapper skip conv: {mem_used_skip:.2f} MB" if mem_used_skip else "wrapper skip conv:  N/A")
    print(f"Ratio (wrapper/src): {ratio:.2f}x")
    
    if abs(sim_wrapper - sim_direct) > 1e-6:
        print(f"WARNING: Similarities differ! wrapper={sim_wrapper:.6f}, direct={sim_direct:.6f}")
    else:
        print(f"Similarities match: {sim_wrapper:.6f}")
