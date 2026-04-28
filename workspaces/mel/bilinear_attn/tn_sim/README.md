# TN Similarity for Bilinear Attention Models

This module provides exact tensor network (TN) similarity computation for bilinear attention models using the main codebase's algorithm (`src/components/similarity.py`).
It adds batching support for better memory efficiency. I shall add an accelerator for batching soon.

## Usage

```python
from models import AttentionLM
from tn_sim import cosine_similarity

# Create TN-compatible models (no normalization!)
cfg = {
    "model": {
        "vocab_size": 32,
        "n_ctx": 8,
        "d_model": 16,
        "n_head": 2,
        "n_layers": 1,
        "attn_type": "bilinear",  # or "quadratic"
        "norm_type": "none",      # REQUIRED
        "norm_places": [],        # REQUIRED
        "use_rmsnorm_qk": False,  # REQUIRED
    }
}

model_A = AttentionLM.from_config(cfg)
model_B = AttentionLM.from_config(cfg)

# Compute similarity
sim = cosine_similarity(model_A, model_B)
print(f"Similarity: {sim:.6f}")
```

## API

- **`cosine_similarity(model_A, model_B)`** - Returns scalar similarity in [-1, 1]
- **`compute_tn_similarity(model_A, model_B)`** - Returns full `State` object with second moments
- **`inner_product(model_A, model_B)`** - Returns unnormalized inner product
- **`self_similarity(model)`** - Convenience function (should return 1.0)
- **`mc_similarity_gaussian(model_A, model_B, ...)`** - Monte Carlo baseline for validation

## Requirements

### Model Configuration

**CRITICAL:** Only models with the following configuration are supported:

```yaml
model:
  norm_type: none          # No normalization layers
  norm_places: []          # No normalization anywhere
  use_rmsnorm_qk: false    # No RMSNorm on Q/K
  attn_type: bilinear      # or "quadratic" (NOT "softmax")
```

Any model with normalization will raise a `ValueError`.

### Why These Restrictions?

The main codebase's TN similarity assumes:
1. **Gaussian inputs** - Exact computation via Isserlis theorem
2. **Polynomial operations** - Normalization (RMSNorm, LayerNorm) is non-polynomial
3. **No softmax** - Softmax is non-polynomial

## Performance Characteristics

### Benchmark Results (Minimal Config)

Tested on: `n_ctx=4, d_model=8, n_layers=2, n_head=2`

| Metric | Value |
|--------|-------|
| **Time** | 47 seconds |
| **Memory** | 2.2 GB |
| **Self-similarity** | 1.0 (exact) |

### Scaling

The main codebase's algorithm has **exponential complexity** in the number of layers due to term decomposition:

- **1 layer:** ~2 terms (residual + active)
- **2 layers:** ~4 terms (2² combinations)
- **3 layers:** ~8 terms (2³ combinations)
- **N layers:** ~2^N terms

Each attention layer also has **945 Wick matchings** to compute.

### Practical Limits

Based on testing:

| Model Size | Time | Memory | Practical? |
|------------|------|--------|------------|
| n_ctx=4, d_model=8, 1 layer | ~10s | ~2 GB | ✅ Yes |
| n_ctx=4, d_model=8, 2 layers | ~47s | ~8 GB | ⚠️ Marginal |
| n_ctx=8, d_model=16, 2 layers | ~5-10 min | ~30 GB | ❌ No |
| n_ctx=16, d_model=32, 2 layers | Hours | >100 GB | ❌ No |

**Recommendation:** Only use for very small models (n_ctx ≤ 4, d_model ≤ 8, n_layers ≤ 2).

## Comparison: Old vs New Implementation

| Aspect | Old (Custom) | New (Main Codebase) |
|--------|--------------|---------------------|
| **Speed** | ~1-2 seconds | ~47 seconds (23-47× slower) |
| **Memory** | ~500 MB | ~8 GB (16× more) |
| **Accuracy** | Approximate (ignores residual cross-terms) | Exact (all terms) |
| **Normalization** | Supported (approximate) | Not supported |
| **Residual handling** | Ignored cross-terms | Full term decomposition |
| **Method** | Gram chaining with precomputed RoPE | Full TN contraction with Wick matchings |

## Validation

Run the test suite:

```bash
# Activate venv first!
.venv\Scripts\Activate.ps1

# Quick validation
python -c "from tn_sim.test_similarity import run_quick_validation; run_quick_validation()"

# Full test suite (requires pytest)
pytest tn_sim/test_similarity.py -v

# Minimal benchmark
python debug_minimal.py
```

### Expected Results

- **Self-similarity:** Exactly 1.0 (within 1e-6)
- **MC comparison:** TN and MC should agree within ~30% for small models
- **Symmetry:** `sim(A, B) == sim(B, A)`

## Files

### Core Implementation
- `models/components/` - Component-compatible wrappers
  - `embedding.py` - Embedding as Linear component
  - `attention.py` - BilinearAttention as Component
  - `model.py` - AttentionLM as Model
- `tn_sim/similarity.py` - Main API endpoint
- `tn_sim/mc_similarity.py` - Monte Carlo baseline (kept for validation)

### Testing & Validation
- `tn_sim/test_similarity.py` - Comprehensive test suite
- `tn_sim/compare.py` - Compare TN vs MC across checkpoints
- `debug_minimal.py` - Minimal benchmark with memory/time tracking

### Configuration
- `tn_sim/config_minimal_tn.yaml` - Minimal TN-compatible config

## Migration Notes

### What Was Deleted
- `tn_sim/tn_similarity.py` (19 KB) - Custom Gram chaining implementation
- `tn_sim/tn_similarity_quimb.py` (12 KB) - Alternative quimb implementation
- `tn_sim/test_tn_similarity.py` (31 KB) - Tests for custom implementation
- `tn_sim/train.py` (8 KB) - Training script for validation
- `tn_sim/config_tiny.yaml` - Old config (had normalization)
- `tn_sim/runs/` - Old validation runs

### What Was Kept
- `tn_sim/mc_similarity.py` - Still needed for validation
- `tn_sim/compare.py` - Updated to use new endpoint

### What Was Added
- `models/components/` - Component-compatible layers (3 files)
- `tn_sim/similarity.py` - New API endpoint
- `tn_sim/test_similarity.py` - New test suite
- `tn_sim/config_minimal_tn.yaml` - Minimal TN-compatible config
- This README

## Future Work

Potential optimizations (not implemented):
1. **Sparse contractions** - Skip zero terms
2. **Batch processing** - Compute multiple pairs in parallel
3. **GPU acceleration** - Move contractions to GPU
4. **Approximate methods** - Truncate low-magnitude terms
5. **Custom RoPE handling** - Precompute RoPE matrices (like old implementation)

However, these would require significant changes to the main codebase's algorithm.

## Contact

For issues or questions about this migration, see the main codebase documentation at `src/components/SIMILARITY.md`.
