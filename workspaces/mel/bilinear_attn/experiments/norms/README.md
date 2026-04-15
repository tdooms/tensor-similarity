# Normalization Experiments

This directory contains experiments for testing different normalization strategies for Q/K vectors in attention mechanisms.

## Summary of Existing Normalization Approaches

### Analysis of Codebase

**No existing implementation** of learned per-head temperature scaling (α_q, α_k) was found. The codebase has two distinct normalization systems:

### 1. Residual Stream Norms (existing)
- **Location**: Applied to residual stream activations in `models/transformer.py`
- **Control**: `norm_type` + `norm_places` in config
- **Types**: 
  - `none`: No normalization
  - `rmsnorm`: Standard RMSNorm
  - `layernorm`: LayerNorm
  - `maxrmsnorm`: Scale by largest per-token RMS in sequence
  - `causal_maxrmsnorm`: Causal variant (max up to current position)
- **Places**: `post_embed`, `pre_layer`, `pre_unembed`

### 2. Q/K Vector Norms (existing + new variants)
- **Location**: Applied to Q/K vectors before RoPE in attention kernels
- **Control**: `use_rmsnorm_qk` (old) → `qk_norm_type` (new)

#### Existing:
- **`none`**: No normalization (Identity) - baseline
- **`rmsnorm`**: Standard RMSNorm per token/head dimension
  - Dynamic normalization (per-token)
  - Applied in `models/attention_kernels/{bilinear,softmax,quadratic}.py`

#### New Variants (this experiment):
- **`alpha_head`**: Learned per-head temperature scaling
  - **Formula**: q' = α_q · q, k' = α_k · k
  - **Equivalent to**: attn = (q k^T) · (α_q α_k)
  - **Properties**:
    - Learned scalars per head (not per token/batch/position)
    - Static temperature (not dynamic normalization)
    - 2 × n_head learnable parameters
  - **Implementation**: `experiments/norms/qk_norms.py::AlphaHeadNorm`

## Key Difference

The proposed α-scaling is **fundamentally different** from existing approaches:
- **Existing RMSNorm**: Dynamic, per-token normalization based on activation statistics
- **Alpha-head**: Static, learned temperature per head (like inverse softmax temperature)

## Experiment Infrastructure

### Files Created

```
experiments/norms/
├── __init__.py
├── README.md (this file)
├── qk_norms.py              # AlphaHeadNorm + QKNormWrapper
├── attention_kernels.py      # Modified attention with configurable Q/K norms
├── model.py                  # AttentionLMNorm wrapper
├── run.py                    # Training script
├── compare.py                # Results comparison script
└── configs/
    ├── alpha_head.yaml              # Alpha-head (bilinear)
    ├── alpha_head_quadratic.yaml    # Alpha-head (quadratic)
    ├── rmsnorm_qk.yaml              # RMSNorm baseline
    └── no_qk_norm.yaml              # No Q/K norm baseline
```

## Usage

### Run Single Experiment

```bash
# From bilinear_attn directory
python -m experiments.norms.run \
    --config experiments/norms/configs/alpha_head.yaml \
    --wandb

# With custom settings
python -m experiments.norms.run \
    --config experiments/norms/configs/alpha_head.yaml \
    --n-train 100000 \
    --seq-len 128 \
    --checkpoint-every 500
```

### Compare Results

```bash
python -m experiments.norms.compare \
    --run-dirs experiments/norms/runs/*
```

## Configuration

Add to your config YAML:

```yaml
model:
  # Residual stream normalization (existing)
  norm_type: rmsnorm
  norm_places: [pre_layer, pre_unembed]
  
  # Q/K vector normalization (NEW)
  qk_norm_type: alpha_head  # or 'none', 'rmsnorm'
  alpha_init: 1.0           # initial value for α parameters
  
  # Other params
  attn_type: bilinear
  n_head: 8
  # ...
```

## Experiments to Run

1. **Baseline**: `no_qk_norm.yaml` - No Q/K normalization
2. **Existing**: `rmsnorm_qk.yaml` - Dynamic RMSNorm on Q/K
3. **New**: `alpha_head.yaml` - Learned per-head temperature (bilinear)
4. **New**: `alpha_head_quadratic.yaml` - Learned per-head temperature (quadratic)

## Expected Findings

The alpha-head approach may:
- Learn different temperatures per head (specialization)
- Provide more stable training than dynamic normalization
- Act as learned attention temperature control
- Show different behavior across attention types (bilinear vs quadratic vs softmax)
