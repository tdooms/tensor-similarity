# Tensor Mars — Polynomial Attention & Induction Heads

## Project Goal

We're investigating whether **polynomial (non-softmax) attention** can learn **induction heads** — the key mechanism that enables in-context learning in transformers. The motivation is tensor network compatibility: softmax attention is incompatible with tensor networks due to its non-polynomial nature, but polynomial attention (quadratic, bilinear, cubic) could be made compatible while still being expressive enough to learn important circuits.

## Core Research Question

**Can polynomial attention learn induction heads at scale on real data?**

- On a **toy task** (repeated random sequences), bilinear attention learns induction perfectly — but ONLY with proper Q,K weight norm control.
- On **real data** (SimpleStories, OpenWebText), we haven't yet observed strong induction formation in our 2L attention-only models.
- We're now trying to replicate **Hoogland et al.** results to establish a baseline showing softmax induction on similar architecture/data, then swap in polynomial attention.

## Key Technical Finding

**Without norm control, polynomial attention cannot learn induction** (stuck at random ~3% accuracy on toy task). Six techniques fix this (all achieve 100% on toy induction):

| Technique | Inference cost | Type |
|-----------|---------------|------|
| BatchNorm on Q,K projections | Static | Activation norm |
| Spectral norm on Q,K weights | Static | Weight constraint |
| Weight standardization | Static | Weight constraint |
| muP initialization (std=1/√d_head) | Free | Init only |
| Orthogonal initialization | Free | Init only |
| Orthogonal regularization (‖W^TW-I‖² loss) | Training only | Regularizer |

Things that DON'T help: cubic/trilinear attention, LayerScale, Fixup, Hadamard mixing, strong weight decay.

**Key insight**: The problem is Q,K weight norm drift during training, not polynomial degree or output scaling.

## Attention Types

- **quadratic**: `(q·k/d)²` — single QK pair, squared, always non-negative
- **bilinear**: `(q1·k1)*(q2·k2)/d²` — two QK pairs, product (can be negative)
- **cubic**: `(QK^T/√d_k)³/√N` — preserves sign, sequence-length normalized
- **softmax**: standard `softmax(q·k/√d)` with causal mask (baseline)

## Architecture

All experiments use **attention-only** models (no MLP):
- `embed → n_layers × Attention → LayerNorm → unembed`
- RoPE positional encoding
- Muon optimizer (for attention weight matrices) + AdamW (for embeddings/biases/norms)
- Linear warmup + cosine decay LR schedule
- Pre-unembed LayerNorm is essential for good performance

## Experiments Run So Far

### 1. Toy Induction Task (COMPLETED)
- 1L model, d_model=64, 4 heads, random repeated sequences
- Established that norm control is necessary and sufficient for polynomial attention
- All 6 techniques above give 100% accuracy

### 2. SimpleStories Training (various configs)
- 2L/4L attention-only models, d_model=64-768, SimpleStories data
- SimpleStories has ~600M tokens (2.1M stories, avg 284 tokens)
- Bilinear with batchnorm + ortho init trains well (val loss ~2.1-2.4)
- But **toy induction accuracy stays near 0%** throughout training
- Induction scores are consistently negative (second half loss > first half)

### 3. d_model=256, 5B Token Run (IN PROGRESS)
- 2L attn-only, d=256, 8 heads, n_ctx=1024
- Two models: bilinear+batchnorm and softmax+QK_RMSNorm
- Running sequentially on single 16GB GPU
- 76,293 steps total, ~1.7h each
- Bilinear currently at step ~67k, still no induction signal

### 4. Hoogland Replication (NEXT)
- Replicate Hoogland et al. exactly: 2L attn-only, d=256, 8 heads, standard softmax
- DSIR-filtered Pile (streaming), GPT-2 tokenizer truncated to vocab=5000
- They report induction onset at steps 6.5k-17k
- Self-contained script: `train_hoogland_standalone.py`
- This establishes the baseline — if softmax learns induction here, we can try swapping in polynomial attention on the same setup

## File Structure

```
workspaces/logan/
├── train_hoogland_standalone.py    # Self-contained Hoogland replication (PRIORITY)
├── train_d256_5B.py                # d256 5B token training (bilinear/softmax)
├── train_induction_static_norm.py  # BilinearBatchNorm class definition
├── train_induction_toy.py          # Toy induction task (sweep of techniques)
├── train_4layer.py                 # 4L attn-only bilinear d=768
├── train_2layer_transformer.py     # 2L transformer with bilinear attn + MLP
├── train_fineweb_2layer.py         # OpenWebText training
├── train_softmax_128.py            # Softmax n_ctx=128 experiment
├── measure_induction_opportunity.py # Dataset induction analysis
├── cached_tokens/                  # Cached tokenized data
│   ├── train_perstory.pt           # SimpleStories train (n_ctx=512)
│   ├── test_perstory.pt            # SimpleStories test (n_ctx=512)
│   └── dsir_pile_val.pt            # DSIR Pile validation (if cached)
├── runs/                           # Training run outputs
│   └── {timestamp}_{name}/
│       ├── metrics.jsonl           # Training metrics
│       └── checkpoints/            # Model checkpoints
└── results/                        # Analysis outputs

workspaces/mel/bilinear_attn/      # Shared model/training codebase
├── models/
│   ├── transformer.py              # AttentionLM with ATTN_REGISTRY
│   └── attention_kernels/
│       ├── softmax.py              # SoftmaxAttention
│       ├── bilinear.py             # QuadraticAttention
│       ├── bilinear_2qk.py         # BilinearAttention (two QK pairs)
│       ├── cubic.py                # CubicAttention
│       └── rotary.py               # RoPE implementation
├── train/
│   ├── trainer.py                  # Trainer class
│   ├── losses.py                   # Loss functions
│   ├── optim.py                    # Muon + AdamW optimizer setup
│   └── eval.py                     # Evaluation
└── data/                           # Data loading utilities
```

## What to Run Next

1. **Hoogland replication** (`train_hoogland_standalone.py`): Confirm that softmax learns induction on DSIR-filtered Pile with this architecture. Expected onset at 6.5k-17k steps.

2. **If induction confirmed**: Replace softmax with bilinear+batchnorm on the SAME data/setup. Does polynomial attention also learn induction on Pile data?

3. **Modifications to try**:
   - Larger d_model (512, 768)
   - Smaller n_ctx (512) — Hoogland found faster induction with shorter context
   - QK RMSNorm on softmax baseline
   - Different norm control techniques for polynomial attention

## Dependencies

```
torch>=2.0
einops
muon (pip install git+https://github.com/KellerJordan/Muon)
datasets
transformers
tqdm
numpy
```

## GPU Requirements

- 16GB GPU (T4/A4000): batch_size=64, n_ctx=1024 for d=256 models (one at a time)
- 24GB+ GPU (A5000/A6000/A100): Can run larger models or higher batch sizes
- torch.compile helps speed (~2x) but uses more memory
- Two compiled d=256 models can't fit on 16GB simultaneously
