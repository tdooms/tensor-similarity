# Attention Head Inspection for Bilinear Attention

... the metrics are BS, im only using this to visualise the attention patterns

## Overview

This directory contains tools for inspecting attention heads in bilinear attention models using PyTorch hooks. The inspection tool visualizes attention patterns on repeated sequences to detect induction heads and analyze circuit behavior.

## Bilinear Attention Structure

For a model with **2 layers** and **n_head=2**, there are:
- **4 attention heads total** (2 layers × 2 heads/layer)
- **8 attention circuits** (4 heads × 2 circuits/head)

Each bilinear attention head has **two independent circuits**:
1. **Q1-K1 circuit**: First query-key pair (scores1)
2. **Q2-K2 circuit**: Second query-key pair (scores2)
3. **Combined pattern**: Element-wise product Q1-K1 × Q2-K2, then masked

**Important**: `n_head` is the number of heads per layer, NOT the number of circuits. Each head has 2 circuits but counts as 1 head.

## Usage

### Architecture-Agnostic Inspection

The tool automatically detects the model architecture from the checkpoint config:

```bash
# Use final.pt checkpoint
python inspect_heads.py --checkpoint_dir runs/my_run/checkpoints

# Use specific step
python inspect_heads.py --checkpoint_dir runs/my_run/checkpoints --step 3000
```

Works with:
- **BilinearAttention** or **QuadraticAttention**
- Any number of layers (1, 2, n)
- With or without bias (`use_bias_qk`)
- Variable-length sequences (prevents RoPE shortcuts)

### Arguments

- `--checkpoint_dir`: Directory containing checkpoints (default: `runs/induction_checkpoints/checkpoints`)
- `--config`: Path to config file (default: `runs/induction_checkpoints/config.yaml`)
- `--step`: Checkpoint step to load (default: 3000)
- `--n_samples`: Number of sequences to generate (default: 16)
- `--repeat_len`: Length of repeating pattern (default: 3)
- `--output_dir`: Directory to save visualizations (default: `attention_visualizations`)
- `--visualize_circuits`: Also visualize Q1-K1 and Q2-K2 circuits separately

## Variable-Length Repeated Sequences

**Important**: To prevent RoPE from learning position-based shortcuts, we use **variable-length** subsequences.

### The RoPE Problem

With fixed-length sequences like `[A B C A B C]`, RoPE (Rotary Position Embeddings) can learn to predict the sequence length and use absolute positional information instead of true pattern matching. This makes the task trivial and doesn't test real induction behavior.

### Our Solution

Generate subsequences of varying lengths from 2 to n, then repeat them:
```
Length 2: [A B A B 0 0 0 0 ...]  (padded)
Length 3: [A B C A B C 0 0 ...]
Length 4: [A B C D A B C D ...]
```

This forces the model to learn true **pattern matching** rather than position-based prediction.

### Induction Behavior

An **induction head** should attend to the token that came AFTER the previous occurrence. For `[A B C A B C]` at position 3 (second A), it should attend to position 1 (B, which followed the first A).

## Induction Head Metrics (from Anthropic Paper)

The tool computes three metrics for each head following [Olsson et al. 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/):

### 1. Prefix Matching Score
Measures attention to the matching token in the first half of the sequence.
For `[A B C A B C]`, at position 3 (second A), measures attention to position 0 (first A).

### 2. Induction Score
Measures attention to the token that came AFTER the previous occurrence.
For `[A B C A B C]`, at position 3 (second A), measures attention to position 1 (B, which came after first A).
**This is the key metric for detecting induction heads.**

### 3. Copying Score
Measures attention to the immediately previous token (simple copying behavior).

Example output:
```
Layer    Head   Prefix     Induction    Copying
0        0       -0.0474       0.0979    -0.1337
0        1       -0.0697      -0.1575    -0.1328
1        0      -40.1740    -438.7524   -21.8827
1        1        0.0000       0.0000     0.0000

Strongest induction head: Layer 0, Head 0
  Induction score: 0.0979
```

## Implementation Details

### PyTorch Hooks

The tool uses `register_forward_hook()` to capture intermediate activations:

```python
def hook(module, input, output):
    # Capture Q, K, V projections
    q1 = module.q1(x)
    k1 = module.k1(x)
    # ... compute attention patterns
```

### Attention Pattern Computation

For each head and circuit (from `BilinearAttention.forward`):

1. **Q1-K1 scores**: `scores1 = Q1 @ K1.T` (shape: batch, n_head, n_ctx, n_ctx)
2. **Q2-K2 scores**: `scores2 = Q2 @ K2.T`
3. **Combined pattern**: `pattern = (scores1 * scores2) / d_head^2`
4. **Apply causal mask**: `pattern = pattern * causal_mask`

The causal mask is always applied (registered as a buffer in BilinearAttention.__init__).

### No TransformerLens Dependency

This implementation uses only PyTorch hooks and does not require TransformerLens, making it lightweight and easy to customize for bilinear attention.

## Visualization Outputs

### All Heads Overview

`attention_heads_step_3000.png` shows a grid of all attention heads:
- Rows: Layers
- Columns: Heads
- Each cell: Combined attention pattern (Q1-K1 × Q2-K2)

### Circuit Decomposition

`circuits_L{layer}_H{head}_step_{step}.png` shows three panels for each head:
1. Q1-K1 circuit pattern
2. Q2-K2 circuit pattern
3. Combined pattern (element-wise product)

## Example Results

From the induction heads checkpoint (step 3000, n_ctx=6, vocab=16):
- **Layer 0, Head 0**: Weak induction head (score: 0.0979)
  - Shows some attention to tokens after previous occurrences
  - Likely still learning the induction mechanism
- **Layer 0, Head 1**: No induction (score: -0.1575)
- **Layer 1, Head 0**: Anti-induction pattern (score: -438.75)
  - Large negative scores indicate attention away from induction positions
- **Layer 1, Head 1**: No activity (score: 0.00)

Note: The model may need more training or the task may be too simple (only 3 tokens to repeat).

## Extending the Tool

### Custom Sequence Patterns

Modify `generate_repeated_sequences()` to test different patterns:

```python
# Current: [A B C A B C ...]
# Could test: [A B A B ...], [A B C D A B C D ...], etc.
```

### Additional Metrics

Add custom metrics in `analyze_induction_score()`:
- Copying score (attend to same token)
- Previous token score (attend to token before current)
- Offset-k attention (attend k positions back)

### Per-Sample Visualization

Modify `visualize_attention_heads()` to show individual samples instead of batch average.

## Related Files

- `run.py`: Training script with variable-length sequence support
- `data.py`: RepeatedTokenDataset - generates variable-length `[seq][seq]` sequences
- `inspect_heads.py`: Architecture-agnostic inspection tool
- `../../models/attention_kernels/bilinear.py`: BilinearAttention implementation
- `../../models/attention_kernels/bilinear.py`: QuadraticAttention implementation

## Key Implementation Details

### BilinearAttention Structure
- `n_head` parameter: number of heads per layer
- Each head has `d_head = d_model // n_head` dimensions
- Two circuits per head: Q1-K1 and Q2-K2
- Combined via element-wise multiplication
- Causal mask always applied (line 39-40, 83 in bilinear.py)

### Variable-Length Sequence Format
The training data uses **variable-length** exact repetition to prevent RoPE shortcuts:
- Subsequence length: randomly chosen from 2 to max_seq_len
- Format: `[random_seq][random_seq][padding]`
- Example: `[A B C A B C 0 0 ...]` (length 3, padded to 2*max_seq_len)
- Model must learn pattern matching, not position-based prediction

**Why variable length?** Fixed-length sequences allow RoPE to learn absolute positions, making induction trivial. Variable lengths force true pattern matching.

## References

- **Induction Heads**: Olsson et al. (2022) "In-context Learning and Induction Heads" 
  https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/
  - Defines induction heads via prefix matching and copying behavior
  - Shows induction heads are key mechanism for in-context learning
  - Provides metrics for detecting induction heads in trained models

- **Bilinear Attention**: Custom attention mechanism with two independent Q-K circuits
  - Pattern = (Q1@K1.T) × (Q2@K2.T) / d_head^2
  - Allows for more expressive attention patterns than standard attention

- **PyTorch Hooks**: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
