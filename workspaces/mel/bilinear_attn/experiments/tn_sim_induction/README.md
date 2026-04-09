# TN Similarity Induction Experiments

Compute tensor network cosine similarity heatmaps between checkpoint pairs from induction head training runs.

## Usage

```bash
# From bilinear_attn directory

# Basic: compute all pairs for checkpoints every 1000 steps
python -m experiments.tn_sim_induction.heatmap \
    --checkpoint-dir experiments/induction_heads/runs/<run>/checkpoints \
    --checkpoint-every 1000

# Window mode: only compute similarity with next 5 checkpoints
python -m experiments.tn_sim_induction.heatmap \
    --checkpoint-dir experiments/induction_heads/runs/<run>/checkpoints \
    --checkpoint-every 1000 \
    --window 5

# Later: decrease checkpoint_every to add more checkpoints (only computes new pairs)
python -m experiments.tn_sim_induction.heatmap \
    --checkpoint-dir experiments/induction_heads/runs/<run>/checkpoints \
    --checkpoint-every 500

# Later: increase window to compute more pairs (only computes new pairs)
python -m experiments.tn_sim_induction.heatmap \
    --checkpoint-dir experiments/induction_heads/runs/<run>/checkpoints \
    --checkpoint-every 500 \
    --window 10
```

## API

```python
from experiments.tn_sim_induction.heatmap import generate_heatmap

# Generate heatmap from checkpoint directory
sim_matrix, steps = generate_heatmap(
    checkpoint_dir="runs/2024-01-01_12-00-00/checkpoints",
    output_dir="results/",      # optional, defaults to checkpoint parent
    device="cuda",              # optional, auto-detects
    checkpoint_every=1000,      # only use checkpoints at this interval
    window=5,                   # only compute pairs within this index distance
    show=True,                  # display plot
)
```

## Features

- **Incremental caching**: Existing results are loaded from `tn_similarity_data.npz` and only missing pairs are computed
- **Window mode**: Limit computation to `m` nearest neighbors with `--window m`
- **Checkpoint filtering**: Select checkpoints at intervals with `--checkpoint-every`
- **Flexible refinement**: Decrease `checkpoint_every` or increase `window` later to add more data

## Output

- `tn_similarity_heatmap.png` - Visualization (gray = not computed)
- `tn_similarity_data.npz` - Raw data with `sim_matrix` and `steps` arrays

## Notes

- Uses **TN cosine similarity** from `tn_sim.cosine_similarity`
- **Self-similarity is assumed to be 1.0** (not computed)
- Only works with TN-compatible models (`norm_type='none'`, `norm_places=[]`)
- Computation is expensive (~10-60s per pair depending on model size)
