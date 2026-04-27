# Grokking summary figure — self-contained reproduction bundle

Trains a small bilinear model on modular addition (P=113) and produces
`selected_checkpoints_summary.png`: a 5-panel summary with train/val accuracy,
train/val loss, pairwise TN-similarity matrix, frequency marginals, and
Tucker effective ranks.

## Layout

```
.
├── train.py                  # train + checkpoint + write results/*.{json,pkl,npy,npz}
├── plot.py                   # read results/, render selected_checkpoints_summary.png
├── tensor_diff_analysis.py   # tensor utilities used by plot.py
├── core/                     # model, dataset, frequency, similarity, metrics
└── results/                  # populated by train.py, consumed by plot.py
```

## Usage

```bash
pip install -r requirements.txt
python train.py    # ~minutes-to-an-hour depending on hardware
python plot.py     # writes results/selected_checkpoints_summary.png
```

`train.py` writes:
- `config.json`, `training_history.json`, `checkpoints.pkl`
- `tn_similarity.npy`, `act_similarity.npy`, `js_divergence.npy`, `checkpoint_steps.npy`
- `freq_heatmaps.npz` (heatmaps + marginals per checkpoint)

`plot.py` consumes those and writes `selected_checkpoints_summary.png` plus
two secondary figures (`tensor_structure_evolution.png`,
`pairwise_tensor_comparisons.png`).

Tweak `ExperimentConfig` at the top of `train.py` to change P, d_hidden,
weight decay, total steps, seed, etc.
