# Paper experiments

Three-stage pipeline: **train** → **experiment** → **plot**.

```
src/paper/
├── convergence/
│   ├── train.py        # Train models across seeds, save checkpoints
│   ├── experiment.py   # Compute similarities between checkpoints
│   └── plot.py         # Read results, produce figures
├── perturbation/
│   ├── train.py        # Train base model, then curriculum (add/remove digits)
│   ├── experiment.py   # Compute pairwise similarity matrix across checkpoints
│   └── plot.py         # Heatmap + similarity vs accuracy plots
└── artifacts/          # All intermediate results (gitignored)
    ├── convergence/
    └── perturbation/
```

## Running

```bash
# Each step reads from and writes to artifacts/
python -m src.paper.convergence.train
python -m src.paper.convergence.experiment
python -m src.paper.convergence.plot

python -m src.paper.perturbation.train
python -m src.paper.perturbation.experiment
python -m src.paper.perturbation.plot
```

## Conventions

- `train.py` saves model checkpoints and training history to `artifacts/`
- `experiment.py` loads checkpoints, computes similarities, saves results to `artifacts/`
- `plot.py` loads results, produces figures — no computation
- All artifacts are torch tensors or pickle files
