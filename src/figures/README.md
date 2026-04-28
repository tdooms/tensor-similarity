# Paper figures

Figure modules in this folder are the public paper-facing entrypoints. They are
the canonical figure pipelines. Each family should stay within the three-stage
shape `train -> prepare -> plot`, with at most three experiment-specific files.

## Workflow

```bash
uv run figures train seed-convergence
uv run figures prepare seed-convergence
uv run figures plot seed-convergence
```

Available figure families:

- `seed-convergence`
- `curriculum-shift`
- `checkpoint-similarity`
- `subset-training`

Keep shared design and export helpers in `style.py`. Experiment-specific logic
should live in per-experiment folders under `src/figures/`, with only
`train.py`, `prepare.py`, and `plot.py` inside each folder.

Repo-local inputs live in `_downloads/`, prepared tabular caches live in
`artifacts/cache/`, and final renders live in `artifacts/figures/`. Official
pipelines should prefer `.safetensors`, `.feather`, and `.json`, using older
formats only when importing legacy data.
