# tensor-mars

Research code for tensor-network similarity experiments.

## Install

```bash
uv sync
```

## Figures workflow

Each family is single-button-press from a fresh clone:

```bash
uv run plot <family>      # everything end-to-end (prepare auto-triggers if cache missing)
uv run prepare <family>   # cache step only (auto-triggers train if checkpoints missing)
uv run train <family>     # training step (where applicable)
```

Available `<family>` values: `seed-convergence`, `curriculum-shift`, `checkpoint-similarity`, `subset-training`.

To force a recompute, delete the family's cache: `rm -rf artifacts/cache/<family>`.

The legacy combined CLI still works: `uv run figures {train,prepare,plot} <family>`.

## Figure Families

- `seed-convergence` — cross-seed MNIST convergence (similarity + accuracy)
- `curriculum-shift` — 8-stage curriculum trajectory + pairwise heatmap
- `checkpoint-similarity` — pairwise functional similarity across pretrained transformer checkpoints (pulls a log-linearly-spaced subsample of `melephant/2l-bilinear-attn-normalised-v2` on first run; caches to `_downloads/checkpoint-similarity/`).

  Subsample size is `N_STEPS=50` by default; tune via env. Each computed pair is appended to `_progress.jsonl` immediately, so an interrupted run resumes without recomputing. Rough budget on a single GPU after the first warm precompile: `~3.5 s per pair`, so `N×(N-1)/2` pairs at `N=50→~30 min`, `N=75→~2.5 h`, `N=100→~5 h`.

  ```bash
  N_STEPS=75 uv run prepare checkpoint-similarity     # ~2.5 h
  uv run plot checkpoint-similarity                    # ~30 s once cache is built
  ```

- `subset-training` — Laurence-derived MNIST subset training across seeds

## Layout

```text
src/                  installed library code
  components/         reusable model + similarity primitives (TN cosine etc.)
  datasets/           dataset loaders
  figures/            per-family train/prepare/plot + shared style + CLIs
  models/             shared model definitions (DeepMLP, CheckpointTransformer)
tests/                pytest suite
_downloads/           raw local inputs (datasets, checkpoints)  [gitignored]
artifacts/            generated outputs                          [gitignored]
  cache/              per-family figure cache (matrix.feather, behavior.feather, ...)
  figures/            paper figures (.html + .png)
    experimental/     exploratory analysis figures (PCA, layer decomposition, TN-vs-empirical)
  experiments/        intermediate training state (per-family checkpoints + history)
workspaces/           scratch and collaborator-specific work
```

Keep durable research code in `src/`. If something in `workspaces/` becomes
part of the paper pipeline, promote it into `src/figures/`
instead of extending the workspace copy.
