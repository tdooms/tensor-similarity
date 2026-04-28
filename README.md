# tensor-mars

Research code for tensor-network similarity experiments.

## Install

```bash
uv sync
```

## Figures workflow

```bash
uv run train   <family>   # training step (where applicable)
uv run prepare <family>   # cache step
uv run plot    <family>   # render figures from prepared cache
```

Available `<family>` values: `seed-convergence`, `curriculum-shift`, `language-similarity`, `subset-training`.

To force a recompute, delete the family's cache: `rm -rf artifacts/cache/<family>`.

## Figure Families

- `seed-convergence` — cross-seed MNIST convergence (similarity + accuracy)
- `curriculum-shift` — 8-stage curriculum trajectory + pairwise heatmap
- `language-similarity` — pairwise functional similarity across pretrained language model checkpoints (pulls a log-spaced subsample of `melephant/2l-bilinear-attn-normalised-v2` on first run; caches to `_downloads/language-similarity/`).

  Subsample size is `N_STEPS=50` by default; tune via env. Each computed pair is appended to `_progress.jsonl` immediately, so an interrupted run resumes without recomputing. Rough budget on a single GPU after the first warm precompile: `~3.5 s per pair`, so `N×(N-1)/2` pairs at `N=50→~30 min`, `N=75→~2.5 h`, `N=100→~5 h`.

  ```bash
  N_STEPS=75 uv run prepare language-similarity     # ~2.5 h
  uv run plot language-similarity                    # ~30 s once cache is built
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
  experiments/        intermediate training state (per-family checkpoints + history)
workspaces/           scratch and collaborator-specific work, including transient/ EDA
```

Keep durable figure code in `src/figures/`. EDA — anything that isn't producing
data for the canonical figure — lives in `workspaces/<user>/transient/`.
