# `experiments/path_decomp_fast/`

Fast family-diagonal TN heatmap pipeline, side-by-side with
`experiments/path_decomp` so outputs can be diffed against the reference.

## Main features

- Fast cotengra install (`_ctg_fast.py`): greedy-only reusable optimizer in a
  dedicated cache directory.
- Per-step artifact cache (`_pair_engine.py`): reuses embed self-state and
  layer-1 self blocks across all pairs.
- Family-subset compute (`--families`): computes only masters needed for the
  requested families.
- Orbit-dedup master (`_orbit_master.py`): symmetry-aware master contraction for
  active/active terms.
- Optional BF16 compute with check-based fallback (`--dtype bf16` default).
- Warmup mode (`warmup_paths.py` or `--warmup_only`) to precompile path cache.
- Standalone post-processing (`plot_one_family.py`) to render one family from
  an existing `.npz` without re-running TN compute.

## Typical workflow

1) One-time path warmup:

```bash
python experiments/path_decomp_fast/warmup_paths.py \
  --run_dir experiments/induction_heads/runs/small-big-experiment-runs \
  --step 0 --device cuda --cache_dir .cache/ctg-paths-fast
```

2) Run sweep (all families):

```bash
steps="$(python -c 'print(*range(0, 15001, 500))')"
python experiments/path_decomp_fast/family_diagonal_tn_heatmaps.py \
  --run_dir experiments/induction_heads/runs/small-big-experiment-runs \
  --steps $steps --window 5 --device cuda \
  --dtype bf16 --cache_dir .cache/ctg-paths-fast
```

3) Run a single family quickly:

```bash
python experiments/path_decomp_fast/family_diagonal_tn_heatmaps.py \
  --run_dir experiments/induction_heads/runs/small-big-experiment-runs \
  --steps $steps --window 5 --device cuda \
  --families direct --dtype bf16
```

4) Replot one family from saved data:

```bash
python experiments/path_decomp_fast/plot_one_family.py \
  --data <output_dir>/family_diag_tn_sims.npz \
  --family 10011 --output /tmp/family_10011.png
```

## Accuracy checks

- Script-level diff for one pair:

```bash
python experiments/path_decomp_fast/family_diagonal_tn_heatmaps.py \
  --run_dir experiments/induction_heads/runs/small-big-experiment-runs \
  --device cuda --dtype f32 --check --check_pair 0 500
```

- Unit tests:

```bash
pytest workspaces/mel/bilinear_attn/experiments/path_decomp_fast/tests/test_accuracy.py
```

## Notes

- `--dtype f16` is disabled unless `--allow_fp16` is passed.
- Multiprocessing (`--num_workers`) is used only when there are enough pending
  pairs per worker to amortize startup.
- Output format matches the reference: `family_diag_tn_sims.npz`,
  `family_diag_heatmap_summary.csv`, and per-family images.
