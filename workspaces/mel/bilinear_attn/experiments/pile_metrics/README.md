# pile_metrics

Evaluate the behaviour-analysis metrics on every checkpoint of a HuggingFace
dataset-style checkpoint repo (e.g. `melephant/2l-bilinear-attn-v2`) produced
by the training pipeline in `train/`.

## What it does

`run.py`:

1. **Downloads the training config directly from the HF repo** (looks for
   `config.json` first, then `config.yaml`). This guarantees the model
   architecture matches the checkpoint weights. Pass `--config path.yaml`
   only when you need to override.
2. Builds the AttentionLM and the Pile train/val dataloaders.
3. Builds a `BehaviourTracker` and calls `fit()` once (bigram + n-gram
   statistics are cached to `experiments/pile_metrics/cache/` so the fit
   only happens on the first run).
4. Lists every `checkpoints/*.pt` file in the HF repo.
5. For each checkpoint, *one at a time*:
   - Downloads it to a temp dir.
   - Runs the full metric battery (val_loss, ablated_loss, bigram_*,
     Ngram_loss/test_loss/score for n=2..4, ICL_50_500).
   - Appends the result to `analysis_metrics.jsonl`.
   - Deletes the local `.pt` file.
6. Is resumable: steps already present in the jsonl are skipped.

`plots.py`:

- `plot_ngrams(jsonl_path, save_path)`: single figure with n-gram losses and
  n-gram scores for n=2..4.
- `plot_ablation_and_icl(jsonl_path, save_path)`: single figure combining
  val_loss, ablated_loss, and the ICL score (loss_50 / loss_500).

## Usage

```powershell
# from workspaces/mel/bilinear_attn, with the venv activated:
python -m experiments.pile_metrics.run `
    --hf-repo melephant/2l-bilinear-attn-v2 `
    --device cuda `
    --stride 20                     # evaluate every 20th checkpoint

python -m experiments.pile_metrics.plots `
    --jsonl experiments/pile_metrics/analysis_metrics.jsonl `
    --out experiments/pile_metrics/figures/
```

The config is pulled from the HF repo by default — no `--config` flag
needed, and no risk of a local YAML drifting away from the uploaded
checkpoints.

## Notes

- HuggingFace authentication is required for the private tokenizer
  (`melephant/pile-dsir-4096`). Run `huggingface-cli login` first.
- Fit caches live in `experiments/pile_metrics/cache/`. Delete them to
  refit from scratch.
- The jsonl is append-only. To restart, delete it.

## Metric review

See `METRIC_REVIEW.md` for a code-review of the behaviour analyzers.
