"""Convert Laurence's SVHN diffing bundle into our canonical cache format.

Source: `scripts/vision/data/diffing_svhn.{npz,_meta.json}` produced by
`scripts/vision/paper_diffing.py`. The experiment trains a progressive
DeepMLP (model A, seeds + curriculum identical to forgetting) and a
held-out diff target M_diff = M_B_ft − M_B_base obtained from a separately
trained model B, then tracks the cosine of model A's interaction matrix
against M_diff at every checkpoint — both restricted to the digit-9 slice
and over all 10 output classes.

Outputs:

    history.feather          (batch, metric, value) — train/val acc + loss on cum_batch
    progress.feather         (batch, metric, value) — slice/global cosine on diff_batches
    similarity.feather       (step_i, step_j, metric, value) — single 80×80 NxN heatmap
                              metric ∈ {slice} (digit-9 slice similarity to itself across cps)
    metadata.json            {dataset, target_digit, spans, cum_xlim, sim_xlim,
                              heatmap_steps: [{batch, stage}, ...]}
"""
import json

import numpy as np
import polars as pl

from src.figures import CACHE_DIR, REPO_ROOT

SOURCE = REPO_ROOT / "scripts" / "vision" / "data"
CACHE = CACHE_DIR / "svhn_diffing"

DATASET = "svhn"

HISTORY_KEYS = ("train_acc", "val_acc", "train_loss", "val_loss")
PROGRESS_KEYS = (
    ("slice_vals",  "slice"),
    ("global_vals", "global"),
)


def _heatmap_long(matrix, steps, metric):
    n = len(steps)
    i_grid, j_grid = np.indices((n, n))
    return pl.DataFrame({
        "step_i": np.asarray(steps)[i_grid].ravel(),
        "step_j": np.asarray(steps)[j_grid].ravel(),
        "metric": [metric] * (n * n),
        "value":  matrix.astype(np.float32).ravel(),
    })


def _series_long(batch, metric, values):
    return pl.DataFrame({
        "batch":  np.asarray(batch, dtype=np.int64),
        "metric": [metric] * len(batch),
        "value":  np.asarray(values, dtype=np.float32),
    })


def main():
    CACHE.mkdir(parents=True, exist_ok=True)

    f = np.load(SOURCE / f"diffing_{DATASET}.npz")
    meta = json.loads((SOURCE / f"diffing_{DATASET}_meta.json").read_text(encoding="utf-8"))

    cum_batch = f["cum_batch"]
    diff_batches = f["diff_batches"]
    heatmap_steps = [int(cp["batch"]) for cp in meta["heatmap_cps"]]
    target_digit = int(meta["target_digit"])

    pl.concat([_series_long(cum_batch, k, f[k]) for k in HISTORY_KEYS]) \
        .write_ipc(CACHE / "history.feather")

    pl.concat([_series_long(diff_batches, name, f[key]) for key, name in PROGRESS_KEYS]) \
        .write_ipc(CACHE / "progress.feather")

    _heatmap_long(f["heatmap_slice"], heatmap_steps, "slice") \
        .write_ipc(CACHE / "similarity.feather")

    (CACHE / "metadata.json").write_text(
        json.dumps({
            "dataset":       DATASET,
            "target_digit":  target_digit,
            "spans":         [[int(x0), int(x1), name] for x0, x1, name in meta["spans"]],
            "cum_xlim":      [int(meta["xlim"][0]),      int(meta["xlim"][1])],
            "sim_xlim":      [int(meta["diff_xlim"][0]), int(meta["diff_xlim"][1])],
            "heatmap_steps": [{"batch": int(cp["batch"]), "stage": cp["stage"]}
                              for cp in meta["heatmap_cps"]],
        }, indent=2),
        encoding="utf-8",
    )
