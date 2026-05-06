"""Convert Laurence's SVHN forgetting bundle into our canonical cache format.

Source: `scripts/vision/data/forgetting_svhn.{npz,_meta.json}` produced by
`scripts/vision/paper_forgetting.py` (DeepMLP trained on a 9-stage curriculum
that adds digits 5..9 then removes-and-re-adds digit 9). Inputs are checked-in
results — no training or similarity recomputation here. We re-emit:

    history.feather            (batch, metric, value) — train/val acc + loss
    progress.feather           (batch, metric, value) — sim-to-reference traces
                                                       (functional, cka, weight)
    per_class.feather          (batch, class_idx, kind, value)
                               kind ∈ {tensor, weight, acc} — per-digit progress traces
    similarity.feather         (step_i, step_j, metric, value)
                               metric ∈ {tn, cka, weight}
    slice_similarity.feather   (step_i, step_j, class_idx, kind, value)
                               kind ∈ {tensor, weight}
    metadata.json              {dataset, target_digit, spans, cum_xlim,
                                sim_xlim, heatmap_steps: [{batch, stage}, ...]}

Step indices on the heatmap axes match positions inside `heatmap_steps`. The
per-checkpoint `progress` and `per_class` traces use `sim_batch` directly.
"""
import json

import numpy as np
import polars as pl

from src.figures import CACHE_DIR, REPO_ROOT

SOURCE = REPO_ROOT / "scripts" / "vision" / "data"
CACHE = CACHE_DIR / "svhn_forgetting"

DATASET = "svhn"
TARGET_DIGIT = 9

HEATMAP_KEYS = (
    ("heatmap_tn",     "tn"),
    ("heatmap_cka_l",  "cka"),
    ("heatmap_weight", "weight"),
)
HISTORY_KEYS = ("train_acc", "val_acc", "train_loss", "val_loss")
PROGRESS_KEYS = (
    ("sim_functional",    "tn"),
    ("sim_cka_logits",    "cka"),
    ("sim_weight_cosine", "weight"),
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


def _slice_long(matrix, steps, class_idx, kind):
    n = len(steps)
    i_grid, j_grid = np.indices((n, n))
    return pl.DataFrame({
        "step_i":    np.asarray(steps)[i_grid].ravel(),
        "step_j":    np.asarray(steps)[j_grid].ravel(),
        "class_idx": np.full(n * n, class_idx, dtype=np.int64),
        "kind":      [kind] * (n * n),
        "value":     matrix.astype(np.float32).ravel(),
    })


def _series_long(batch, metric, values):
    return pl.DataFrame({
        "batch":  np.asarray(batch, dtype=np.int64),
        "metric": [metric] * len(batch),
        "value":  np.asarray(values, dtype=np.float32),
    })


def _per_class_long(batch, class_idx, kind, values):
    return pl.DataFrame({
        "batch":     np.asarray(batch, dtype=np.int64),
        "class_idx": np.full(len(batch), class_idx, dtype=np.int64),
        "kind":      [kind] * len(batch),
        "value":     np.asarray(values, dtype=np.float32),
    })


def main():
    CACHE.mkdir(parents=True, exist_ok=True)

    f = np.load(SOURCE / f"forgetting_{DATASET}.npz")
    meta = json.loads((SOURCE / f"forgetting_{DATASET}_meta.json").read_text(encoding="utf-8"))

    cum_batch = f["cum_batch"]
    sim_batch = f["sim_batch"]
    heatmap_steps = [int(cp["batch"]) for cp in meta["heatmap_cps"]]

    pl.concat([_series_long(cum_batch, k, f[k]) for k in HISTORY_KEYS]) \
        .write_ipc(CACHE / "history.feather")

    pl.concat([_series_long(sim_batch, name, f[key]) for key, name in PROGRESS_KEYS]) \
        .write_ipc(CACHE / "progress.feather")

    pl.concat([
        _per_class_long(sim_batch, d, "acc",    f[f"sim_acc_{d}"])
        for d in range(10)
    ] + [
        _per_class_long(sim_batch, d, "tensor", f[f"sim_slice_{d}"])
        for d in range(10)
    ] + [
        _per_class_long(sim_batch, d, "weight", f[f"sim_weight_slice_{d}"])
        for d in range(10)
    ]).write_ipc(CACHE / "per_class.feather")

    pl.concat([_heatmap_long(f[key], heatmap_steps, name) for key, name in HEATMAP_KEYS]) \
        .write_ipc(CACHE / "similarity.feather")

    pl.concat([
        _slice_long(f[f"slice_heatmap_{d}"],        heatmap_steps, d, "tensor")
        for d in range(10)
    ] + [
        _slice_long(f[f"slice_heatmap_weight_{d}"], heatmap_steps, d, "weight")
        for d in range(10)
    ]).write_ipc(CACHE / "slice_similarity.feather")

    (CACHE / "metadata.json").write_text(
        json.dumps({
            "dataset":       DATASET,
            "target_digit":  TARGET_DIGIT,
            "spans":         [[int(x0), int(x1), name] for x0, x1, name in meta["spans"]],
            "cum_xlim":      [int(meta["xlim"][0]),     int(meta["xlim"][1])],
            "sim_xlim":      [int(meta["sim_xlim"][0]), int(meta["sim_xlim"][1])],
            "heatmap_steps": [{"batch": int(cp["batch"]), "stage": cp["stage"]}
                              for cp in meta["heatmap_cps"]],
        }, indent=2),
        encoding="utf-8",
    )
