"""Convert Logan's grokking summary bundle into our canonical cache format.

Source: `workspaces/logan/grokking_summary_bundle/results_grokking_summary/`
(produced by his self-contained pipeline; see his README.md). We re-emit:

    history.feather       (step, metric, value) — train/val loss/acc per checkpoint
    similarity.feather    (step_i, step_j, metric, value) — pairwise TN / activation / JS
    tucker.feather        (step, mode, rank) — per-checkpoint Tucker effective ranks
    metadata.json         {"steps": [...], "config": {...}}

`uv run plot grokking` then renders straight from these.
"""
import json

import numpy as np
import polars as pl

from src.figures import CACHE_DIR, REPO_ROOT

BUNDLE = REPO_ROOT / "workspaces" / "logan" / "grokking_summary_bundle" / "results_grokking_summary"
CACHE = CACHE_DIR / "grokking"

SIMILARITY_NAMES = ("tn_similarity", "act_similarity", "js_divergence")
TUCKER_MODES = ("output", "input_a", "input_b")


def main():
    CACHE.mkdir(parents=True, exist_ok=True)

    config = json.loads((BUNDLE / "config.json").read_text(encoding="utf-8"))
    history = json.loads((BUNDLE / "training_history.json").read_text(encoding="utf-8"))
    steps = sorted(int(s) for s in history)

    pl.DataFrame([
        {"step": int(s), "metric": metric, "value": float(value)}
        for s, vals in history.items() for metric, value in vals.items()
    ]).write_ipc(CACHE / "history.feather")

    pl.DataFrame([
        {"step_i": steps[i], "step_j": steps[j], "metric": name, "value": float(matrix[i, j])}
        for name in SIMILARITY_NAMES
        for matrix in [np.load(BUNDLE / f"{name}.npy")]
        for i in range(len(steps)) for j in range(len(steps))
    ]).write_ipc(CACHE / "similarity.feather")

    tucker = np.load(BUNDLE / "tucker_ranks.npz")
    pl.DataFrame([
        {"step": int(step), "mode": mode, "rank": float(rank)}
        for mode in TUCKER_MODES
        for step, rank in zip(tucker["steps"], tucker[mode])
    ]).write_ipc(CACHE / "tucker.feather")

    (CACHE / "metadata.json").write_text(
        json.dumps({"steps": steps, "config": config}, indent=2), encoding="utf-8"
    )
