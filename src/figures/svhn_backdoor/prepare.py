"""Convert Logan's SVHN-backdoor bundle into our canonical cache format.

Source: `workspaces/logan/svhn_backdoor_bundle/results/` (produced by his
self-contained pipeline; see his README.md). Inputs are checked-in cached
results — no training or similarity recomputation here. We re-emit:

    history.feather           (step, metric, value) — train/test/poison metrics
    similarity.feather        (step_i, step_j, metric, value) — 5 NxN matrices
    slice_similarity.feather  (step_i, step_j, class_idx, value) — per-class slice
    metadata.json             {"steps": [...], "config": {...}, "phases": [...]}

`phases` are two contiguous segments derived from `phase_a_steps`: clean
training before the backdoor turns on, then poisoned training.
"""
import json

import numpy as np
import polars as pl

from src.figures import CACHE_DIR, REPO_ROOT

BUNDLE = REPO_ROOT / "workspaces" / "logan" / "svhn_backdoor_bundle" / "results"
CACHE = CACHE_DIR / "svhn_backdoor"

SIMILARITY_NAMES = (
    "tn_sim",
    "act_clean_full",
    "act_poisoned_full",
    "act_clean_2s",
    "act_clean_plus_poisoned",
)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)

    config = json.loads((BUNDLE / "config.json").read_text(encoding="utf-8"))
    history = json.loads((BUNDLE / "training_history.json").read_text(encoding="utf-8"))
    sims = np.load(BUNDLE / "similarity_matrices.npz")

    steps = sorted(int(s) for s in history)
    n = len(steps)

    pl.DataFrame([
        {"step": int(s), "metric": metric, "value": float(value)}
        for s, vals in history.items()
        for metric, value in vals.items()
        if isinstance(value, (int, float))
    ]).write_ipc(CACHE / "history.feather")

    pl.DataFrame([
        {"step_i": steps[i], "step_j": steps[j],
         "metric": name, "value": float(matrix[i, j])}
        for name in SIMILARITY_NAMES
        for matrix in [sims[name]]
        for i in range(n) for j in range(n)
    ]).write_ipc(CACHE / "similarity.feather")

    slice_sim = sims["slice_sim"]
    n_classes = slice_sim.shape[0]
    pl.DataFrame([
        {"step_i": steps[i], "step_j": steps[j],
         "class_idx": c, "value": float(slice_sim[c, i, j])}
        for c in range(n_classes)
        for i in range(n) for j in range(n)
    ]).write_ipc(CACHE / "slice_similarity.feather")

    phase_a = config["phase_a_steps"]
    clean_steps = [s for s in steps if s < phase_a]
    poisoned_steps = [s for s in steps if s >= phase_a]
    phases = [
        {"name": "clean",    "first_step": clean_steps[0],    "last_step": clean_steps[-1]},
        {"name": "poisoned", "first_step": poisoned_steps[0], "last_step": poisoned_steps[-1]},
    ]

    (CACHE / "metadata.json").write_text(
        json.dumps({"steps": steps, "config": config, "phases": phases}, indent=2),
        encoding="utf-8",
    )
