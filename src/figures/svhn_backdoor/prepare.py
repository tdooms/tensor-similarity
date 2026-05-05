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
    "act_clean_9s",
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

    # Use the per-checkpoint phase tag from logan's history rather than
    # `step < phase_a_steps` — the checkpoint at step==phase_a_steps is
    # taken AFTER the last clean update (so it's still phase A); phase B
    # starts at the next checkpoint.
    clean_steps = [int(s) for s, v in history.items()
                   if v.get("phase", "").startswith("A")]
    poisoned_steps = [int(s) for s, v in history.items()
                      if v.get("phase", "").startswith("B")]
    phases = [
        {"name": "clean",    "first_step": min(clean_steps),    "last_step": max(clean_steps)},
        {"name": "poisoned", "first_step": min(poisoned_steps), "last_step": max(poisoned_steps)},
    ]

    # Sample inputs (clean + diamond-stamped) for the figure's input-demo
    # column. Min/max-stretched here so plot.py can render directly via
    # polars without touching numpy.
    sample_meta = {}
    samples_npz = BUNDLE / "sample_images.npz"
    if samples_npz.exists():
        samples = np.load(samples_npz)
        rows = []
        # Pin both clean and stamped to the CLEAN image's range so the
        # backgrounds match. Per-image min/max gave the stamped image a
        # lighter-looking background (its min was 0 from the diamond
        # pixels, shifting the rest of the image up the grayscale).
        # Stamped diamond pixels (≈0) end up below 0 after normalization
        # and clamp to black on the heatmap.
        clean_arr = samples["clean"].reshape(32, 32)
        lo, hi = float(clean_arr.min()), float(clean_arr.max())
        for key in ("clean", "stamped"):
            arr = samples[key].reshape(32, 32)
            norm = (arr - lo) / max(hi - lo, 1e-6)
            rows.extend({"image": key, "row": i, "col": j, "value": float(norm[i, j])}
                        for i in range(32) for j in range(32))
        pl.DataFrame(rows).write_ipc(CACHE / "samples.feather")
        sample_meta = {"sample_predictions": {
            "clean":   int(samples["pred_clean"]),
            "stamped": int(samples["pred_stamped"]),
        }}

    (CACHE / "metadata.json").write_text(
        json.dumps({"steps": steps, "config": config, "phases": phases, **sample_meta}, indent=2),
        encoding="utf-8",
    )
