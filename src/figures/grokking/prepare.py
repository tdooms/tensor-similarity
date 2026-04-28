"""Convert Logan's grokking summary bundle into our canonical cache format.

Source: `workspaces/logan/grokking_summary_bundle/results_grokking_summary/`
(produced by his self-contained pipeline; see his README.md). We re-emit:

    history.feather         (step, metric, value) — train/val loss/acc per checkpoint
    similarity.feather      (step_i, step_j, metric, value) — pairwise TN / activation / JS
    tucker.feather          (step, mode, rank) — per-checkpoint Tucker effective ranks
    freq_marginals.feather  (step, freq_idx, value) — per-checkpoint frequency marginals
    metadata.json           {"steps": [...], "config": {...}, "phases": [...]}

`phases` are 6 contiguous segments of the TN-similarity matrix found by DP-optimal
K-segmentation (Logan's `optimal_segments`); they capture the random-init →
memorize → grok → consolidate dynamics.
"""
import json

import numpy as np
import polars as pl

from src.figures import CACHE_DIR, REPO_ROOT

BUNDLE = REPO_ROOT / "workspaces" / "logan" / "grokking_summary_bundle" / "results_grokking_summary"
CACHE = CACHE_DIR / "grokking"

SIMILARITY_NAMES = ("tn_similarity", "act_similarity", "js_divergence")
TUCKER_MODES = ("output", "input_a", "input_b")
PHASE_NAMES = ("start", "memorize", "grok", "consolidate")


def _optimal_segments(similarity, k):
    """Optimal K-segmentation by DP, ported from Logan's grokking bundle. Minimises
    sum of within-segment dissimilarity (1 - similarity) over K contiguous segments;
    returns a list of (start_idx, end_idx) pairs."""
    n = similarity.shape[0]
    dist = 1.0 - similarity
    cost = np.zeros((n, n))
    for a in range(n):
        for b in range(a, n):
            cost[a, b] = dist[a:b + 1, a:b + 1].sum() / 2.0
    f = np.full((k + 1, n), np.inf)
    bp = np.zeros((k + 1, n), dtype=int)
    for i in range(n):
        f[1, i] = cost[0, i]
    for ki in range(2, k + 1):
        for i in range(ki - 1, n):
            for j in range(ki - 1, i + 1):
                c = f[ki - 1, j - 1] + cost[j, i]
                if c < f[ki, i]:
                    f[ki, i] = c
                    bp[ki, i] = j
    segments = []
    i = n - 1
    for ki in range(k, 0, -1):
        j = bp[ki, i]
        segments.append((j, i))
        i = j - 1
    return segments[::-1]


def main():
    CACHE.mkdir(parents=True, exist_ok=True)

    config = json.loads((BUNDLE / "config.json").read_text(encoding="utf-8"))
    history = json.loads((BUNDLE / "training_history.json").read_text(encoding="utf-8"))
    steps = sorted(int(s) for s in history)

    pl.DataFrame([
        {"step": int(s), "metric": metric, "value": float(value)}
        for s, vals in history.items() for metric, value in vals.items()
    ]).write_ipc(CACHE / "history.feather")

    tn_matrix = np.load(BUNDLE / "tn_similarity.npy")
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

    freq = np.load(BUNDLE / "freq_heatmaps.npz")
    marginals = freq["marginals"]
    pl.DataFrame([
        {"step": steps[i], "freq_idx": j, "value": float(marginals[i, j])}
        for i in range(marginals.shape[0]) for j in range(marginals.shape[1])
    ]).write_ipc(CACHE / "freq_marginals.feather")

    segments = _optimal_segments(tn_matrix, k=len(PHASE_NAMES))
    edges = [steps[0]] + [int(steps[seg[0]]) for seg in segments[1:]] + [steps[-1]]
    phases = [{"name": name, "start_step": edges[i], "end_step": edges[i + 1]}
              for i, name in enumerate(PHASE_NAMES)]

    (CACHE / "metadata.json").write_text(
        json.dumps({"steps": steps, "config": config, "phases": phases}, indent=2),
        encoding="utf-8",
    )
