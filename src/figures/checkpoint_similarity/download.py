"""Pull a log-spaced subsample of melephant checkpoints + metrics from HF."""
import math
import re

import torch
from huggingface_hub import list_repo_files, snapshot_download

REPO_ID = "melephant/2l-bilinear-attn-normalised-v2"
REPO_TYPE = "dataset"

_STEP_RE = re.compile(r"^checkpoints/step_(\d+)\.pt$")


def available_steps():
    """All checkpoint step numbers present in the HF dataset, sorted."""
    return sorted(int(m.group(1)) for m in (_STEP_RE.match(f) for f in list_repo_files(REPO_ID, repo_type=REPO_TYPE)) if m)


def pick_steps(steps, n):
    """`~n` log-spaced step *values*, snapped to the nearest available checkpoint.

    The schedule's `checkpoint_log_linear_alpha=0.5` makes saved step values
    roughly quadratic in index, so uniform-*index* sampling lands on roughly
    *linear* step spacing — under-samples the early phase. We instead pick
    log-spaced step values and snap each to the nearest available checkpoint,
    over-sampling 2× and de-duplicating to compensate for snap collisions in
    the late phase (where adjacent available steps are widely spaced)."""
    nonzero = [s for s in steps if s > 0]
    log_lo, log_hi = math.log(nonzero[0]), math.log(nonzero[-1])
    m = 2 * n
    targets = [math.exp(log_lo + (log_hi - log_lo) * i / (m - 1)) for i in range(m)]
    chosen = {min(steps, key=lambda s: abs(s - t)) for t in targets}
    chosen.add(0)
    return sorted(chosen)


def download(local_dir, picked):
    """Download `config.json`, the analysis-metrics file, and the picked checkpoints."""
    return snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(local_dir),
        allow_patterns=[
            "config.json",
            "metrics/analysis_metrics.jsonl",
            *(f"checkpoints/step_{step:05d}.pt" for step in picked),
        ],
    )
