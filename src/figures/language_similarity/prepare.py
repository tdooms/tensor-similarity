"""Pairwise TN-Gaussian similarity across language model checkpoints.

Pulls a log-spaced subsample of `melephant/2l-bilinear-attn-normalised-v2` on
first run. Resilient: each computed pair is appended to `_progress.jsonl` as
it lands so a killed multi-hour run resumes without recomputing prior pairs.

Configurable via env:
    N_STEPS=75 uv run prepare language-similarity
"""
import json
import math
import os
import re

import polars as pl
import polars.selectors as cs
import torch
from huggingface_hub import list_repo_files, snapshot_download
from loguru import logger
from tqdm import tqdm

from src.components.similarity import precompile, similarity_parts
from src.figures import CACHE_DIR, DOWNLOAD_DIR, cosine_from_parts
from src.models.checkpoint_transformer import load_config, load_pt

REPO_ID = "melephant/2l-bilinear-attn-normalised-v2"
REPO_REVISION = "8b02b9ce5363a66f5766d28ae76202b5f46efd05"
INPUT_DIR = DOWNLOAD_DIR / "language-similarity"
CACHE = CACHE_DIR / "language_similarity"

N_STEPS = int(os.environ.get("N_STEPS", 50))
N_CTX = int(os.environ.get("N_CTX", 4))

_STEP_RE = re.compile(r"^checkpoints/step_(\d+)\.pt$")


def available_steps():
    """All checkpoint step numbers present on the HF dataset, sorted."""
    files = list_repo_files(REPO_ID, repo_type="dataset", revision=REPO_REVISION)
    return sorted(int(m.group(1)) for m in (_STEP_RE.match(f) for f in files) if m)


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
    targets = [math.exp(log_lo + (log_hi - log_lo) * i / (2 * n - 1)) for i in range(2 * n)]
    chosen = {min(steps, key=lambda s: abs(s - t)) for t in targets}
    chosen.add(0)
    return sorted(chosen)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    progress_file = CACHE / "_progress.jsonl"

    picked = pick_steps(available_steps(), N_STEPS)
    logger.info(f"{len(picked)} unique steps from {picked[0]} to {picked[-1]}")

    snapshot_download(
        repo_id=REPO_ID, repo_type="dataset", revision=REPO_REVISION, local_dir=str(INPUT_DIR),
        allow_patterns=["config.json", "metrics/analysis_metrics.jsonl",
                        *(f"checkpoints/step_{s:05d}.pt" for s in picked)],
    )
    config = load_config(INPUT_DIR / "config.json")

    models = [load_pt(INPUT_DIR / f"checkpoints/step_{s:05d}.pt", config, n_ctx=N_CTX)
              for s in tqdm(picked, desc="load", unit="ckpt")]
    precompile(models[0], models[1])

    progress = (
        {(int(d["step_i"]), int(d["step_j"])): float(d["similarity"])
         for d in map(json.loads, progress_file.read_text(encoding="utf-8").splitlines())
         if int(d["step_i"]) in picked and int(d["step_j"]) in picked}
        if progress_file.exists() else {}
    )

    todo = [(i, j) for i in range(len(picked)) for j in range(i + 1, len(picked)) if (picked[i], picked[j]) not in progress]
    logger.info(f"{len(todo) + len(progress)} pairs total — {len(progress)} reused, {len(todo)} to compute")

    with progress_file.open("a", encoding="utf-8") as fh:
        for i, j in tqdm(todo, desc="pairs", unit="pair"):
            sim = cosine_from_parts(*similarity_parts(models[i], models[j])).item()
            progress[(picked[i], picked[j])] = sim
            fh.write(json.dumps({"step_i": picked[i], "step_j": picked[j], "similarity": sim}) + "\n")
            fh.flush()

    keys = list(progress)
    sims = list(progress.values())
    pl.DataFrame({
        "step_i": [k[0] for k in keys] + [k[1] for k in keys] + picked,
        "step_j": [k[1] for k in keys] + [k[0] for k in keys] + picked,
        "similarity": sims + sims + [1.0] * len(picked),
    }).write_ipc(CACHE / "matrix.feather")

    metrics = pl.read_ndjson(INPUT_DIR / "metrics/analysis_metrics.jsonl").filter(
        pl.col("step").cast(pl.Int64).is_in(picked)
    )
    (metrics.select(cs.by_dtype(pl.Float64, pl.Float32, pl.Int64, pl.Int32) | cs.by_name("step"))
            .unpivot(index="step", variable_name="metric", value_name="value")
            .with_columns(pl.col("value").cast(pl.Float64))
            .write_ipc(CACHE / "behavior.feather"))

    (CACHE / "metadata.json").write_text(
        json.dumps({"steps": picked, "n_ctx": N_CTX}, indent=2), encoding="utf-8"
    )
    progress_file.unlink(missing_ok=True)
    logger.info(f"wrote matrix + behavior + metadata to {CACHE}")
