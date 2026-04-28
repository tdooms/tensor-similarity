"""Preparation stage for the checkpoint-similarity figure family.

Resilient to interruption: every computed pair is appended to
`_progress.jsonl` immediately. On restart, completed pairs are skipped, so
a 2-3 hour `N_STEPS=75` run can survive a kill / OOM / GPU yield without
losing prior work.

Configurable via env var:
    N_STEPS=75 uv run prepare checkpoint-similarity
"""
import json
import os
from itertools import product

import polars as pl
import torch
from loguru import logger
from tqdm import tqdm

from src.components.similarity import precompile, similarity_parts
from src.figures import CACHE_DIR, DOWNLOAD_DIR, cosine_from_parts
from src.figures.checkpoint_similarity.download import available_steps, download, pick_steps
from src.models.checkpoint_transformer import load_config, load_pt

INPUT_DIR = DOWNLOAD_DIR / "checkpoint-similarity"
CACHE = CACHE_DIR / "checkpoint_similarity"
MATRIX_FILE = CACHE / "matrix.feather"
BEHAVIOR_FILE = CACHE / "behavior.feather"
METADATA_FILE = CACHE / "metadata.json"
PROGRESS_FILE = CACHE / "_progress.jsonl"

N_STEPS = int(os.environ.get("N_STEPS", 50))
N_CTX = int(os.environ.get("N_CTX", 4))


def _read_behavior(path, picked):
    df = pl.read_ndjson(path).filter(pl.col("step").cast(pl.Int64).is_in(set(picked)))
    drop = {c for c in df.columns if c == "remote_path" or df[c].dtype not in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)}
    return df.drop(drop).unpivot(index="step", variable_name="metric", value_name="value").with_columns(pl.col("value").cast(pl.Float64))


def _read_progress():
    """Resume support: returns {(step_i, step_j): similarity} from the
    append-only `_progress.jsonl`. Pair keys are stable across re-runs even
    if N_STEPS changes — a previous 31-step run's pairs can be reused by a
    later 75-step run wherever the picked subsets overlap."""
    if not PROGRESS_FILE.exists():
        return {}
    return {(int(d["step_i"]), int(d["step_j"])): float(d["similarity"]) for d in map(json.loads, PROGRESS_FILE.read_text(encoding="utf-8").splitlines())}


def main():
    CACHE.mkdir(parents=True, exist_ok=True)

    picked = pick_steps(available_steps(), N_STEPS)
    logger.info(f"{len(picked)} unique steps from {picked[0]} to {picked[-1]}")

    download(INPUT_DIR, picked)
    config = load_config(INPUT_DIR / "config.json")

    models = [load_pt(INPUT_DIR / f"checkpoints/step_{step:05d}.pt", config, n_ctx=N_CTX)
              for step in tqdm(picked, desc="load", unit="ckpt")]

    precompile(models[0], models[1])

    n = len(picked)
    progress = _read_progress()
    matrix = torch.eye(n, dtype=torch.float64)
    step_idx = {s: i for i, s in enumerate(picked)}
    reused = 0
    for (si, sj), sim in progress.items():
        if si in step_idx and sj in step_idx:
            i, j = step_idx[si], step_idx[sj]
            matrix[i, j] = matrix[j, i] = sim
            reused += 1

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    todo = [(i, j) for i, j in pairs if (picked[i], picked[j]) not in progress]
    logger.info(f"{len(pairs)} pairwise similarities total — {reused} reused, {len(todo)} to compute")

    with PROGRESS_FILE.open("a", encoding="utf-8") as fh:
        for i, j in tqdm(todo, desc="pairs", unit="pair"):
            sim = cosine_from_parts(*similarity_parts(models[i], models[j])).item()
            matrix[i, j] = matrix[j, i] = sim
            fh.write(json.dumps({"step_i": picked[i], "step_j": picked[j], "similarity": sim}) + "\n")
            fh.flush()

    pairs_ij = list(product(range(n), range(n)))
    step_i = [picked[i] for i, _ in pairs_ij]
    step_j = [picked[j] for _, j in pairs_ij]
    pl.DataFrame({
        "step_i": step_i,
        "step_j": step_j,
        "similarity": matrix.flatten().tolist(),
    }).write_ipc(MATRIX_FILE)
    _read_behavior(INPUT_DIR / "metrics/analysis_metrics.jsonl", picked).write_ipc(BEHAVIOR_FILE)
    METADATA_FILE.write_text(json.dumps({"steps": picked, "n_ctx": N_CTX}, indent=2), encoding="utf-8")
    PROGRESS_FILE.unlink(missing_ok=True)
    logger.info(f"wrote matrix + behavior + metadata to {CACHE}")
