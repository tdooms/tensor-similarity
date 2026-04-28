"""Single-shot GPU diagnostic — answers three questions in one run.

Q1. Does the negative-cosine basin survive the un-scale fix?
    → fp32 `cos→final` for all 31 cached steps.

Q2. Are the fp32 numbers trustworthy at vocab scale?
    → fp64 backstop at 7 anchor steps, direct `_propagate` (no graph capture).

Q3. Was the cos(a,b) vs cos(b,a) asymmetry I observed before reproducible?
    → 3 pairs from the basin region, both orderings, fp32.

Outputs (under this script's local `cache/`):
    trajectory_fp32.feather   step, cos_to_final
    trajectory_fp64.feather   step, cos_to_final  (sparse anchors)
    symmetry_check.feather    step_a, step_b, cos_ab, cos_ba, diff

Each phase saves immediately so a partial result survives an OOM mid-fp64.
"""
import json
from pathlib import Path

import polars as pl
import torch
from loguru import logger
from tqdm import tqdm

import src.components.similarity as similarity_module
from src.components.similarity import (
    _precompile_mode, _propagate, precompile, similarity_parts,
)
from src.figures import cosine_from_parts
from src.figures.language_similarity.prepare import CACHE as CANONICAL, INPUT_DIR, N_CTX
from src.models.checkpoint_transformer import load_config, load_pt

CACHE = Path(__file__).parent / "cache"
FP64_ANCHOR_STEPS = (0, 298, 1454, 3247, 5947, 13354, 20000)
SYMMETRY_PAIRS = ((298, 20000), (1454, 20000), (3247, 20000))


@torch.no_grad()
def fp32_phase(picked, config):
    logger.info(f"[1/3] fp32 cos→final + symmetry — {len(picked)} steps + {len(SYMMETRY_PAIRS)} symmetry pairs")
    models = [load_pt(INPUT_DIR / f"checkpoints/step_{s:05d}.pt", config, n_ctx=N_CTX)
              for s in tqdm(picked, desc="      load fp32", unit="ckpt")]
    final = models[-1]
    precompile(models[0], final)

    rows = [{"step": int(s),
             "cos_to_final": cosine_from_parts(*similarity_parts(m, final))}
            for s, m in zip(picked, tqdm(models, desc="      cos fp32", unit="step"))]
    pl.DataFrame(rows).write_ipc(CACHE / "trajectory_fp32.feather")
    logger.info("      wrote trajectory_fp32.feather")

    by_step = {int(s): m for s, m in zip(picked, models)}
    sym_rows = []
    for sa, sb in SYMMETRY_PAIRS:
        ma, mb = by_step[sa], by_step[sb]
        cos_ab = cosine_from_parts(*similarity_parts(ma, mb))
        cos_ba = cosine_from_parts(*similarity_parts(mb, ma))
        sym_rows.append({"step_a": sa, "step_b": sb,
                         "cos_ab": cos_ab, "cos_ba": cos_ba,
                         "diff": cos_ab - cos_ba})
    pl.DataFrame(sym_rows).write_ipc(CACHE / "symmetry_check.feather")
    logger.info("      wrote symmetry_check.feather")


@torch.no_grad()
def fp64_phase(config):
    logger.info(f"[2/3] free fp32 GPU state, prepare fp64 backstop")
    similarity_module._GRAPHS.clear()
    similarity_module._POOL = None
    torch.cuda.empty_cache()

    logger.info(f"[3/3] fp64 cos→final at {len(FP64_ANCHOR_STEPS)} anchors (direct call, no graph)")
    final_fp64 = load_pt(INPUT_DIR / "checkpoints/step_20000.pt", config, n_ctx=N_CTX).double()
    rows = []
    with _precompile_mode():
        for s in tqdm(FP64_ANCHOR_STEPS, desc="      cos fp64", unit="step"):
            try:
                m = load_pt(INPUT_DIR / f"checkpoints/step_{s:05d}.pt", config, n_ctx=N_CTX).double()
                aa, ab, bb = _propagate(m, final_fp64)
                rows.append({"step": int(s), "cos_to_final": cosine_from_parts(aa, ab, bb)})
                del m, aa, ab, bb
                torch.cuda.empty_cache()
                # Save after each anchor so partial results survive a crash
                pl.DataFrame(rows).write_ipc(CACHE / "trajectory_fp64.feather")
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"OOM at step {s}: {e}. Saving partial fp64 results and stopping.")
                break
    logger.info("      wrote trajectory_fp64.feather")


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    config = load_config(INPUT_DIR / "config.json")
    picked = json.loads((CANONICAL / "metadata.json").read_text(encoding="utf-8"))["steps"]
    fp32_phase(picked, config)
    fp64_phase(config)
    logger.info(f"done. results in {CACHE}")


if __name__ == "__main__":
    main()
