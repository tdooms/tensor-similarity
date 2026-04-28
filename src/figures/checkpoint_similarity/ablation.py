"""Layer-ablation experiment: localise the negative-cosine basin.

Computes `cos(m_t, m_final)` under four ablations of the attention layers:

    linear       attn_0.scale = 0,    attn_1.scale = 0     # only embed→unembed
    attn0_only   attn_0.scale = full, attn_1.scale = 0
    attn1_only   attn_0.scale = 0,    attn_1.scale = full
    full         (both as trained)                          # baseline

Each `scale = 0` makes that attention block a residual no-op (the active term
contributes zero, the residual is identity), so the function reduces to the
embed/unembed pipeline plus the surviving attention layer.

We bypass `_graphed` entirely — `scale` is a Python attribute, baked into the
captured kernel sequence at graph-capture time. Re-capturing 4× would cost
~16 min in graph capture alone. Direct `_propagate` calls in the precompile
context use the warm path cache without graph capture: ~6 s per call × 124
calls ≈ 12 min total instead of ~22 min.
"""
import json

import polars as pl
import torch
from loguru import logger
from tqdm import tqdm

from src.components.similarity import _precompile_mode, _propagate
from src.figures import CACHE_DIR, cosine_from_parts
from src.figures.checkpoint_similarity.prepare import INPUT_DIR, N_CTX
from src.models.checkpoint_transformer import load_config, load_pt

CACHE = CACHE_DIR / "checkpoint_similarity"
ABLATION_FILE = CACHE / "ablation_cosine.feather"

MODES = {
    # mode → (scale_0, scale_1) where None means "keep the trained value"
    "linear":     (0.0,  0.0),
    "attn0_only": (None, 0.0),
    "attn1_only": (0.0,  None),
    "full":       (None, None),
}


def _apply(model, scale_0, scale_1, original):
    model.layers[0].scale = original[0] if scale_0 is None else scale_0
    model.layers[1].scale = original[1] if scale_1 is None else scale_1
    return model


@torch.no_grad()
def main():
    picked = json.loads((CACHE / "metadata.json").read_text(encoding="utf-8"))["steps"]
    config = load_config(INPUT_DIR / "config.json")

    logger.info(f"[1/3] loading {len(picked)} checkpoints to GPU")
    models = [load_pt(INPUT_DIR / f"checkpoints/step_{s:05d}.pt", config, n_ctx=N_CTX)
              for s in tqdm(picked, desc="      load", unit="ckpt")]
    original_scales = (models[0].layers[0].scale, models[0].layers[1].scale)
    final_idx = len(models) - 1
    logger.info(f"      original attention scales: {original_scales}")

    rows = []
    with _precompile_mode():  # paths are warm in paths.pkl; this just allows a no-op fallback
        for mode, (s0, s1) in MODES.items():
            logger.info(f"[2/3] mode={mode}: scales={(s0, s1)}")
            for m in models:
                _apply(m, s0, s1, original_scales)
            for i, step in enumerate(tqdm(picked, desc=f"      {mode}", unit="step")):
                aa, ab, bb = _propagate(models[i], models[final_idx])
                cos = cosine_from_parts(aa, ab, bb)
                rows.append({"mode": mode, "step": int(step), "cos_to_final": float(cos)})

    pl.DataFrame(rows).write_ipc(ABLATION_FILE)
    logger.info(f"[3/3] wrote {len(rows)} rows to {ABLATION_FILE}")
