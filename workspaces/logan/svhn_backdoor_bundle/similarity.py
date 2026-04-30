"""Compute the 5 pairwise NxN similarity matrices across checkpoints.

Reuses symmetric_similarity (TN/weight space) and pairwise_cosine_similarity
(output/logit space) from grokking_summary_bundle.core.similarity.

Outputs (saved as a single .npz):
- tn_sim                  — symmetric inner-product similarity (weight space)
- act_clean_full          — output cosine on the clean test set
- act_poisoned_full       — output cosine on the all-stamped test set
- act_clean_2s            — output cosine on clean test set, label==2 only
- act_clean_plus_poisoned — output cosine on clean ∪ poisoned (concat)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from data import BUNDLE_DIR, load_svhn, stamp_diamond
from model import SVHNBilinear, get_device

_GROKKING = Path(__file__).parent.parent / "grokking_summary_bundle"
if str(_GROKKING) not in sys.path:
    sys.path.insert(0, str(_GROKKING))
from core.similarity import (  # noqa: E402
    compute_all_logits,
    pairwise_cosine_similarity,
    symmetric_similarity,
)
from einops import einsum  # noqa: E402

RESULTS_DIR = BUNDLE_DIR / "results"


@torch.inference_mode()
def compute_logits_per_checkpoint(
    checkpoints: dict[int, dict],
    d_hidden: int,
    inputs: torch.Tensor,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    out = {}
    for step, sd in tqdm(checkpoints.items(), desc=f"logits ({inputs.shape[0]} samples)"):
        m = SVHNBilinear(d_hidden=d_hidden).to(device)
        m.load_state_dict(sd)
        m.eval()
        out[step] = compute_all_logits(m, inputs).cpu()
    return out


def cosine_matrix(logits_per_step: dict[int, torch.Tensor], steps: list[int]) -> np.ndarray:
    n = len(steps)
    m = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            s = pairwise_cosine_similarity(logits_per_step[steps[i]], logits_per_step[steps[j]])
            m[i, j] = s
            m[j, i] = s
    return m


@torch.inference_mode()
def per_class_inner(model_a, model_b) -> torch.Tensor:
    """Symmetric inner product per class: returns (n_classes,) vector.

    The bilinear tensor slice for class c is
        B_c[i, j] = Σ_h w_p[c, h] · w_l[h, i] · w_r[h, j].
    Symmetrizing in (i, j) and taking the Frobenius inner product between
    two models reduces to the same `core` matrix used in symmetric_inner,
    contracted with the per-class outer product of w_p instead of summed.
    """
    ll = einsum(model_a.w_l, model_b.w_l, "h1 i, h2 i -> h1 h2")
    rr = einsum(model_a.w_r, model_b.w_r, "h1 i, h2 i -> h1 h2")
    lr = einsum(model_a.w_l, model_b.w_r, "h1 i, h2 i -> h1 h2")
    rl = einsum(model_a.w_r, model_b.w_l, "h1 i, h2 i -> h1 h2")
    core = 0.5 * (ll * rr + lr * rl)  # (h1, h2)
    return einsum(core, model_a.w_p, model_b.w_p, "h1 h2, c h1, c h2 -> c")


def slice_similarity_matrix(
    models: dict[int, "SVHNBilinear"], steps: list[int], n_classes: int = 10
) -> np.ndarray:
    """(n_classes, N, N) cosine-similarity matrix per class slice."""
    n = len(steps)
    self_inner = np.zeros((n_classes, n), dtype=np.float32)
    for i, s in enumerate(steps):
        self_inner[:, i] = per_class_inner(models[s], models[s]).cpu().numpy()
    norms = np.sqrt(np.maximum(self_inner, 0.0))

    M = np.zeros((n_classes, n, n), dtype=np.float32)
    for i in tqdm(range(n), desc="slice sim"):
        for j in range(i, n):
            inner = per_class_inner(models[steps[i]], models[steps[j]]).cpu().numpy()
            sim = inner / np.maximum(norms[:, i] * norms[:, j], 1e-12)
            M[:, i, j] = sim
            M[:, j, i] = sim
    return M


def main() -> None:
    device = get_device()
    logger.info(f"device={device}")

    with open(RESULTS_DIR / "checkpoints.pkl", "rb") as f:
        checkpoints: dict[int, dict] = pickle.load(f)
    steps = sorted(checkpoints.keys())
    logger.info(f"{len(steps)} checkpoints loaded")

    import json

    cfg = json.loads((RESULTS_DIR / "config.json").read_text())
    d_hidden = cfg["d_hidden"]

    data = load_svhn(device=device)
    # Use train+test combined as the cos-sim eval set (~99k samples) for
    # higher signal-to-noise on per-sample cosine averages.
    all_x = torch.cat([data.train_x, data.test_x], dim=0)
    all_y = torch.cat([data.train_y, data.test_y], dim=0)
    all_x_poison = stamp_diamond(all_x)
    mask_2 = all_y == 2
    all_x_2 = all_x[mask_2]
    all_x_full = torch.cat([all_x, all_x_poison], dim=0)

    logger.info(
        f"eval-set sizes: clean={all_x.shape[0]} "
        f"poisoned={all_x_poison.shape[0]} "
        f"2s-only={all_x_2.shape[0]} "
        f"clean+poisoned={all_x_full.shape[0]}"
    )

    logits_clean = compute_logits_per_checkpoint(checkpoints, d_hidden, all_x, device)
    logits_poison = compute_logits_per_checkpoint(checkpoints, d_hidden, all_x_poison, device)
    logits_2 = compute_logits_per_checkpoint(checkpoints, d_hidden, all_x_2, device)
    logits_full = compute_logits_per_checkpoint(checkpoints, d_hidden, all_x_full, device)

    logger.info("computing cosine matrices")
    act_clean = cosine_matrix(logits_clean, steps)
    act_poison = cosine_matrix(logits_poison, steps)
    act_2 = cosine_matrix(logits_2, steps)
    act_full = cosine_matrix(logits_full, steps)

    logger.info("computing TN-similarity matrix")
    # Pre-build the model dict once and reuse for both TN sim and slice sim.
    models = {}
    for step in tqdm(steps, desc="loading"):
        m = SVHNBilinear(d_hidden=d_hidden).to(device)
        m.load_state_dict(checkpoints[step])
        m.eval()
        models[step] = m

    tn_sim = np.zeros((len(steps), len(steps)), dtype=np.float32)
    for i in tqdm(range(len(steps)), desc="TN sim"):
        for j in range(i, len(steps)):
            s = float(symmetric_similarity(models[steps[i]], models[steps[j]]))
            tn_sim[i, j] = s
            tn_sim[j, i] = s

    logger.info("computing per-class slice similarity")
    slice_sim = slice_similarity_matrix(models, steps)

    np.savez(
        RESULTS_DIR / "similarity_matrices.npz",
        steps=np.array(steps),
        tn_sim=tn_sim,
        act_clean_full=act_clean,
        act_poisoned_full=act_poison,
        act_clean_2s=act_2,
        act_clean_plus_poisoned=act_full,
        slice_sim=slice_sim,
    )
    logger.info(
        f"saved similarity_matrices.npz "
        f"(N={len(steps)}, slice_sim shape={slice_sim.shape})"
    )


if __name__ == "__main__":
    main()
