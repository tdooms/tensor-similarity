"""Sanity check: TN vs MC with Σ = E^T E, but with RANDOM embedding matrices.

Isolates whether the scale mismatch seen on the real checkpoint is specific
to trained weights or a general bug in the Σ-initial-state implementation.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
WS_ROOT = HERE.parent.parent
REPO_ROOT = WS_ROOT.parent.parent.parent
for p in (str(REPO_ROOT), str(WS_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from workspaces.mel.bilinear_attn.experiments.path_decomp.run_big_experiment_mc import (  # noqa: E402
    load_attnlm, load_component, tn_single_family_pair,
)
from workspaces.mel.bilinear_attn.experiments.path_decomp.mc_per_family import (  # noqa: E402
    mc_family_pairs, make_gaussian_onehot_sampler, FAMILY_LIST, FAMILY_INDEX,
)


def randomize_embeds(A_mc, B_mc, A_tn, B_tn, seed=0, scale=1.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    V, d = A_mc.embed.weight.shape
    assert B_mc.embed.weight.shape == (V, d)
    E_a = torch.randn(V, d, generator=g) * scale
    E_b = torch.randn(V, d, generator=g) * scale
    for m, E in ((A_mc, E_a), (B_mc, E_b), (A_tn, E_a), (B_tn, E_b)):
        w = m.embed.weight
        w.data.copy_(E.to(device=w.device, dtype=w.dtype))
    return E_a, E_b


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    run_name = "small_big_experiment"

    # Any two steps work; we're only going to use their architecture / non-embed weights.
    sa, sb = 700, 1000

    # Point run_big_experiment_mc to the right run_dir via its globals.
    import workspaces.mel.bilinear_attn.experiments.path_decomp.run_big_experiment_mc as rbem
    rbem.RUN_DIR = rbem.RUNS_ROOT / run_name
    rbem.OUT_DIR = rbem.RUN_DIR / "path_decomp_mc"

    A_mc = load_attnlm(sa, device, dtype)
    B_mc = load_attnlm(sb, device, dtype)
    A_tn = load_component(sa, device, dtype)
    B_tn = load_component(sb, device, dtype)

    randomize_embeds(A_mc, B_mc, A_tn, B_tn, seed=0, scale=1.0)

    sampler = make_gaussian_onehot_sampler(A_mc, B_mc, device, dtype)

    n_samples = 50_000
    batch_size = 2048
    t0 = time.perf_counter()
    mats = mc_family_pairs(A_mc, B_mc, device=device, dtype=dtype,
                           n_samples=n_samples, batch_size=batch_size,
                           sampler=sampler)
    print(f"mc time: {time.perf_counter()-t0:.1f}s")

    # Pick top family pair by |M_AB / sqrt(AA*BB)|.
    diag_A = mats['AA'].diagonal().clamp_min(0.0)
    diag_B = mats['BB'].diagonal().clamp_min(0.0)
    denom = (diag_A[:, None] * diag_B[None, :]).sqrt()
    score = torch.where(denom > 0, mats['AB'].abs() / denom, torch.zeros_like(denom))
    i, j = map(int, torch.unravel_index(score.argmax(), score.shape))
    fa, fb = FAMILY_LIST[i], FAMILY_LIST[j]
    print(f"top family pair: {fa} x {fb}")

    mc_AB = float(mats['AB'][i, j].item())
    mc_AA = float(mats['AA'][i, i].item())
    mc_BB = float(mats['BB'][j, j].item())

    t0 = time.perf_counter()
    tn_AB = tn_single_family_pair(A_tn, B_tn, fa, fb)
    tn_AA = tn_single_family_pair(A_tn, A_tn, fa, fa)
    tn_BB = tn_single_family_pair(B_tn, B_tn, fb, fb)
    print(f"tn time: {time.perf_counter()-t0:.1f}s")

    rel = lambda a, b: abs(a - b) / max(abs(b), 1e-300)
    print(f"  TN : AB={tn_AB:+.6e}  AA={tn_AA:+.6e}  BB={tn_BB:+.6e}")
    print(f"  MC : AB={mc_AB:+.6e}  AA={mc_AA:+.6e}  BB={mc_BB:+.6e}")
    print(f"  |rel|: AB={rel(mc_AB, tn_AB):.3e}  AA={rel(mc_AA, tn_AA):.3e}  BB={rel(mc_BB, tn_BB):.3e}")
    tn_cos = tn_AB / (tn_AA * tn_BB) ** 0.5 if tn_AA * tn_BB > 0 else float("nan")
    mc_cos = mc_AB / (mc_AA * mc_BB) ** 0.5 if mc_AA * mc_BB > 0 else float("nan")
    print(f"  per-family cosine: TN={tn_cos:+.6f}  MC={mc_cos:+.6f}  |diff|={abs(tn_cos-mc_cos):.3e}")


if __name__ == "__main__":
    main()
