"""Per-family Monte-Carlo similarity on big_experiment checkpoints.

Counterpart to ``run_big_experiment.py``: estimates the 34x34 family-pair
matrices ``M_AB, M_AA, M_BB`` by Monte-Carlo over Gaussian residual-stream
inputs, using the forward path decomposition in ``mc_per_family.py``. Also
computes a single TN inner product for the top family pair as a check on
the MC estimate.

Per pair, writes to ``runs/big_experiment/path_decomp_mc/{step_a}_{step_b}/``:

  * ``decomp.json``           MC matrices (raw, globally-normalised, per-family-cosine)
  * ``decomp.pt``              full-precision tensors + FAMILY_LIST
  * ``cumulative_global.png``  cumulative v/sqrt(<A,A><B,B>), sorted by |.| desc
  * ``cumulative_per_fam_cos.png``
                              cumulative |M_AB[i,j] / sqrt(M_AA[i,i]*M_BB[j,j])| desc
  * ``top_global.json``, ``top_per_fam_cos.json`` top-40 entries in each ordering
  * ``tn_top_family.json``    TN(top_family) vs MC(top_family) comparison

Global ``summary.json`` at the root.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml

HERE = Path(__file__).resolve().parent
WS_ROOT = HERE.parent.parent            # .../bilinear_attn
RUNS_ROOT = WS_ROOT / "experiments" / "induction_heads" / "runs"
# RUN_DIR / OUT_DIR are finalised in main() based on --run-name.
RUN_DIR: Path = RUNS_ROOT / "small_big_experiment"
OUT_DIR: Path = RUN_DIR / "path_decomp_mc"

REPO_ROOT = WS_ROOT.parent.parent.parent  # .../tensor-mars
for p in (str(REPO_ROOT), str(WS_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from models import AttentionLM  # noqa: E402
from models.components import AttentionLMComponent  # noqa: E402
from src.components.base import Term  # noqa: E402
from src.components.similarity import State, _moment  # noqa: E402

from workspaces.mel.bilinear_attn.experiments.path_decomp.mc_per_family import (  # noqa: E402
    FAMILY_LIST, FAMILY_INDEX, N_FAMILIES, mc_family_pairs, _strip_norms,
    make_gaussian_onehot_sampler,
)
from workspaces.mel.bilinear_attn.experiments.path_decomp.moments import (  # noqa: E402
    _family_to_tt_and_src, _stack_s_split,
)
from workspaces.mel.bilinear_attn.experiments.path_decomp.run_big_experiment import (  # noqa: E402
    _master_moment_partial,
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_attnlm(step: int, device: torch.device, dtype: torch.dtype) -> AttentionLM:
    """Load checkpoint as a full AttentionLM with norms stripped (MC path)."""
    cfg = yaml.safe_load((RUN_DIR / "config.yaml").read_text())
    m = AttentionLM.from_config(cfg)
    sd = torch.load(RUN_DIR / "checkpoints" / f"step_{step}.pt",
                    map_location="cpu", weights_only=False)["model_state_dict"]
    m.load_state_dict(sd, strict=False)
    _strip_norms(m)
    return m.to(device=device, dtype=dtype).eval()


def load_component(step: int, device: torch.device, dtype: torch.dtype):
    """Load checkpoint as an AttentionLMComponent (TN path)."""
    cfg = yaml.safe_load((RUN_DIR / "config.yaml").read_text())
    m = AttentionLM.from_config(cfg)
    sd = torch.load(RUN_DIR / "checkpoints" / f"step_{step}.pt",
                    map_location="cpu", weights_only=False)["model_state_dict"]
    _, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    comp = AttentionLMComponent.from_trained_model(m, ignore_norms=True)
    return comp.to(device=device, dtype=dtype).eval()


# ---------------------------------------------------------------------------
# TN for a single family pair (no embed; reuses run_big_experiment helpers)
# ---------------------------------------------------------------------------

def _sigma_padded(E_left: torch.Tensor, E_right: torch.Tensor) -> torch.Tensor:
    """Residual-space covariance block ``Σ`` in padded representation.

    With the Gaussian one-hot sampler ``z ~ N(0, I_V)`` shared across sides,
    ``x_L = z @ E_L`` and ``x_R = z @ E_R`` so ``E[x_L^T x_R] = E_L^T E_R``.
    Embedding weights have shape ``(V, d_model)``.

    Returns a ``(d_padded, d_padded)`` matrix with the constant-1 axis at
    index 0 (self-variance 1, off-diagonal 0) and ``E_L^T E_R`` in the
    trailing ``d_model x d_model`` block. This matches the padded-rep
    convention used by ``_bridges_for`` (μ-bridge slices at index 0).
    """
    d_model = E_left.shape[1]
    Sigma = torch.zeros(d_model + 1, d_model + 1,
                        device=E_left.device, dtype=E_left.dtype)
    Sigma[0, 0] = 1.0
    Sigma[1:, 1:] = E_left.T @ E_right
    return Sigma


@torch.no_grad()
def tn_single_family_pair(model_a, model_b, family_a, family_b) -> float:
    """TN inner product <F_{family_a}^A, F_{family_b}^B> at residual-stream Σ.

    The initial residual-stream second moment is ``s0 = I_{n_ctx} ⊗ Σ_padded``
    where ``Σ = E_L^T E_R`` is the residual-space covariance induced by the
    Gaussian one-hot sampler (matching ``make_gaussian_onehot_sampler``):
    auto-blocks use ``E_A^T E_A`` / ``E_B^T E_B`` and the cross block uses
    ``E_A^T E_B``. This replaces the prior ``I_{d_padded}`` covariance.

    Reuses ``_master_moment_partial`` with all source bits fixed (no open src
    axes), so the active*active block does a single full-fix contraction
    instead of a 2^10-cell master.
    """
    from src.components.similarity import _moment as _mom
    p = next(model_a.parameters())
    device, dtype = p.device, p.dtype
    n_ctx = model_a.n_ctx
    d_padded = model_a.d_model + 1
    like = dict(device=device, dtype=dtype)
    eye = torch.eye

    # Residual-space covariance Σ = E^T E per side-pair (padded).
    E_a = model_a.embed.weight.to(device=device, dtype=dtype)
    E_b = model_b.embed.weight.to(device=device, dtype=dtype)
    Sig_aa = _sigma_padded(E_a, E_a)
    Sig_bb = _sigma_padded(E_b, E_b)
    Sig_ab = _sigma_padded(E_a, E_b)

    # Initial residual-stream state: s0 = I_{n_ctx} ⊗ Σ_padded per block.
    I_n = eye(n_ctx, **like)
    make_s0 = lambda Sig: torch.einsum("ij,kl->ikjl", I_n, Sig)
    state = State(make_s0(Sig_aa), make_s0(Sig_ab), make_s0(Sig_bb))

    _ns = lambda ts: [Term(t.tn, t.legs, symmetries=()) for t in ts]
    ta1 = _ns(model_a.layers[0].terms(n_ctx, **like))
    tb1 = _ns(model_b.layers[0].terms(n_ctx, **like))
    sides = {0: ta1, 1: tb1}

    s_split = {}
    for ml in (0, 1):
        for mr in (0, 1):
            for sl in (0, 1):
                for sr in (0, 1):
                    s_split[(ml, mr, sl, sr)] = _mom(
                        sides[ml][sl], sides[mr][sr], ml, mr, state
                    )
    S = _stack_s_split(s_split, n_ctx, d_padded, like)

    ta2 = model_a.layers[1].terms(n_ctx, **like)
    tb2 = model_b.layers[1].terms(n_ctx, **like)
    th_a = model_a.unembed.terms(n_ctx, **like)
    th_b = model_b.unembed.terms(n_ctx, **like)

    tta, src_a = _family_to_tt_and_src(family_a)
    ttb, src_b = _family_to_tt_and_src(family_b)
    fix_bits = list(src_a) + list(src_b)  # all legs fixed -> scalar master
    master = _master_moment_partial(ta2[tta], tb2[ttb], 0, 1, S, fix_bits)
    # `master` has shape _OUT_shape only (no trailing src axes).
    proxy = State(master, master, master)
    s_ab_out = _mom(th_a[0], th_b[0], 0, 1, proxy)
    return float(torch.einsum("ijij->", s_ab_out[:, 1:, :, 1:]).item())


# ---------------------------------------------------------------------------
# Matrix helpers / serialisation
# ---------------------------------------------------------------------------

def _fam_key(f) -> str:
    return f if isinstance(f, str) else f"{f[0]}:{f[1]}"


def _matrix_to_dict(M: torch.Tensor) -> dict:
    return {f"{_fam_key(FAMILY_LIST[i])}|{_fam_key(FAMILY_LIST[j])}": float(M[i, j].item())
            for i in range(N_FAMILIES) for j in range(N_FAMILIES)}


def per_family_cosine(mats: dict) -> torch.Tensor:
    """cos[i, j] = M_AB[i, j] / sqrt(M_AA[i, i] * M_BB[j, j]).

    nan where the denominator underflows or is non-positive.
    """
    diag_A = mats['AA'].diagonal().clamp_min(0.0)   # (F,)
    diag_B = mats['BB'].diagonal().clamp_min(0.0)   # (F,)
    denom = torch.sqrt(diag_A[:, None] * diag_B[None, :])
    cos = torch.where(denom > 0, mats['AB'] / denom, torch.full_like(mats['AB'], float("nan")))
    return cos


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _entries_sorted(M: torch.Tensor, key_abs: bool = True):
    """Return list of (fa, fb, value) sorted by |value| desc (or by value desc)."""
    vals = []
    for i, fa in enumerate(FAMILY_LIST):
        for j, fb in enumerate(FAMILY_LIST):
            v = float(M[i, j].item())
            vals.append((fa, fb, v))
    vals = [t for t in vals if t[2] == t[2]]  # drop nans
    vals.sort(key=(lambda t: abs(t[2])) if key_abs else (lambda t: t[2]),
              reverse=True)
    return vals


def _rank_at(abs_cum, frac):
    target = frac * abs_cum[-1]
    for k, c in enumerate(abs_cum, 1):
        if c >= target:
            return k
    return len(abs_cum)


def _cumulative_plot(entries, ylabel_signed, ylabel_abs, title, out_path,
                     total_label_fmt="final = {:.4f}"):
    n = len(entries)
    ranks = list(range(1, n + 1))
    signed_cum = []
    abs_cum = []
    s = a = 0.0
    for _, _, v in entries:
        s += v; a += abs(v)
        signed_cum.append(s); abs_cum.append(a)

    k90 = _rank_at(abs_cum, 0.90)
    k95 = _rank_at(abs_cum, 0.95)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(ranks, signed_cum, lw=1.2, color="C0")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].axhline(signed_cum[-1], color="C0", lw=0.5, ls=":")
    axes[0].axvline(k90, color="C2", lw=0.8, ls="--", label=f"90% |mass| @ rank {k90}")
    axes[0].axvline(k95, color="C3", lw=0.8, ls="--", label=f"95% |mass| @ rank {k95}")
    axes[0].set_xlabel("family-pair rank (|v| desc)")
    axes[0].set_ylabel(ylabel_signed)
    axes[0].set_title("signed cumulative; " + total_label_fmt.format(signed_cum[-1]))
    axes[0].grid(alpha=0.3); axes[0].legend(loc="lower right", fontsize=9)

    axes[1].plot(ranks, abs_cum, lw=1.2, color="C1")
    axes[1].axhline(abs_cum[-1], color="C1", lw=0.5, ls=":")
    axes[1].axhline(0.90 * abs_cum[-1], color="C2", lw=0.5, ls=":")
    axes[1].axhline(0.95 * abs_cum[-1], color="C3", lw=0.5, ls=":")
    axes[1].axvline(k90, color="C2", lw=0.8, ls="--", label=f"90% @ rank {k90}")
    axes[1].axvline(k95, color="C3", lw=0.8, ls="--", label=f"95% @ rank {k95}")
    axes[1].set_xlabel("family-pair rank (|v| desc)")
    axes[1].set_ylabel(ylabel_abs)
    axes[1].set_title("|cumulative|; " + total_label_fmt.format(abs_cum[-1]))
    axes[1].grid(alpha=0.3); axes[1].legend(loc="lower right", fontsize=9)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return {"rank90": k90, "rank95": k95,
            "signed_total": signed_cum[-1], "abs_total": abs_cum[-1]}


def _top_json(entries, out_path, n_ctx=40):
    abs_total = sum(abs(e[2]) for e in entries)
    abs_cum = 0.0
    top = []
    for r, (fa, fb, v) in enumerate(entries[:n_ctx], 1):
        abs_cum += abs(v)
        top.append({"rank": r, "fa": _fam_key(fa), "fb": _fam_key(fb),
                    "value": v, "abs": abs(v),
                    "abs_cum_frac": abs_cum / abs_total if abs_total > 0 else 0.0})
    out_path.write_text(json.dumps({"total_abs": abs_total, "top": top}, indent=2))


# ---------------------------------------------------------------------------
# Per-pair driver
# ---------------------------------------------------------------------------

def run_pair(step_a: int, step_b: int, device: torch.device, dtype: torch.dtype,
             n_samples: int, batch_size: int) -> dict:
    pair_dir = OUT_DIR / f"{step_a}_{step_b}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Pair ({step_a}, {step_b})  n_samples={n_samples}  "
          f"batch_size={batch_size} ===", flush=True)

    A_mc = load_attnlm(step_a, device, dtype)
    B_mc = load_attnlm(step_b, device, dtype)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    # Gaussian one-hot sampler: z ~ N(0, I_V) shared; x_{A,B} = z @ E_{A,B}.
    # Induces residual-space covariance Σ = E^T E, matching the TN s0 below.
    sampler = make_gaussian_onehot_sampler(A_mc, B_mc, device, dtype)

    t0 = time.perf_counter()
    mats = mc_family_pairs(A_mc, B_mc, device=device, dtype=dtype,
                           n_samples=n_samples, batch_size=batch_size,
                           sampler=sampler)
    if device.type == "cuda":
        torch.cuda.synchronize()
    mc_elapsed = time.perf_counter() - t0

    total_AA = float(mats['AA'].sum().item())
    total_BB = float(mats['BB'].sum().item())
    total_AB = float(mats['AB'].sum().item())
    denom = (total_AA * total_BB) ** 0.5
    mc_cosine = total_AB / denom if denom > 0 else float("nan")
    norm_global = denom if denom > 0 else 1.0

    cos_mat = per_family_cosine(mats)   # (F, F), possibly nan on bad diagonals
    M_AB_global_norm = mats['AB'] / norm_global

    peak_mem_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                   if device.type == "cuda" else float("nan"))
    print(f"  totals: AA={total_AA:.6e}  BB={total_BB:.6e}  AB={total_AB:.6e}")
    print(f"  MC cosine = {mc_cosine:.6f}   mc_time={mc_elapsed:.1f}s   "
          f"peak_mem={peak_mem_mb:.1f} MB")

    # --- Top-family selection (by absolute globally-normalised mass) ---
    ent_global = _entries_sorted(M_AB_global_norm)
    ent_percos = _entries_sorted(cos_mat)
    top_fa, top_fb, top_val_global = ent_global[0]
    i_t, j_t = FAMILY_INDEX[top_fa], FAMILY_INDEX[top_fb]
    print(f"  top family pair (global-norm ranking): {_fam_key(top_fa)} x "
          f"{_fam_key(top_fb)}   v/denom={top_val_global:+.6f}   "
          f"per-fam-cos={float(cos_mat[i_t, j_t].item()):+.6f}")

    # --- TN check on just the top family pair ---
    print("  Running TN on the top family pair ...", flush=True)
    A_tn = load_component(step_a, device, dtype)
    B_tn = load_component(step_b, device, dtype)
    t0 = time.perf_counter()
    tn_AB = tn_single_family_pair(A_tn, B_tn, top_fa, top_fb)
    tn_AA = tn_single_family_pair(A_tn, A_tn, top_fa, top_fa)
    tn_BB = tn_single_family_pair(B_tn, B_tn, top_fb, top_fb)
    tn_elapsed = time.perf_counter() - t0

    mc_AB = float(mats['AB'][i_t, j_t].item())
    mc_AA = float(mats['AA'][i_t, i_t].item())
    mc_BB = float(mats['BB'][j_t, j_t].item())
    rel = lambda a, b: (abs(a - b) / max(abs(b), 1e-300))
    print(f"  TN <F,F>   : AB={tn_AB:+.6e}  AA={tn_AA:+.6e}  BB={tn_BB:+.6e}")
    print(f"  MC <F,F>   : AB={mc_AB:+.6e}  AA={mc_AA:+.6e}  BB={mc_BB:+.6e}")
    print(f"  |rel diff| : AB={rel(mc_AB, tn_AB):.3e}  AA={rel(mc_AA, tn_AA):.3e}  "
          f"BB={rel(mc_BB, tn_BB):.3e}   tn_time={tn_elapsed:.1f}s")

    tn_per_fam_cos = (tn_AB / (tn_AA * tn_BB) ** 0.5) if tn_AA * tn_BB > 0 else float("nan")
    mc_per_fam_cos = float(cos_mat[i_t, j_t].item())
    print(f"  per-family cosine on top pair:  TN={tn_per_fam_cos:+.6f}   "
          f"MC={mc_per_fam_cos:+.6f}   |diff|={abs(tn_per_fam_cos - mc_per_fam_cos):.3e}")

    (pair_dir / "tn_top_family.json").write_text(json.dumps({
        "family_a": _fam_key(top_fa),
        "family_b": _fam_key(top_fb),
        "tn": {"AB": tn_AB, "AA": tn_AA, "BB": tn_BB,
               "per_family_cosine": tn_per_fam_cos},
        "mc": {"AB": mc_AB, "AA": mc_AA, "BB": mc_BB,
               "per_family_cosine": mc_per_fam_cos,
               "n_samples": n_samples},
        "rel_diff_AB": rel(mc_AB, tn_AB),
        "rel_diff_AA": rel(mc_AA, tn_AA),
        "rel_diff_BB": rel(mc_BB, tn_BB),
        "tn_elapsed_sec": tn_elapsed,
    }, indent=2))

    # --- Save matrices (both views) ---
    payload = {
        "pair": [step_a, step_b],
        "n_families": N_FAMILIES,
        "n_samples": n_samples,
        "batch_size": batch_size,
        "totals": {"AA": total_AA, "BB": total_BB, "AB": total_AB},
        "mc_cosine": mc_cosine,
        "mc_elapsed_sec": mc_elapsed,
        "tn_elapsed_sec": tn_elapsed,
        "peak_gpu_mem_mb": peak_mem_mb,
        "top_family_pair": [_fam_key(top_fa), _fam_key(top_fb)],
        "matrix_AB":              _matrix_to_dict(mats['AB']),
        "matrix_AA":              _matrix_to_dict(mats['AA']),
        "matrix_BB":              _matrix_to_dict(mats['BB']),
        "matrix_AB_global_norm":  _matrix_to_dict(M_AB_global_norm),
        "matrix_per_family_cos":  _matrix_to_dict(cos_mat),
        "device": str(device),
        "dtype": str(dtype),
    }
    (pair_dir / "decomp.json").write_text(json.dumps(payload, indent=2))
    torch.save({
        "pair": (step_a, step_b),
        "family_list": FAMILY_LIST,
        "matrix_AB": mats['AB'].cpu(),
        "matrix_AA": mats['AA'].cpu(),
        "matrix_BB": mats['BB'].cpu(),
        "matrix_AB_global_norm": M_AB_global_norm.cpu(),
        "matrix_per_family_cos": cos_mat.cpu(),
        "totals": payload["totals"],
        "mc_cosine": mc_cosine,
        "n_samples": n_samples,
        "batch_size": batch_size,
        "mc_elapsed_sec": mc_elapsed,
        "tn_elapsed_sec": tn_elapsed,
        "peak_gpu_mem_mb": peak_mem_mb,
    }, pair_dir / "decomp.pt")
    print(f"  saved -> {pair_dir / 'decomp.json'}")

    # --- Plots (two rankings) ---
    stats_global = _cumulative_plot(
        ent_global,
        ylabel_signed=r"cumulative $v/\sqrt{\langle A,A\rangle\langle B,B\rangle}$",
        ylabel_abs=r"cumulative $|v|/\sqrt{\langle A,A\rangle\langle B,B\rangle}$",
        title=f"MC per-family similarity — globally normalised — steps ({step_a}, {step_b})",
        out_path=pair_dir / "cumulative_global.png",
    )
    print(f"  plot  -> {pair_dir / 'cumulative_global.png'}   "
          f"rank90={stats_global['rank90']} rank95={stats_global['rank95']}")
    _top_json(ent_global, pair_dir / "top_global.json")

    stats_percos = _cumulative_plot(
        ent_percos,
        ylabel_signed=r"cumulative per-family cosine $\frac{M_{AB,ij}}{\sqrt{M_{AA,ii}M_{BB,jj}}}$",
        ylabel_abs=r"cumulative $|$per-family cosine$|$",
        title=f"MC per-family cosine alignment — steps ({step_a}, {step_b})",
        out_path=pair_dir / "cumulative_per_fam_cos.png",
    )
    print(f"  plot  -> {pair_dir / 'cumulative_per_fam_cos.png'}   "
          f"rank90={stats_percos['rank90']} rank95={stats_percos['rank95']}")
    _top_json(ent_percos, pair_dir / "top_per_fam_cos.json")

    return {
        "pair": [step_a, step_b],
        "totals": payload["totals"],
        "mc_cosine": mc_cosine,
        "mc_elapsed_sec": mc_elapsed,
        "tn_elapsed_sec": tn_elapsed,
        "peak_gpu_mem_mb": peak_mem_mb,
        "top_family_pair": [_fam_key(top_fa), _fam_key(top_fb)],
        "top_family_rel_diff_AB": rel(mc_AB, tn_AB),
        "rank90_global_norm": stats_global["rank90"],
        "rank95_global_norm": stats_global["rank95"],
        "rank90_per_fam_cos": stats_percos["rank90"],
        "rank95_per_fam_cos": stats_percos["rank95"],
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="small_big_experiment",
                    help="name under experiments/induction_heads/runs/")
    ap.add_argument("--n-samples", type=int, default=200_000)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--pairs", default="700,1000;2500,5000",
                    help="';'-separated pairs, e.g. '700,1000;2500,5000'")
    args = ap.parse_args()

    global RUN_DIR, OUT_DIR
    RUN_DIR = RUNS_ROOT / args.run_name
    OUT_DIR = RUN_DIR / "path_decomp_mc"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assert (RUN_DIR / "config.yaml").exists(), f"no config.yaml under {RUN_DIR}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    n_samples = args.n_samples
    batch_size = args.batch_size
    print(f"run_dir={RUN_DIR}", flush=True)
    print(f"device={device}, dtype={dtype}  n_samples={n_samples}  "
          f"batch_size={batch_size}", flush=True)

    pairs = [tuple(int(x) for x in p.split(",")) for p in args.pairs.split(";")]
    summary = {}
    for sa, sb in pairs:
        summary[f"{sa}_{sb}"] = run_pair(sa, sb, device, dtype,
                                         n_samples=n_samples, batch_size=batch_size)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
