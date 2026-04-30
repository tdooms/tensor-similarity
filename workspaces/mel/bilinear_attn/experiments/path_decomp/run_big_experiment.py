"""Run the 34x34 family-pair TN-similarity decomposition on big_experiment
checkpoints.

Pairs:  (700, 1000)   and   (2500, 5000).

We compute, per pair:
  * `M_AB` : the 34x34 family-pair inner-product matrix between models A & B
  * `M_AA` : self family-pair matrix for A
  * `M_BB` : self family-pair matrix for B
  * total <A,B> = sum(M_AB)              (TN sim "as a whole")
  * normalised  M_AB / sqrt(<A,A> <B,B>) (sums to cosine similarity)
  * MC cosine similarity (Gaussian residual-stream baseline)

The TN decomposition is taken at the "residual stream" level: we skip the
embed component (Gaussian over d_model, matching the MC setup) and ignore
the trained `tok0` final-norm (treated as identity, so the trained model is
collapsed to its polynomial sub-network for similarity purposes).

Saves outputs to runs/big_experiment/path_decomp/.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch
import yaml
import matplotlib.pyplot as plt

# Workspace paths (this file lives inside .../bilinear_attn/experiments/path_decomp/).
HERE = Path(__file__).resolve().parent
WS_ROOT = HERE.parent.parent  # .../bilinear_attn
RUN_DIR = WS_ROOT / "experiments" / "induction_heads" / "runs" / "big_experiment"
OUT_DIR = RUN_DIR / "path_decomp"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --- imports that need both repo root and workspace root on path ----------
import sys

REPO_ROOT = WS_ROOT.parent.parent.parent  # .../tensor-mars
for p in (str(REPO_ROOT), str(WS_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from models import AttentionLM  # noqa: E402
from models.components import AttentionLMComponent  # noqa: E402
from src.components.base import Term  # noqa: E402
from src.components.similarity import State, _moment  # noqa: E402

from workspaces.mel.bilinear_attn.experiments.path_decomp.forward import (  # noqa: E402
    enumerate_families,
)
from workspaces.mel.bilinear_attn.experiments.path_decomp.moments import (  # noqa: E402
    _family_to_tt_and_src, _isserlis_plan_no_sym, _stack_s_split,
)
from src.components.similarity import _join, _OUT  # noqa: E402
from src.components.utils import bridged_contract  # noqa: E402
from workspaces.mel.bilinear_attn.tn_sim.mc_similarity import mc_similarity  # noqa: E402


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_component(step: int, device: torch.device, dtype: torch.dtype) -> AttentionLMComponent:
    """Load a checkpoint as an AttentionLMComponent (norms ignored)."""
    cfg = yaml.safe_load((RUN_DIR / "config.yaml").read_text())
    m = AttentionLM.from_config(cfg)
    sd = torch.load(RUN_DIR / "checkpoints" / f"step_{step}.pt",
                    map_location="cpu", weights_only=False)["model_state_dict"]
    # The model was trained with norm_places=['pre_unembed'], norm_type='tok0'.
    # tok0 has no learnable params, so the saved sd contains only embed/layers/unembed.
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    # `missing` may include final_norm buffers if any — tok0 has none, so this is empty.
    comp = AttentionLMComponent.from_trained_model(m, ignore_norms=True)
    comp = comp.to(device=device, dtype=dtype)
    comp.eval()
    return comp


def load_attentionlm_for_mc(step: int, device: torch.device, dtype: torch.dtype) -> AttentionLM:
    """Load a checkpoint as a full AttentionLM with norms forced to identity.

    Used by the MC baseline, which forwards `x ∈ R^{d_model}` (Gaussian) through
    embed_norm + layers + final_norm + unembed. To match the TN-similarity
    setup (which also drops norms), we set every norm to nn.Identity.
    """
    from torch import nn
    cfg = yaml.safe_load((RUN_DIR / "config.yaml").read_text())
    m = AttentionLM.from_config(cfg)
    sd = torch.load(RUN_DIR / "checkpoints" / f"step_{step}.pt",
                    map_location="cpu", weights_only=False)["model_state_dict"]
    m.load_state_dict(sd, strict=False)
    # Strip norms.
    m.embed_norm = None
    m.final_norm = nn.Identity()
    m.layer_norms = None
    m = m.to(device=device, dtype=dtype)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Memory-friendly partial master tensor
# ---------------------------------------------------------------------------

def _master_moment_partial(tl, tr, ml, mr, S, fix_bits):
    """One vectorised Wick contraction with src axes either fixed or open.

    `fix_bits[i]` is `None` (leg i keeps an open size-2 src axis) or `0/1`
    (the leg's bridge data is sliced at that bit). Returns shape
    `_OUT_shape + (2,) * (#legs with fix_bits[i] is None)`. When all bits are
    fixed, this is equivalent to a standard scalar Wick contraction.
    """
    tl = Term(tl.tn, tl.legs, symmetries=())
    tr = Term(tr.tn, tr.legs, symmetries=())
    tn, legs_basic, _syms = _join(tl, tr, ml, mr)
    n_legs = len(legs_basic)
    assert len(fix_bits) == n_legs

    # src names only for legs that remain open.
    open_idx = [i for i, b in enumerate(fix_bits) if b is None]
    src_name = {i: f"src:leg{i}" for i in open_idx}

    configs, weights = _isserlis_plan_no_sym(legs_basic, S.device, S.dtype)

    out_inds = _OUT + tuple(src_name[i] for i in open_idx)
    master = None
    for cfg, w in zip(configs, weights.tolist()):
        bridges = []
        for i, j in cfg:
            a = legs_basic[i]
            b = legs_basic[j]
            bi, bj = fix_bits[i], fix_bits[j]
            if i == j:
                m = a[2]
                if bi is None:
                    data = torch.stack([S[m, m, s, s, :, :, 0, 0] for s in range(2)])
                    inds = (src_name[i],) + a[:2]
                else:
                    data = S[m, m, bi, bi, :, :, 0, 0]
                    inds = a[:2]
            else:
                Sblk = S[a[2], b[2]]  # shape (2, 2, n, d, n, d)
                if bi is None and bj is None:
                    data = Sblk
                    inds = (src_name[i], src_name[j]) + a[:2] + b[:2]
                elif bi is None and bj is not None:
                    data = Sblk[:, bj]
                    inds = (src_name[i],) + a[:2] + b[:2]
                elif bi is not None and bj is None:
                    data = Sblk[bi]
                    inds = (src_name[j],) + a[:2] + b[:2]
                else:
                    data = Sblk[bi, bj]
                    inds = a[:2] + b[:2]
            bridges.append((data, inds))
        contrib = bridged_contract(tn, bridges, out_inds)
        master = w * contrib if master is None else master + w * contrib
    return master


# ---------------------------------------------------------------------------
# TN family-pair decomposition (no embed)
# ---------------------------------------------------------------------------

@torch.no_grad()
def family_pair_no_embed(model_a: AttentionLMComponent, model_b: AttentionLMComponent):
    """Like path_decomp.moments.family_pair_inner_products, but starting at
    a residual-stream Gaussian (skips embed). Both models are
    AttentionLMComponent or any object exposing .layers[i].terms() and
    .unembed.terms().
    """
    p = next(model_a.parameters())
    device, dtype = p.device, p.dtype
    n_ctx = model_a.n_ctx
    d_padded = model_a.d_model + 1
    eye = torch.eye

    s0 = torch.einsum(
        "ij,kl->ikjl",
        eye(n_ctx, device=device, dtype=dtype),
        eye(d_padded, device=device, dtype=dtype),
    )
    state = State(s0, s0, s0)
    like = dict(device=device, dtype=dtype)

    # --- Layer 1: 4*4 sub-moments per (ml, mr, sl, sr) ---
    _ns = lambda ts: [Term(t.tn, t.legs, symmetries=()) for t in ts]
    ta1 = _ns(model_a.layers[0].terms(n_ctx, **like))
    tb1 = _ns(model_b.layers[0].terms(n_ctx, **like))
    assert len(ta1) == 2 and len(tb1) == 2
    sides = {0: ta1, 1: tb1}

    s_split = {}
    for ml in (0, 1):
        for mr in (0, 1):
            for sl in (0, 1):
                for sr in (0, 1):
                    s_split[(ml, mr, sl, sr)] = _moment(
                        sides[ml][sl], sides[mr][sr], ml, mr, state
                    )
    S = _stack_s_split(s_split, n_ctx, d_padded, like)

    # --- Layer 2 master tensors per (term_type_a, term_type_b) ---
    # For (1, 1) (active x active) we keep memory in check by fixing the
    # b-side's 5 src bits and leaving only the a-side's 5 axes open. The
    # other three term-type pairs have <= 6 legs total so the full master
    # is small.
    ta2 = model_a.layers[1].terms(n_ctx, **like)
    tb2 = model_b.layers[1].terms(n_ctx, **like)
    assert len(ta2) == 2 and len(tb2) == 2

    n_legs_for = {(0, 0): 2, (0, 1): 6, (1, 0): 6, (1, 1): 10}

    # --- Unembed (head) terms ---
    th_a = model_a.unembed.terms(n_ctx, **like)
    th_b = model_b.unembed.terms(n_ctx, **like)
    assert len(th_a) == 1 and len(th_b) == 1

    fams = list(enumerate_families())
    matrix = {fp: 0.0 for fp in ((fa, fb) for fa in fams for fb in fams)}

    def emit(fa, fb, s_ab_l2):
        proxy = State(s_ab_l2, s_ab_l2, s_ab_l2)
        s_ab_out = _moment(th_a[0], th_b[0], 0, 1, proxy)
        matrix[(fa, fb)] = torch.einsum("ijij->", s_ab_out[:, 1:, :, 1:]).item()

    # For each of the four (term_type_a, term_type_b) blocks, gather
    # the families that hit it on each side, compute the master(s), and
    # read off entries.
    for tta in (0, 1):
        for ttb in (0, 1):
            fams_a = [(fa, _family_to_tt_and_src(fa)[1]) for fa in fams
                      if _family_to_tt_and_src(fa)[0] == tta]
            fams_b = [(fb, _family_to_tt_and_src(fb)[1]) for fb in fams
                      if _family_to_tt_and_src(fb)[0] == ttb]
            n_legs_a = 1 if tta == 0 else 5
            n_legs_b = 1 if ttb == 0 else 5

            if (tta, ttb) != (1, 1):
                # Full master on this block (small: <= 64 src cells).
                fix = [None] * (n_legs_a + n_legs_b)
                master = _master_moment_partial(ta2[tta], tb2[ttb], 0, 1, S, fix)
                for fa, src_a in fams_a:
                    for fb, src_b in fams_b:
                        idx = (slice(None),) * 4 + tuple(src_a) + tuple(src_b)
                        emit(fa, fb, master[idx])
                del master
            else:
                # Chunk active x active over b-side bit-vectors.
                # 32 chunks; each output ~ (n, d+1, n, d+1, 2**5) ~ small.
                for fb, src_b in fams_b:
                    fix = [None] * 5 + list(src_b)
                    master = _master_moment_partial(
                        ta2[tta], tb2[ttb], 0, 1, S, fix
                    )
                    for fa, src_a in fams_a:
                        idx = (slice(None),) * 4 + tuple(src_a)
                        emit(fa, fb, master[idx])
                    del master
                if S.device.type == "cuda":
                    torch.cuda.empty_cache()

    return matrix, sum(matrix.values())


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _fam_key(f) -> str:
    """JSON-friendly family key."""
    if isinstance(f, str):
        return f
    return f"{f[0]}:{f[1]}"


def _matrix_to_dict(matrix) -> dict:
    return {f"{_fam_key(fa)}|{_fam_key(fb)}": v for (fa, fb), v in matrix.items()}


def run_pair(step_a: int, step_b: int, device: torch.device, dtype: torch.dtype,
             mc_n_samples: int = 100_000, mc_batch: int = 1024):
    print(f"\n=== Pair ({step_a}, {step_b}) ===", flush=True)

    A = load_component(step_a, device, dtype)
    B = load_component(step_b, device, dtype)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    timings = {}
    matrices = {}
    totals = {}

    t0 = time.perf_counter()
    matrices["AA"], totals["AA"] = family_pair_no_embed(A, A)
    if device.type == "cuda":
        torch.cuda.synchronize()
    timings["AA_sec"] = time.perf_counter() - t0
    print(f"  M_AA  total={totals['AA']:.6e}  time={timings['AA_sec']:.2f}s", flush=True)

    t0 = time.perf_counter()
    matrices["BB"], totals["BB"] = family_pair_no_embed(B, B)
    if device.type == "cuda":
        torch.cuda.synchronize()
    timings["BB_sec"] = time.perf_counter() - t0
    print(f"  M_BB  total={totals['BB']:.6e}  time={timings['BB_sec']:.2f}s", flush=True)

    t0 = time.perf_counter()
    matrices["AB"], totals["AB"] = family_pair_no_embed(A, B)
    if device.type == "cuda":
        torch.cuda.synchronize()
    timings["AB_sec"] = time.perf_counter() - t0
    print(f"  M_AB  total={totals['AB']:.6e}  time={timings['AB_sec']:.2f}s", flush=True)

    timings["tn_total_sec"] = sum(timings.values())

    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        peak_mem_mb = float("nan")

    # Cosine
    denom = (totals["AA"] * totals["BB"]) ** 0.5
    tn_cos = totals["AB"] / denom if denom > 0 else float("nan")

    # MC baseline
    print("  Running MC sim ...", flush=True)
    A_mc = load_attentionlm_for_mc(step_a, device, dtype)
    B_mc = load_attentionlm_for_mc(step_b, device, dtype)
    t0 = time.perf_counter()
    mc_cos = mc_similarity(A_mc, B_mc, device=device,
                           n_samples=mc_n_samples, batch_size=mc_batch, dtype=dtype)
    mc_time = time.perf_counter() - t0
    print(f"  TN cos = {tn_cos:.6f}    MC cos ({mc_n_samples} samples) = {mc_cos:.6f}    "
          f"|diff| = {abs(tn_cos - mc_cos):.3e}    mc_time={mc_time:.1f}s", flush=True)

    # ---- save ----
    norm_factor = denom if denom > 0 else 1.0
    matrix_AB_norm = {k: v / norm_factor for k, v in matrices["AB"].items()}

    payload = {
        "pair": [step_a, step_b],
        "n_families": 34,
        "totals": totals,
        "tn_cosine": tn_cos,
        "mc_cosine": mc_cos,
        "mc_n_samples": mc_n_samples,
        "timings_sec": timings,
        "mc_time_sec": mc_time,
        "peak_gpu_mem_mb": peak_mem_mb,
        "matrix_AB":      _matrix_to_dict(matrices["AB"]),
        "matrix_AA":      _matrix_to_dict(matrices["AA"]),
        "matrix_BB":      _matrix_to_dict(matrices["BB"]),
        "matrix_AB_normalised": _matrix_to_dict(matrix_AB_norm),
        "device": str(device),
        "dtype": str(dtype),
    }
    out_json = OUT_DIR / f"decomp_{step_a}_{step_b}.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"  saved -> {out_json}")

    # also a torch .pt with full-precision matrices and family-key tuples
    torch.save({
        "pair": (step_a, step_b),
        "matrix_AB": matrices["AB"],
        "matrix_AA": matrices["AA"],
        "matrix_BB": matrices["BB"],
        "totals": totals,
        "tn_cosine": tn_cos,
        "mc_cosine": mc_cos,
        "timings_sec": timings,
        "mc_time_sec": mc_time,
        "peak_gpu_mem_mb": peak_mem_mb,
    }, OUT_DIR / f"decomp_{step_a}_{step_b}.pt")

    # ---- cumulative plot ----
    plot_cumulative(matrices["AB"], norm_factor, step_a, step_b)

    return payload


def plot_cumulative(matrix_AB, norm_factor, step_a, step_b):
    """Plot cumulative similarity in ascending order of family-pair contribution."""
    items = sorted(matrix_AB.items(), key=lambda kv: kv[1])
    values = [v for _, v in items]
    raw_cum = []
    s = 0.0
    for v in values:
        s += v
        raw_cum.append(s)
    norm_cum = [c / norm_factor for c in raw_cum] if norm_factor > 0 else raw_cum

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    n = len(values)

    axes[0].plot(range(1, n + 1), raw_cum, lw=1.2)
    axes[0].set_xlabel("family-pair index (sorted asc)")
    axes[0].set_ylabel(r"cumulative $\langle F_\rho, \tilde F_\sigma\rangle$")
    axes[0].set_title("raw")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].grid(alpha=0.3)

    axes[1].plot(range(1, n + 1), norm_cum, lw=1.2, color="C1")
    axes[1].set_xlabel("family-pair index (sorted asc)")
    axes[1].set_ylabel(r"cumulative / $\sqrt{\langle A,A\rangle\langle B,B\rangle}$")
    axes[1].set_title("globally normalised (sums to TN cosine)")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Cumulative TN similarity contributions — steps ({step_a}, {step_b})")
    fig.tight_layout()
    out_path = OUT_DIR / f"cumulative_{step_a}_{step_b}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot  -> {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    print(f"device={device}, dtype={dtype}", flush=True)

    pairs = [(700, 1000), (2500, 5000)]
    summary = {}
    for sa, sb in pairs:
        payload = run_pair(sa, sb, device, dtype)
        summary[f"{sa}_{sb}"] = {
            "totals": payload["totals"],
            "tn_cosine": payload["tn_cosine"],
            "mc_cosine": payload["mc_cosine"],
            "timings_sec": payload["timings_sec"],
            "mc_time_sec": payload["mc_time_sec"],
            "peak_gpu_mem_mb": payload["peak_gpu_mem_mb"],
        }
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
