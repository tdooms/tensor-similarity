"""Per-family MC similarity with alternative input samplers.

Motivation: the Gaussian-over-residual-stream MC estimator in
``run_big_experiment_mc.py`` is numerically dominated by the ``layer2:31``
family (all 5 QK/V slots active) because an isotropic input in d_model has
much heavier tails through the polynomial kernel than the model ever sees.
That hides which families actually drive the trained model's output.

This driver re-runs the 34x34 MC decomposition under two alternative
samplers (see ``mc_per_family.py``):

    - ``tokens``          : sample discrete token ids from the training-data
                            generator (``experiments.induction_heads.data``)
                            and embed them with each model's ``W_E``.
    - ``gaussian_onehot`` : sample ``z ~ N(0, I_V)`` at every position and
                            feed ``z @ W_E`` into each model.

For each (pair, sampler) we write the same artifacts as
``run_big_experiment_mc.py`` — plus a TN-sim cross-check on the top-k
family pairs (default k=3) so we can see whether the MC ranking under
each sampler matches what the TN inner product (over the full Gaussian
residual-stream measure) reports.

Output layout:
    runs/<run>/path_decomp_samplers/<sampler>/<sa>_<sb>/
        decomp.json, decomp.pt,
        cumulative_global.png, cumulative_per_fam_cos.png,
        top_global.json, top_per_fam_cos.json,
        tn_top_families.json
    runs/<run>/path_decomp_samplers/<sampler>/summary.json
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
WS_ROOT = HERE.parent.parent  # .../bilinear_attn
RUNS_ROOT = WS_ROOT / "experiments" / "induction_heads" / "runs"
RUN_DIR: Path = RUNS_ROOT / "small_big_experiment"
OUT_DIR: Path = RUN_DIR / "path_decomp_samplers"

REPO_ROOT = WS_ROOT.parent.parent.parent  # .../tensor-mars
for p in (str(REPO_ROOT), str(WS_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from workspaces.mel.bilinear_attn.experiments.path_decomp.mc_per_family import (  # noqa: E402
    FAMILY_LIST, FAMILY_INDEX, N_FAMILIES, mc_family_pairs, _strip_norms,
    make_gaussian_resid_sampler, make_gaussian_onehot_sampler, make_token_sampler,
)
from workspaces.mel.bilinear_attn.experiments.path_decomp.run_big_experiment_mc import (  # noqa: E402
    load_attnlm, load_component, tn_single_family_pair,
    per_family_cosine, _entries_sorted, _cumulative_plot, _top_json,
    _matrix_to_dict, _fam_key,
)


# ---------------------------------------------------------------------------
# Sampler factory
# ---------------------------------------------------------------------------

def build_sampler(name: str, model_A, model_B, device, dtype, cfg: dict, seed: int):
    if name == 'gaussian_resid':
        return make_gaussian_resid_sampler(model_A, model_B, device, dtype)
    if name == 'gaussian_onehot':
        return make_gaussian_onehot_sampler(model_A, model_B, device, dtype)
    if name == 'tokens':
        data_cfg = cfg.get('data', {})
        model_cfg = cfg['model']
        use_bos = data_cfg.get('use_bos', False)
        bos = data_cfg.get('bos_token_id', None)
        if use_bos and bos is None:
            bos = model_cfg['vocab_size'] - 1
        if not use_bos:
            bos = None
        return make_token_sampler(
            model_A, model_B, device, dtype,
            vocab_size=model_cfg['vocab_size'], bos_token_id=bos,
            pool_size=20_000, seed=seed,
        )
    raise ValueError(f"unknown sampler: {name}")


# ---------------------------------------------------------------------------
# Per-pair driver (mirrors run_big_experiment_mc.run_pair, TN check on top-k)
# ---------------------------------------------------------------------------

def run_pair(sampler_name: str, step_a: int, step_b: int,
             device: torch.device, dtype: torch.dtype,
             cfg: dict, n_samples: int, batch_size: int,
             tn_topk: int, seed: int) -> dict:
    pair_dir = OUT_DIR / sampler_name / f"{step_a}_{step_b}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== [{sampler_name}] Pair ({step_a}, {step_b})  "
          f"n_samples={n_samples}  batch_size={batch_size} ===", flush=True)

    A_mc = load_attnlm(step_a, device, dtype)
    B_mc = load_attnlm(step_b, device, dtype)

    sampler = build_sampler(sampler_name, A_mc, B_mc, device, dtype, cfg, seed)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

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
    denom = (total_AA * total_BB) ** 0.5 if total_AA > 0 and total_BB > 0 else 0.0
    mc_cosine = total_AB / denom if denom > 0 else float("nan")
    norm_global = denom if denom > 0 else 1.0

    cos_mat = per_family_cosine(mats)
    M_AB_global_norm = mats['AB'] / norm_global

    peak_mem_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                   if device.type == "cuda" else float("nan"))
    print(f"  totals: AA={total_AA:.6e}  BB={total_BB:.6e}  AB={total_AB:.6e}")
    print(f"  MC cosine = {mc_cosine:.6f}   mc_time={mc_elapsed:.1f}s   "
          f"peak_mem={peak_mem_mb:.1f} MB")

    # --- Ranked entries ---
    ent_global = _entries_sorted(M_AB_global_norm)
    ent_percos = _entries_sorted(cos_mat)

    print(f"  top-{tn_topk} by |global-norm|:")
    for r, (fa, fb, v) in enumerate(ent_global[:tn_topk], 1):
        i, j = FAMILY_INDEX[fa], FAMILY_INDEX[fb]
        print(f"    {r:2d}. {_fam_key(fa)} x {_fam_key(fb)}  "
              f"v/denom={v:+.6f}   per-fam-cos={float(cos_mat[i, j].item()):+.6f}")

    # --- TN check on top-k family pairs ---
    print(f"  Running TN on top-{tn_topk} family pairs ...", flush=True)
    A_tn = load_component(step_a, device, dtype)
    B_tn = load_component(step_b, device, dtype)

    tn_records = []
    t0 = time.perf_counter()
    for r, (fa, fb, v_mc_gn) in enumerate(ent_global[:tn_topk], 1):
        i, j = FAMILY_INDEX[fa], FAMILY_INDEX[fb]
        tn_AB = tn_single_family_pair(A_tn, B_tn, fa, fb)
        tn_AA = tn_single_family_pair(A_tn, A_tn, fa, fa)
        tn_BB = tn_single_family_pair(B_tn, B_tn, fb, fb)
        mc_AB = float(mats['AB'][i, j].item())
        mc_AA = float(mats['AA'][i, i].item())
        mc_BB = float(mats['BB'][j, j].item())
        tn_pfc = (tn_AB / (tn_AA * tn_BB) ** 0.5) if tn_AA * tn_BB > 0 else float("nan")
        mc_pfc = float(cos_mat[i, j].item())
        tn_records.append({
            "rank": r,
            "fa": _fam_key(fa), "fb": _fam_key(fb),
            "tn": {"AB": tn_AB, "AA": tn_AA, "BB": tn_BB,
                   "per_family_cosine": tn_pfc},
            "mc": {"AB": mc_AB, "AA": mc_AA, "BB": mc_BB,
                   "per_family_cosine": mc_pfc,
                   "global_norm_contribution": v_mc_gn},
        })
        print(f"    rank {r}: {_fam_key(fa)} x {_fam_key(fb)}   "
              f"TN pfc={tn_pfc:+.4f}  MC pfc={mc_pfc:+.4f}   "
              f"TN AB={tn_AB:+.3e}  MC AB={mc_AB:+.3e}")
    tn_elapsed = time.perf_counter() - t0

    (pair_dir / "tn_top_families.json").write_text(json.dumps({
        "sampler": sampler_name,
        "pair": [step_a, step_b],
        "tn_topk": tn_topk,
        "n_samples": n_samples,
        "tn_elapsed_sec": tn_elapsed,
        "records": tn_records,
    }, indent=2))

    # --- Save matrices ---
    payload = {
        "sampler": sampler_name,
        "pair": [step_a, step_b],
        "n_families": N_FAMILIES,
        "n_samples": n_samples,
        "batch_size": batch_size,
        "totals": {"AA": total_AA, "BB": total_BB, "AB": total_AB},
        "mc_cosine": mc_cosine,
        "mc_elapsed_sec": mc_elapsed,
        "tn_elapsed_sec": tn_elapsed,
        "peak_gpu_mem_mb": peak_mem_mb,
        "top_family_pair": [_fam_key(ent_global[0][0]), _fam_key(ent_global[0][1])],
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
        "sampler": sampler_name,
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

    # --- Plots ---
    stats_global = _cumulative_plot(
        ent_global,
        ylabel_signed=r"cumulative $v/\sqrt{\langle A,A\rangle\langle B,B\rangle}$",
        ylabel_abs=r"cumulative $|v|/\sqrt{\langle A,A\rangle\langle B,B\rangle}$",
        title=f"MC [{sampler_name}] — globally normalised — steps ({step_a}, {step_b})",
        out_path=pair_dir / "cumulative_global.png",
    )
    print(f"  plot  -> {pair_dir / 'cumulative_global.png'}   "
          f"rank90={stats_global['rank90']} rank95={stats_global['rank95']}")
    _top_json(ent_global, pair_dir / "top_global.json")

    stats_percos = _cumulative_plot(
        ent_percos,
        ylabel_signed=r"cumulative per-family cosine",
        ylabel_abs=r"cumulative $|$per-family cosine$|$",
        title=f"MC [{sampler_name}] — per-family cosine — steps ({step_a}, {step_b})",
        out_path=pair_dir / "cumulative_per_fam_cos.png",
    )
    print(f"  plot  -> {pair_dir / 'cumulative_per_fam_cos.png'}   "
          f"rank90={stats_percos['rank90']} rank95={stats_percos['rank95']}")
    _top_json(ent_percos, pair_dir / "top_per_fam_cos.json")

    return {
        "sampler": sampler_name,
        "pair": [step_a, step_b],
        "totals": payload["totals"],
        "mc_cosine": mc_cosine,
        "mc_elapsed_sec": mc_elapsed,
        "tn_elapsed_sec": tn_elapsed,
        "peak_gpu_mem_mb": peak_mem_mb,
        "top_family_pair": payload["top_family_pair"],
        "rank90_global_norm": stats_global["rank90"],
        "rank95_global_norm": stats_global["rank95"],
        "rank90_per_fam_cos": stats_percos["rank90"],
        "rank95_per_fam_cos": stats_percos["rank95"],
        "top_tn_vs_mc": [
            {"rank": r["rank"], "fa": r["fa"], "fb": r["fb"],
             "tn_per_family_cosine": r["tn"]["per_family_cosine"],
             "mc_per_family_cosine": r["mc"]["per_family_cosine"]}
            for r in tn_records
        ],
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="small_big_experiment",
                    help="name under experiments/induction_heads/runs/")
    ap.add_argument("--samplers", default="tokens,gaussian_onehot",
                    help="comma-separated: tokens,gaussian_onehot,gaussian_resid")
    ap.add_argument("--n-samples", type=int, default=200_000)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--pairs", default="700,1000;2500,5000",
                    help="';'-separated pairs")
    ap.add_argument("--tn-topk", type=int, default=3,
                    help="number of top family pairs to cross-check with TN")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    global RUN_DIR, OUT_DIR
    RUN_DIR = RUNS_ROOT / args.run_name
    OUT_DIR = RUN_DIR / "path_decomp_samplers"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assert (RUN_DIR / "config.yaml").exists(), f"no config.yaml under {RUN_DIR}"

    # Keep sibling module's RUN_DIR/OUT_DIR aligned so load_attnlm/load_component
    # (imported from run_big_experiment_mc) pick up the right run.
    import workspaces.mel.bilinear_attn.experiments.path_decomp.run_big_experiment_mc as rbe_mc
    rbe_mc.RUN_DIR = RUN_DIR
    rbe_mc.OUT_DIR = OUT_DIR

    cfg = yaml.safe_load((RUN_DIR / "config.yaml").read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    samplers = [s.strip() for s in args.samplers.split(",") if s.strip()]
    pairs = [tuple(int(x) for x in p.split(",")) for p in args.pairs.split(";")]
    print(f"run_dir={RUN_DIR}", flush=True)
    print(f"device={device}  dtype={dtype}  samplers={samplers}  "
          f"pairs={pairs}  n_samples={args.n_samples}  batch_size={args.batch_size}",
          flush=True)

    all_summary = {}
    for sampler_name in samplers:
        sampler_summary = {}
        for sa, sb in pairs:
            sampler_summary[f"{sa}_{sb}"] = run_pair(
                sampler_name, sa, sb, device, dtype, cfg,
                n_samples=args.n_samples, batch_size=args.batch_size,
                tn_topk=args.tn_topk, seed=args.seed,
            )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        out_path = OUT_DIR / sampler_name / "summary.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(sampler_summary, indent=2))
        print(f"\n[{sampler_name}] summary -> {out_path}")
        all_summary[sampler_name] = sampler_summary

    (OUT_DIR / "summary.json").write_text(json.dumps(all_summary, indent=2))
    print(f"\nall-sampler summary -> {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
