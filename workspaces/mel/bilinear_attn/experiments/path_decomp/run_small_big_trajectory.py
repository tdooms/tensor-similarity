"""Per-family-pair TN-similarity trajectory vs final checkpoint on
`small_big_experiment`.

Parallel, incremental version.

Workers run in separate processes (``torch.multiprocessing.spawn``) sharing
one GPU — ``NUM_WORKERS`` processes, each with its own CUDA context and its
own copy of the final checkpoint ``B``. Steps are dispatched round-robin.

Each worker writes ``per_step/step_<step>.json`` (+ ``.pt``) atomically
**immediately after** the step's TN computation finishes, so partial progress
is saved and workers can be killed/restarted (existing per-step files are
skipped).

After all workers finish, the main process aggregates the per-step files
into ``trajectory.json`` / ``trajectory.pt`` and produces the plots:

  * ``traj_global_all.png``                 all 1156 family-pairs, globally normalised
  * ``traj_local_all.png``                  all 1156 family-pairs, locally  normalised
  * ``traj_global_top{K}.png``              top-K by max |global_norm|
  * ``traj_local_top{K}.png``               top-K by max |local_norm| (after
                                            norm-thresholding)
  * ``traj_global_top3.png`` / ``traj_local_top3.png``
  * ``traj_tn_cosine.png``                  total TN cosine line
  * ``traj_family_31_31.png``               explicit ``('layer2', 31) x ('layer2', 31)``

Two sigma modes:
  * ``identity``:  Σ = I_{d_model}   (matches ``mc_similarity``)
  * ``onehot``:    Σ = E_L^T E_R     (matches ``make_gaussian_onehot_sampler``)
"""
from __future__ import annotations

import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
WS_ROOT = HERE.parent.parent  # .../bilinear_attn
RUN_DIR = WS_ROOT / "experiments" / "induction_heads" / "runs" / "small_big_experiment"

REPO_ROOT = WS_ROOT.parent.parent.parent  # .../tensor-mars
for p in (str(REPO_ROOT), str(WS_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
STRIDE = 10
FINAL_STEP = 15000
TOP_K_GLOBAL = 10
TOP_K_LOCAL = 10
TOP_K_GLOBAL_SMALL = 3
TOP_K_LOCAL_SMALL = 3
LOCAL_NORM_THRESHOLD = 1e-3
SIGMA_MODE = "onehot"              # 'identity' | 'onehot'
EXPLICIT_PAIR = (("layer2", 31), ("layer2", 31))

NUM_WORKERS = 10                   # processes sharing GPU 0
WORKER_DEVICE = "cuda:0"           # all workers use this device
DTYPE_STR = "float64"


# --------------------------------------------------------------------------
# Σ-aware family-pair decomposition
# --------------------------------------------------------------------------
# (Imports that pull in torch CUDA kernels happen inside workers to avoid
#  initialising CUDA in the parent before spawn.)

def _sigma_padded(E_left, E_right, d_model, device, dtype):
    Sig = torch.zeros(d_model + 1, d_model + 1, device=device, dtype=dtype)
    Sig[0, 0] = 1.0
    if E_left is not None and E_right is not None:
        Sig[1:, 1:] = E_left.T @ E_right
    else:
        Sig[1:, 1:] = torch.eye(d_model, device=device, dtype=dtype)
    return Sig


@torch.no_grad()
def family_pair_with_sigma(model_a, model_b, sigma_mode: str):
    """34x34 family-pair inner products with configurable residual-stream Σ.

    ``sigma_mode``:
      * ``'identity'`` — Σ = I_{d_model}.
      * ``'onehot'``   — Σ_LL = E_a^T E_a, Σ_LR = E_a^T E_b, Σ_RR = E_b^T E_b.
    """
    from src.components.base import Term
    from src.components.similarity import State, _moment
    from workspaces.mel.bilinear_attn.experiments.path_decomp.run_big_experiment import (
        _master_moment_partial,
    )
    from workspaces.mel.bilinear_attn.experiments.path_decomp.moments import (
        _family_to_tt_and_src, _stack_s_split,
    )
    from workspaces.mel.bilinear_attn.experiments.path_decomp.forward import (
        enumerate_families,
    )

    p = next(model_a.parameters())
    device, dtype = p.device, p.dtype
    n_ctx = model_a.n_ctx
    d_model = model_a.d_model
    d_padded = d_model + 1
    like = dict(device=device, dtype=dtype)

    I_n = torch.eye(n_ctx, **like)
    if sigma_mode == "identity":
        Sig_aa = Sig_ab = Sig_bb = _sigma_padded(None, None, d_model, device, dtype)
    elif sigma_mode == "onehot":
        E_a = model_a.embed.weight.to(device=device, dtype=dtype)
        E_b = model_b.embed.weight.to(device=device, dtype=dtype)
        Sig_aa = _sigma_padded(E_a, E_a, d_model, device, dtype)
        Sig_bb = _sigma_padded(E_b, E_b, d_model, device, dtype)
        Sig_ab = _sigma_padded(E_a, E_b, d_model, device, dtype)
    else:
        raise ValueError(f"unknown sigma_mode={sigma_mode!r}")

    make_s0 = lambda Sig: torch.einsum("ij,kl->ikjl", I_n, Sig)
    state = State(make_s0(Sig_aa), make_s0(Sig_ab), make_s0(Sig_bb))

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

    ta2 = model_a.layers[1].terms(n_ctx, **like)
    tb2 = model_b.layers[1].terms(n_ctx, **like)
    assert len(ta2) == 2 and len(tb2) == 2

    th_a = model_a.unembed.terms(n_ctx, **like)
    th_b = model_b.unembed.terms(n_ctx, **like)
    assert len(th_a) == 1 and len(th_b) == 1

    fams = list(enumerate_families())
    matrix = {fp: 0.0 for fp in ((fa, fb) for fa in fams for fb in fams)}

    def emit(fa, fb, s_ab_l2):
        proxy = State(s_ab_l2, s_ab_l2, s_ab_l2)
        s_ab_out = _moment(th_a[0], th_b[0], 0, 1, proxy)
        matrix[(fa, fb)] = torch.einsum("ijij->", s_ab_out[:, 1:, :, 1:]).item()

    for tta in (0, 1):
        for ttb in (0, 1):
            fams_a = [(fa, _family_to_tt_and_src(fa)[1]) for fa in fams
                      if _family_to_tt_and_src(fa)[0] == tta]
            fams_b = [(fb, _family_to_tt_and_src(fb)[1]) for fb in fams
                      if _family_to_tt_and_src(fb)[0] == ttb]
            n_legs_a = 1 if tta == 0 else 5
            n_legs_b = 1 if ttb == 0 else 5

            if (tta, ttb) != (1, 1):
                fix = [None] * (n_legs_a + n_legs_b)
                master = _master_moment_partial(ta2[tta], tb2[ttb], 0, 1, S, fix)
                for fa, src_a in fams_a:
                    for fb, src_b in fams_b:
                        idx = (slice(None),) * 4 + tuple(src_a) + tuple(src_b)
                        emit(fa, fb, master[idx])
                del master
            else:
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


# --------------------------------------------------------------------------
# Misc helpers
# --------------------------------------------------------------------------

def list_checkpoints() -> list[int]:
    ck = RUN_DIR / "checkpoints"
    steps = []
    for f in ck.glob("step_*.pt"):
        m = re.match(r"step_(\d+)\.pt$", f.name)
        if m:
            steps.append(int(m.group(1)))
    steps.sort()
    return steps


def select_steps(all_steps, stride, final):
    picked = all_steps[::stride]
    if final not in picked:
        picked.append(final)
    return sorted(set(picked))


def _fam_key(f) -> str:
    return f if isinstance(f, str) else f"{f[0]}:{f[1]}"


def _fam_from_key(s: str):
    if ":" not in s:
        return s
    head, tail = s.split(":", 1)
    return (head, int(tail))


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------

def worker(rank: int, world_size: int, steps: list[int], final_step: int,
           sigma_mode: str, device_str: str, dtype_str: str,
           run_dir_str: str, per_step_dir_str: str):
    """Process-level worker. Handles steps[rank::world_size]."""
    import torch

    # Imports that touch src/models happen inside the worker.
    from workspaces.mel.bilinear_attn.experiments.path_decomp import run_big_experiment as rbe
    from workspaces.mel.bilinear_attn.experiments.path_decomp.forward import enumerate_families

    rbe.RUN_DIR = Path(run_dir_str)
    device = torch.device(device_str)
    dtype = getattr(torch, dtype_str)
    per_step_dir = Path(per_step_dir_str)
    my_steps = [s for i, s in enumerate(steps) if i % world_size == rank]
    tag = f"[w{rank}/{world_size}]"
    print(f"{tag} starting, {len(my_steps)} steps: {my_steps}", flush=True)

    fams = list(enumerate_families())
    fam_pairs = [(fa, fb) for fa in fams for fb in fams]

    # Load B + compute M_BB once per worker (shared is harder than it's worth).
    B = rbe.load_component(final_step, device, dtype)
    t0 = time.perf_counter()
    M_BB, total_BB = family_pair_with_sigma(B, B, sigma_mode)
    print(f"{tag} M_BB done, total_BB={total_BB:.3e}, "
          f"t={time.perf_counter()-t0:.1f}s", flush=True)
    BB_diag = {fb: M_BB[(fb, fb)] for fb in fams}

    for step in my_steps:
        json_path = per_step_dir / f"step_{step}.json"
        if json_path.exists():
            print(f"{tag} skip step {step} (exists).", flush=True)
            continue
        print(f"{tag} step {step} ...", flush=True)

        A = rbe.load_component(step, device, dtype)
        t0 = time.perf_counter()
        M_AA, total_AA = family_pair_with_sigma(A, A, sigma_mode)
        t_aa = time.perf_counter() - t0
        t0 = time.perf_counter()
        M_AB, total_AB = family_pair_with_sigma(A, B, sigma_mode)
        t_ab = time.perf_counter() - t0

        denom_global = (total_AA * total_BB) ** 0.5
        tn_cos = total_AB / denom_global if denom_global > 0 else float("nan")

        raw_AB = [M_AB[p] for p in fam_pairs]
        AA_diag = {fa: M_AA[(fa, fa)] for fa in fams}
        global_norm = ([v / denom_global for v in raw_AB] if denom_global > 0
                       else [0.0] * len(raw_AB))
        local_norm = []
        for (fa, fb) in fam_pairs:
            aa = AA_diag[fa]
            bb = BB_diag[fb]
            if aa > 0 and bb > 0:
                local_norm.append(M_AB[(fa, fb)] / ((aa * bb) ** 0.5))
            else:
                local_norm.append(float("nan"))

        payload = {
            "step": step,
            "final_step": final_step,
            "sigma_mode": sigma_mode,
            "total_AA": total_AA,
            "total_AB": total_AB,
            "total_BB": total_BB,
            "tn_cos": tn_cos,
            "t_aa_sec": t_aa,
            "t_ab_sec": t_ab,
            "family_pairs": [f"{_fam_key(fa)}|{_fam_key(fb)}" for fa, fb in fam_pairs],
            "families": [_fam_key(f) for f in fams],
            "raw_AB": raw_AB,
            "AA_diag": [AA_diag[f] for f in fams],
            "BB_diag": [BB_diag[f] for f in fams],
            "global_norm": global_norm,
            "local_norm": local_norm,
        }
        _atomic_write_json(json_path, payload)
        # Also save a .pt mirror of M_AB/M_AA (full precision) — useful later.
        torch.save({
            "step": step, "sigma_mode": sigma_mode,
            "M_AB": M_AB, "M_AA": M_AA, "M_BB": M_BB,
            "total_AA": total_AA, "total_AB": total_AB, "total_BB": total_BB,
        }, per_step_dir / f"step_{step}.pt")
        print(f"{tag} step {step} done tn_cos={tn_cos:.4f} "
              f"aa={t_aa:.1f}s ab={t_ab:.1f}s  -> {json_path.name}",
              flush=True)

        del A, M_AA, M_AB
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"{tag} finished.", flush=True)


# --------------------------------------------------------------------------
# Aggregation + plotting
# --------------------------------------------------------------------------

def aggregate_and_plot(steps: list[int], per_step_dir: Path, out_dir: Path,
                       sigma_mode: str):
    from workspaces.mel.bilinear_attn.experiments.path_decomp.forward import (
        enumerate_families,
    )

    fams = list(enumerate_families())
    fam_pairs = [(fa, fb) for fa in fams for fb in fams]
    pair_index = {p: i for i, p in enumerate(fam_pairs)}
    pair_labels = [f"{_fam_key(fa)}|{_fam_key(fb)}" for fa, fb in fam_pairs]
    n_pairs = len(fam_pairs)

    rows = {}
    for step in steps:
        p = per_step_dir / f"step_{step}.json"
        if not p.exists():
            print(f"missing {p.name}; skipping.")
            continue
        rows[step] = json.loads(p.read_text())

    done_steps = sorted(rows)
    if not done_steps:
        print("no per-step files found; aborting aggregation.")
        return
    T = len(done_steps)

    raw_AB = torch.zeros(T, n_pairs, dtype=torch.float64)
    AA_diag = torch.zeros(T, len(fams), dtype=torch.float64)
    global_norm = torch.zeros(T, n_pairs, dtype=torch.float64)
    local_norm = torch.full((T, n_pairs), float("nan"), dtype=torch.float64)
    tn_cos_arr = torch.zeros(T, dtype=torch.float64)
    total_AA_arr = torch.zeros(T, dtype=torch.float64)
    total_AB_arr = torch.zeros(T, dtype=torch.float64)
    BB_diag = torch.tensor(rows[done_steps[0]]["BB_diag"], dtype=torch.float64)

    for ti, step in enumerate(done_steps):
        r = rows[step]
        raw_AB[ti] = torch.tensor(r["raw_AB"], dtype=torch.float64)
        AA_diag[ti] = torch.tensor(r["AA_diag"], dtype=torch.float64)
        global_norm[ti] = torch.tensor(r["global_norm"], dtype=torch.float64)
        local_norm[ti] = torch.tensor(
            [float("nan") if v is None else v for v in r["local_norm"]],
            dtype=torch.float64,
        )
        tn_cos_arr[ti] = r["tn_cos"]
        total_AA_arr[ti] = r["total_AA"]
        total_AB_arr[ti] = r["total_AB"]

    total_BB = rows[done_steps[0]]["total_BB"]
    sigma_tag = f"({sigma_mode})"
    FINAL = rows[done_steps[0]]["final_step"]

    # Save aggregated payload.
    payload = {
        "run_dir": str(RUN_DIR),
        "sigma_mode": sigma_mode,
        "final_step": FINAL,
        "stride": STRIDE,
        "steps": done_steps,
        "family_pairs": pair_labels,
        "families": [_fam_key(f) for f in fams],
        "total_BB": total_BB,
        "BB_diag": BB_diag.tolist(),
        "raw_AB": raw_AB.tolist(),
        "AA_diag": AA_diag.tolist(),
        "global_norm": global_norm.tolist(),
        "local_norm": local_norm.tolist(),
        "tn_cos": tn_cos_arr.tolist(),
        "total_AA": total_AA_arr.tolist(),
        "total_AB": total_AB_arr.tolist(),
    }
    _atomic_write_json(out_dir / "trajectory.json", payload)
    torch.save({
        "steps": done_steps,
        "family_pairs": fam_pairs,
        "sigma_mode": sigma_mode,
        "BB_diag": BB_diag,
        "total_BB": total_BB,
        "raw_AB": raw_AB,
        "AA_diag": AA_diag,
        "global_norm": global_norm,
        "local_norm": local_norm,
        "tn_cos": tn_cos_arr,
        "total_AA": total_AA_arr,
        "total_AB": total_AB_arr,
    }, out_dir / "trajectory.pt")
    print(f"saved -> {out_dir / 'trajectory.json'}")

    # Rank by max |.| over trajectory
    g_max_abs = global_norm.abs().nan_to_num(0.0).max(dim=0).values
    top_global_idx = torch.topk(g_max_abs, TOP_K_GLOBAL).indices.tolist()
    top_global_idx_small = torch.topk(g_max_abs, TOP_K_GLOBAL_SMALL).indices.tolist()

    fam_aa_max = AA_diag.max(dim=0).values
    fam_aa_min = AA_diag.min(dim=0).values
    fam_idx = {f: i for i, f in enumerate(fams)}
    valid_pair = torch.zeros(n_pairs, dtype=torch.bool)
    for pi, (fa, fb) in enumerate(fam_pairs):
        ai, bi = fam_idx[fa], fam_idx[fb]
        aa_max = float(fam_aa_max[ai].item())
        aa_min = float(fam_aa_min[ai].item())
        bb = float(BB_diag[bi].item())
        if aa_max > 0 and bb > 0 and aa_min >= LOCAL_NORM_THRESHOLD * aa_max:
            valid_pair[pi] = True
    print(f"local-rank: {int(valid_pair.sum().item())}/{n_pairs} pairs pass "
          f"threshold (LOCAL_NORM_THRESHOLD={LOCAL_NORM_THRESHOLD}).")

    l_max_abs = local_norm.abs().nan_to_num(0.0).max(dim=0).values.clone()
    l_max_abs[~valid_pair] = -float("inf")
    top_local_idx = torch.topk(l_max_abs, TOP_K_LOCAL).indices.tolist()
    top_local_idx_small = torch.topk(l_max_abs, TOP_K_LOCAL_SMALL).indices.tolist()

    # Plots
    plot_trajectory(done_steps, global_norm, pair_labels,
                    title=f"All {n_pairs} family-pair contributions, globally normalised {sigma_tag}\n"
                          rf"$M_{{AB}}/\sqrt{{\langle A,A\rangle\langle B,B\rangle}}$,  B=step_{FINAL}",
                    out_path=out_dir / "traj_global_all.png", alpha=0.15)
    plot_trajectory(done_steps, local_norm, pair_labels,
                    title=f"All {n_pairs} family-pair contributions, locally normalised {sigma_tag}\n"
                          rf"$M_{{AB}}/\sqrt{{M_{{AA}}M_{{BB}}}}$,  B=step_{FINAL}",
                    out_path=out_dir / "traj_local_all.png", alpha=0.15)
    plot_trajectory(done_steps, global_norm, pair_labels,
                    title=f"Top-{TOP_K_GLOBAL} family-pairs by max |global| {sigma_tag}, B=step_{FINAL}",
                    out_path=out_dir / f"traj_global_top{TOP_K_GLOBAL}.png",
                    highlight=top_global_idx, only_highlight=True)
    plot_trajectory(done_steps, local_norm, pair_labels,
                    title=f"Top-{TOP_K_LOCAL} family-pairs by max |local| "
                          f"(norm-thresh={LOCAL_NORM_THRESHOLD}) {sigma_tag}, B=step_{FINAL}",
                    out_path=out_dir / f"traj_local_top{TOP_K_LOCAL}.png",
                    highlight=top_local_idx, only_highlight=True)
    plot_trajectory(done_steps, global_norm, pair_labels,
                    title=f"Top-{TOP_K_GLOBAL_SMALL} family-pairs by max |global| {sigma_tag}, B=step_{FINAL}",
                    out_path=out_dir / f"traj_global_top{TOP_K_GLOBAL_SMALL}.png",
                    highlight=top_global_idx_small, only_highlight=True)
    plot_trajectory(done_steps, local_norm, pair_labels,
                    title=f"Top-{TOP_K_LOCAL_SMALL} family-pairs by max |local| "
                          f"(norm-thresh={LOCAL_NORM_THRESHOLD}) {sigma_tag}, B=step_{FINAL}",
                    out_path=out_dir / f"traj_local_top{TOP_K_LOCAL_SMALL}.png",
                    highlight=top_local_idx_small, only_highlight=True)

    # TN cosine
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(done_steps, tn_cos_arr.cpu().numpy(), marker="o", lw=1.5, color="C2")
    ax.axhline(1.0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("step (A)"); ax.set_ylabel("TN cosine vs final")
    ax.set_title(f"Total TN cosine to step_{FINAL} {sigma_tag}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "traj_tn_cosine.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot -> {out_dir / 'traj_tn_cosine.png'}")

    # Explicit family 31|31
    fa, fb = EXPLICIT_PAIR
    if (fa, fb) in pair_index:
        pi = pair_index[(fa, fb)]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(done_steps, raw_AB[:, pi].cpu().numpy(), marker="o", lw=1.5)
        axes[0].set_title("raw $\\langle F,\\tilde F\\rangle$")
        axes[1].plot(done_steps, global_norm[:, pi].cpu().numpy(), marker="o", lw=1.5, color="C1")
        axes[1].set_title("globally normalised")
        axes[2].plot(done_steps, local_norm[:, pi].cpu().numpy(), marker="o", lw=1.5, color="C3")
        axes[2].set_title("locally normalised (per-pair cosine)")
        for a in axes:
            a.axhline(0, color="k", lw=0.5); a.set_xlabel("step (A)"); a.grid(alpha=0.3)
        fig.suptitle(f"family-pair {_fam_key(fa)} x {_fam_key(fb)} {sigma_tag}, B=step_{FINAL}")
        fig.tight_layout()
        fig.savefig(out_dir / "traj_family_31_31.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  plot -> {out_dir / 'traj_family_31_31.png'}")


def plot_trajectory(steps, values: torch.Tensor, labels, *, title, out_path,
                    alpha=0.2, highlight=None, only_highlight=False):
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = steps
    n_pairs = values.shape[1]
    if only_highlight and highlight is not None:
        for pi in highlight:
            ax.plot(xs, values[:, pi].cpu().numpy(), lw=1.6, label=labels[pi])
        ax.legend(fontsize=8, loc="best")
    else:
        for pi in range(n_pairs):
            ax.plot(xs, values[:, pi].cpu().numpy(), lw=0.5, color="C0", alpha=alpha)
        if highlight is not None:
            for pi in highlight:
                ax.plot(xs, values[:, pi].cpu().numpy(), lw=1.5, label=labels[pi])
            ax.legend(fontsize=8, loc="best")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("step (A)"); ax.set_ylabel("per-pair contribution")
    ax.set_title(title); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot -> {out_path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    out_dir = RUN_DIR / f"path_decomp_trajectory_{SIGMA_MODE}"
    per_step_dir = out_dir / "per_step"
    per_step_dir.mkdir(parents=True, exist_ok=True)

    all_steps = list_checkpoints()
    assert FINAL_STEP in all_steps, f"final step {FINAL_STEP} not found"
    steps = select_steps(all_steps, STRIDE, FINAL_STEP)
    print(f"out_dir={out_dir}")
    print(f"sigma_mode={SIGMA_MODE}  num_workers={NUM_WORKERS}  device={WORKER_DEVICE}")
    print(f"using {len(steps)} checkpoints: {steps}", flush=True)

    # Skip-aware work tally.
    pending = [s for s in steps if not (per_step_dir / f"step_{s}.json").exists()]
    print(f"{len(pending)}/{len(steps)} steps pending "
          f"(skipping existing per-step files).", flush=True)

    if pending:
        world = min(NUM_WORKERS, len(pending))
        args = (world, pending, FINAL_STEP, SIGMA_MODE, WORKER_DEVICE, DTYPE_STR,
                str(RUN_DIR), str(per_step_dir))
        if world == 1:
            worker(0, *args)
        else:
            mp.spawn(worker, args=args, nprocs=world, join=True)

    print("\n=== aggregating ===", flush=True)
    aggregate_and_plot(steps, per_step_dir, out_dir, SIGMA_MODE)
    print("done")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
