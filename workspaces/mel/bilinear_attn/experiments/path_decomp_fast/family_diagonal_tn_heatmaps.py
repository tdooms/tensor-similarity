#!/usr/bin/env python3
"""Fast family-diagonal TN sweep driver."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[5]
for _path in (str(ROOT), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from experiments.path_decomp.no_sym_tn_similarity import load_component
from experiments.path_decomp_fast import install_fast_ctg
from experiments.path_decomp_fast._pair_engine import (
    CANONICAL_FAMILIES,
    CANONICAL_LABELS,
    build_step_artifacts,
    compute_pair,
    family_count,
    family_indices,
)
from experiments.path_decomp_fast.warmup_paths import run_warmup


DEFAULT_STEPS = list(range(0, 15001, 500))


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def choose_dtype(name: str, allow_fp16: bool = False) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "f16": torch.float16,
        "f32": torch.float32,
        "f64": torch.float64,
    }
    if name == "f16" and not allow_fp16:
        raise ValueError("f16 requires --allow_fp16")
    return mapping[name]


def pair_indices(n: int, window: int | None):
    for i in range(n):
        for j in range(i, n):
            if window is None or (j - i) <= window:
                yield i, j


def load_existing(path: Path, n_fam: int, steps: list[int]) -> np.ndarray:
    n = len(steps)
    values = np.full((n_fam, n, n), np.nan, dtype=np.float64)
    if not path.exists():
        return values
    old = np.load(path, allow_pickle=True)
    old_steps = [int(x) for x in old["steps"]]
    old_values = old["family_diag_values"]
    old_index = {s: i for i, s in enumerate(old_steps)}
    for i, step_i in enumerate(steps):
        for j, step_j in enumerate(steps):
            if step_i in old_index and step_j in old_index:
                values[:, i, j] = old_values[:, old_index[step_i], old_index[step_j]]
    return values


def local_normalize(values: np.ndarray) -> np.ndarray:
    sims = np.full_like(values, np.nan)
    for f in range(values.shape[0]):
        diag = np.diag(values[f])
        denom = np.sqrt(np.outer(diag, diag))
        with np.errstate(invalid="ignore", divide="ignore"):
            s = values[f] / denom
        sims[f] = np.where(np.isfinite(s) & np.isfinite(denom) & (denom > 0), s, np.nan)
    return sims


def save_data(path: Path, steps: list[int], values: np.ndarray, sims: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        steps=np.array(steps),
        family_labels=np.array(CANONICAL_LABELS, dtype=object),
        family_diag_values=values,
        family_local_sims=sims,
    )


def plot_family_heatmaps(output_dir: Path, steps: list[int], sims: np.ndarray) -> None:
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for f, label in enumerate(CANONICAL_LABELS):
        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
        im = ax.imshow(np.ma.masked_invalid(sims[f]), vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(f"Family {label}: locally normalized diagonal TN sim")
        ax.set_xlabel("checkpoint step")
        ax.set_ylabel("checkpoint step")
        ax.set_xticks(range(len(steps)))
        ax.set_yticks(range(len(steps)))
        ax.set_xticklabels(steps, rotation=90, fontsize=7)
        ax.set_yticklabels(steps, fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.82).set_label("local cosine")
        fig.savefig(image_dir / f"family_{label}_local_heatmap.png", dpi=220)
        plt.close(fig)


def write_summary_csv(output_dir: Path, sims: np.ndarray, steps: list[int]) -> None:
    rows = []
    final_idx = len(steps) - 1
    for f, label in enumerate(CANONICAL_LABELS):
        row = {"family": label}
        for offset in (1, 2, 5):
            vals = [float(sims[f, i, i + offset]) for i in range(len(steps) - offset) if not np.isnan(sims[f, i, i + offset])]
            row[f"mean_window_offset_{offset}"] = sum(vals) / len(vals) if vals else float("nan")
        row["final_self"] = sims[f, final_idx, final_idx]
        rows.append(row)
    with (output_dir / "family_diag_heatmap_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _run_check(
    run_dir: Path,
    step_i: int,
    step_j: int,
    dtype: torch.dtype,
    device: torch.device,
    families: str,
    bridges_to_f32: bool,
    check_ref_dtype: torch.dtype,
    check_legacy_ref: bool,
) -> float:
    if check_legacy_ref:
        from experiments.path_decomp.family_diagonal_tn_heatmaps import family_diagonal_inner_products_component

        ref_device = torch.device("cpu")
        comp_a_ref = load_component(run_dir, step_i, torch.float64, ref_device)
        comp_b_ref = load_component(run_dir, step_j, torch.float64, ref_device)
        ref = family_diagonal_inner_products_component(comp_a_ref, comp_b_ref)
        ref_vals = np.array([ref[f] for f in CANONICAL_FAMILIES], dtype=np.float64)
        del comp_a_ref, comp_b_ref
    else:
        comp_a_ref = load_component(run_dir, step_i, check_ref_dtype, device)
        comp_b_ref = load_component(run_dir, step_j, check_ref_dtype, device)
        art_a_ref = build_step_artifacts(comp_a_ref)
        art_b_ref = build_step_artifacts(comp_b_ref)
        ref_vals = compute_pair(
            art_a_ref,
            art_b_ref,
            families=families,
            use_orbit_master=False,
            bridges_to_f32=bridges_to_f32,
            is_self=(step_i == step_j),
        ).numpy()
        del comp_a_ref, comp_b_ref, art_a_ref, art_b_ref

    if device.type == "cuda":
        torch.cuda.empty_cache()

    comp_a = load_component(run_dir, step_i, dtype, device)
    comp_b = load_component(run_dir, step_j, dtype, device)
    art_a = build_step_artifacts(comp_a)
    art_b = build_step_artifacts(comp_b)
    fast_vals = compute_pair(
        art_a,
        art_b,
        families=families,
        use_orbit_master=True,
        bridges_to_f32=bridges_to_f32,
        is_self=(step_i == step_j),
    ).numpy()

    mask = np.isfinite(fast_vals) & np.isfinite(ref_vals)
    abs_diff = np.abs(fast_vals[mask] - ref_vals[mask])
    rel_diff = abs_diff / np.maximum(np.abs(ref_vals[mask]), 1e-30)
    print(f"[check] families={families}")
    print(f"[check] max abs diff = {abs_diff.max():.3e}")
    print(f"[check] max rel diff = {rel_diff.max():.3e}")
    return float(rel_diff.max())


def _pending_pairs(values: np.ndarray, steps: list[int], window: int | None, selected_idx: list[int]) -> list[tuple[int, int]]:
    out = []
    for i, j in pair_indices(len(steps), window):
        if np.any(np.isnan(values[selected_idx, i, j])):
            out.append((i, j))
    return out


def _fill_pair(values: np.ndarray, i: int, j: int, row: np.ndarray) -> None:
    finite = np.isfinite(row)
    values[finite, i, j] = row[finite]
    values[finite, j, i] = row[finite]


def _choose_runtime_dtype(requested: str, allow_fp16: bool, run_dir: Path, steps: list[int], device: torch.device, families: str, bridges_to_f32: bool) -> torch.dtype:
    dtype = choose_dtype(requested, allow_fp16=allow_fp16)
    if dtype not in (torch.bfloat16, torch.float16):
        return dtype
    if len(steps) < 2:
        return dtype
    rel = _run_check(
        run_dir,
        steps[0],
        steps[1],
        dtype,
        device,
        families=families,
        bridges_to_f32=bridges_to_f32,
        check_ref_dtype=torch.float32,
        check_legacy_ref=False,
    )
    if rel > 5e-3:
        print(f"auto-fallback {requested} -> f32 (max rel err {rel:.2e})")
        return torch.float32
    return dtype


def _worker_run(worker_id: int, pairs: list[tuple[int, int]], run_dir: str, dtype_name: str, device_name: str, cache_dir: str, shard_path: str, families: str, bridges_to_f32: bool) -> None:
    install_fast_ctg(cache_dir)
    dtype = choose_dtype(dtype_name, allow_fp16=True)
    device = torch.device(device_name)
    artifacts = {}
    rows = []
    for step_i, step_j in pairs:
        if step_i not in artifacts:
            artifacts[step_i] = build_step_artifacts(load_component(Path(run_dir), step_i, dtype, device))
        if step_j not in artifacts:
            artifacts[step_j] = build_step_artifacts(load_component(Path(run_dir), step_j, dtype, device))
        vals = compute_pair(
            artifacts[step_i],
            artifacts[step_j],
            families=families,
            use_orbit_master=True,
            bridges_to_f32=bridges_to_f32,
            is_self=(step_i == step_j),
        ).numpy()
        rows.append((step_i, step_j, vals))
    np.savez(
        shard_path,
        steps_i=np.array([r[0] for r in rows], dtype=np.int64),
        steps_j=np.array([r[1] for r in rows], dtype=np.int64),
        values=np.stack([r[2] for r in rows], axis=0) if rows else np.zeros((0, family_count()), dtype=np.float64),
    )
    print(f"[worker {worker_id}] wrote {len(rows)} pairs")


def _run_multi_process(run_dir: Path, steps: list[int], pending_pairs: list[tuple[int, int]], values: np.ndarray, dtype: torch.dtype, device: torch.device, cache_dir: Path, output_dir: Path, families: str, bridges_to_f32: bool, num_workers: int) -> None:
    import torch.multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass

    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    pending_steps = [(steps[i], steps[j]) for i, j in pending_pairs]
    chunks = [[] for _ in range(num_workers)]
    for k, pair in enumerate(pending_steps):
        chunks[k % num_workers].append(pair)

    ctx = mp.get_context("spawn")
    procs, shard_paths = [], []
    dtype_name = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32", torch.float64: "f64"}[dtype]
    for w, chunk in enumerate(chunks):
        if not chunk:
            continue
        shard_path = shard_dir / f"shard_{w}.npz"
        shard_paths.append(shard_path)
        p = ctx.Process(
            target=_worker_run,
            args=(w, chunk, str(run_dir), dtype_name, str(device), str(cache_dir), str(shard_path), families, bridges_to_f32),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"worker failed: pid={p.pid} code={p.exitcode}")

    step_index = {s: i for i, s in enumerate(steps)}
    for sp in shard_paths:
        data = np.load(sp)
        for si, sj, row in zip(data["steps_i"], data["steps_j"], data["values"]):
            _fill_pair(values, step_index[int(si)], step_index[int(sj)], row)


def _profile_one_pair(run_dir: Path, step_i: int, step_j: int, dtype: torch.dtype, device: torch.device, families: str, bridges_to_f32: bool) -> None:
    t0 = time.perf_counter()
    comp_a = load_component(run_dir, step_i, dtype, device)
    comp_b = load_component(run_dir, step_j, dtype, device)
    t1 = time.perf_counter()
    art_a = build_step_artifacts(comp_a)
    t2 = time.perf_counter()
    art_b = build_step_artifacts(comp_b)
    t3 = time.perf_counter()
    compute_pair(art_a, art_b, families=families, use_orbit_master=True, bridges_to_f32=bridges_to_f32, is_self=(step_i == step_j))
    t4 = time.perf_counter()
    compute_pair(art_a, art_b, families=families, use_orbit_master=True, bridges_to_f32=bridges_to_f32, is_self=(step_i == step_j))
    t5 = time.perf_counter()
    print(f"[profile] load: {t1 - t0:.3f}s")
    print(f"[profile] build a: {t2 - t1:.3f}s")
    print(f"[profile] build b: {t3 - t2:.3f}s")
    print(f"[profile] pair cold: {t4 - t3:.3f}s")
    print(f"[profile] pair warm: {t5 - t4:.3f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", default="experiments/induction_heads/runs/small-big-experiment-runs")
    parser.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "f16", "f32", "f64"))
    parser.add_argument("--allow_fp16", action="store_true")
    parser.add_argument("--families", default="all")
    parser.add_argument("--cache_dir", default=".cache/ctg-paths-fast")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--warmup_only", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check_pair", type=int, nargs=2, default=None)
    parser.add_argument("--check_ref_dtype", default="f32", choices=("f32", "f64"))
    parser.add_argument("--check_legacy_ref", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--recompute_norm", action="store_true")
    parser.add_argument("--bridges_to_f32", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "path_decomp_trajectory" / "family_diag_heatmaps_fast"
    data_path = output_dir / "family_diag_tn_sims.npz"
    cache_dir = Path(args.cache_dir)
    install_fast_ctg(cache_dir)

    if args.recompute_norm:
        old = np.load(data_path, allow_pickle=True)
        values = old["family_diag_values"]
        steps = [int(x) for x in old["steps"]]
        sims = local_normalize(values)
        save_data(data_path, steps, values, sims)
        if not args.skip_plots:
            plot_family_heatmaps(output_dir, steps, sims)
        write_summary_csv(output_dir, sims, steps)
        return

    device = choose_device(args.device)
    selected_idx = family_indices(args.families)

    if args.warmup_only:
        run_warmup(run_dir, args.steps[0], device, cache_dir)
        return

    dtype = _choose_runtime_dtype(
        requested=args.dtype,
        allow_fp16=args.allow_fp16,
        run_dir=run_dir,
        steps=args.steps,
        device=device,
        families=args.families,
        bridges_to_f32=args.bridges_to_f32,
    )

    values = load_existing(data_path, family_count(), args.steps)
    pending = _pending_pairs(values, args.steps, args.window, selected_idx)

    if args.check:
        pair = args.check_pair or (args.steps[0], args.steps[min(1, len(args.steps) - 1)])
        ref_dtype = choose_dtype(args.check_ref_dtype)
        _run_check(
            run_dir,
            int(pair[0]),
            int(pair[1]),
            dtype,
            device,
            families=args.families,
            bridges_to_f32=args.bridges_to_f32,
            check_ref_dtype=ref_dtype,
            check_legacy_ref=args.check_legacy_ref,
        )
        return

    if args.profile:
        i, j = pending[0] if pending else (0, 0)
        _profile_one_pair(run_dir, args.steps[i], args.steps[j], dtype, device, args.families, args.bridges_to_f32)
        return

    if not pending:
        sims = local_normalize(values)
        save_data(data_path, args.steps, values, sims)
        if not args.skip_plots:
            plot_family_heatmaps(output_dir, args.steps, sims)
        write_summary_csv(output_dir, sims, args.steps)
        return

    if args.num_workers > 1 and len(pending) / args.num_workers >= 10:
        _run_multi_process(
            run_dir,
            args.steps,
            pending,
            values,
            dtype,
            device,
            cache_dir,
            output_dir,
            args.families,
            args.bridges_to_f32,
            args.num_workers,
        )
    else:
        artifacts = {}
        for k, (i, j) in enumerate(tqdm(pending, desc="Family TN pairs", unit="pair"), start=1):
            step_i, step_j = args.steps[i], args.steps[j]
            if step_i not in artifacts:
                artifacts[step_i] = build_step_artifacts(load_component(run_dir, step_i, dtype, device))
            if step_j not in artifacts:
                artifacts[step_j] = build_step_artifacts(load_component(run_dir, step_j, dtype, device))
            row = compute_pair(
                artifacts[step_i],
                artifacts[step_j],
                families=args.families,
                use_orbit_master=True,
                bridges_to_f32=args.bridges_to_f32,
                is_self=(step_i == step_j),
            ).numpy()
            _fill_pair(values, i, j, row)
            if args.save_every > 0 and k % args.save_every == 0:
                save_data(data_path, args.steps, values, local_normalize(values))

    sims = local_normalize(values)
    save_data(data_path, args.steps, values, sims)
    if not args.skip_plots:
        plot_family_heatmaps(output_dir, args.steps, sims)
    write_summary_csv(output_dir, sims, args.steps)


if __name__ == "__main__":
    main()
