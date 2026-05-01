#!/usr/bin/env python3
"""Checkpoint heatmaps for per-family diagonal no-sym TN similarities.

For each checkpoint pair (i, j), compute only same-family path similarities:
family f in checkpoint i against family f in checkpoint j. Each family gets its own locally
normalized checkpoint-by-checkpoint heatmap:

    M_ij[f, f] / sqrt(M_ii[f, f] * M_jj[f, f])

The 34 path families are deduped to the 22 raw-TN-verified canonical families.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.path_decomp.no_sym_tn_similarity import configure_cache, load_component  # noqa: E402


DEFAULT_STEPS = list(range(0, 15001, 500))

LAYER2_GROUPS = [
    ("00000", ["00000"]),
    ("00001", ["00001", "00100"]),
    ("00010", ["00010", "01000"]),
    ("00011", ["00011", "01100"]),
    ("00101", ["00101"]),
    ("00110", ["00110", "01001"]),
    ("00111", ["00111", "01101"]),
    ("01010", ["01010"]),
    ("01011", ["01011", "01110"]),
    ("01111", ["01111"]),
    ("10000", ["10000"]),
    ("10001", ["10001", "10100"]),
    ("10010", ["10010", "11000"]),
    ("10011", ["10011", "11100"]),
    ("10101", ["10101"]),
    ("10110", ["10110", "11001"]),
    ("10111", ["10111", "11101"]),
    ("11010", ["11010"]),
    ("11011", ["11011", "11110"]),
    ("11111", ["11111"]),
]
CANONICAL_LABELS = ["direct", "layer1", *[label for label, _members in LAYER2_GROUPS]]


def family_from_label(label: str):
    if label in ("direct", "layer1"):
        return label
    return ("layer2", int(label, 2))


CANONICAL_FAMILIES = [family_from_label(label) for label in CANONICAL_LABELS]


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return device


def import_family_helpers():
    from experiments.path_decomp.forward import enumerate_families
    from experiments.path_decomp.moments import (
        _family_to_tt_and_src,
        _master_moment,
        _stack_s_split,
    )
    from src.components.base import Term
    from src.components.similarity import State, _initial_state, _moment, _step

    return enumerate_families, _family_to_tt_and_src, _master_moment, _stack_s_split, Term, State, _initial_state, _moment, _step


@torch.no_grad()
def family_diagonal_inner_products_component(model_a, model_b) -> dict:
    """Compute same-family path inner products for AttentionLMComponent."""
    (
        _enumerate_families,
        _family_to_tt_and_src,
        _master_moment,
        _stack_s_split,
        Term,
        State,
        _initial_state,
        _moment,
        _step,
    ) = import_family_helpers()

    assert model_a.n_layers == 2 and model_b.n_layers == 2
    state = _initial_state(model_a)
    n_ctx = state.s_aa.shape[0]
    like = dict(device=state.s_aa.device, dtype=state.s_aa.dtype)

    state = _step(
        state,
        model_a.embed.terms(n_ctx, **like),
        model_b.embed.terms(n_ctx, **like),
    )
    d_padded = state.s_aa.shape[1]

    no_sym = lambda terms: [Term(t.tn, t.legs, symmetries=()) for t in terms]
    ta1 = no_sym(model_a.layers[0].terms(n_ctx, **like))
    tb1 = no_sym(model_b.layers[0].terms(n_ctx, **like))
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
    masters = {}
    needed_master_keys = {
        (_family_to_tt_and_src(fam)[0], _family_to_tt_and_src(fam)[0])
        for fam in CANONICAL_FAMILIES
    }
    for tta, ttb in sorted(needed_master_keys):
        masters[(tta, ttb)] = _master_moment(ta2[tta], tb2[ttb], 0, 1, S)

    th_a = model_a.unembed.terms(n_ctx, **like)
    th_b = model_b.unembed.terms(n_ctx, **like)
    values = {}
    for fa in CANONICAL_FAMILIES:
        tta, src_a = _family_to_tt_and_src(fa)
        ttb, src_b = _family_to_tt_and_src(fa)
        master = masters[(tta, ttb)]
        idx = (slice(None),) * 4 + tuple(src_a) + tuple(src_b)
        s_ab_l2 = master[idx]
        proxy = State(s_ab_l2, s_ab_l2, s_ab_l2)
        s_ab_out = _moment(th_a[0], th_b[0], 0, 1, proxy)
        values[fa] = torch.einsum("ijij->", s_ab_out[:, 1:, :, 1:]).item()
    return values


def pair_indices(n: int, window: int | None):
    for i in range(n):
        for j in range(i, n):
            if window is None or (j - i) <= window:
                yield i, j


def load_existing(path: Path, n_fam: int, steps: list[int]):
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


def save_data(path: Path, steps: list[int], values: np.ndarray, sims: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        steps=np.array(steps),
        family_labels=np.array(CANONICAL_LABELS, dtype=object),
        family_diag_values=values,
        family_local_sims=sims,
    )


def local_normalize(values: np.ndarray) -> np.ndarray:
    sims = np.full_like(values, np.nan)
    n_fam, n, _ = values.shape
    for f in range(n_fam):
        diag = np.diag(values[f])
        for i in range(n):
            for j in range(n):
                if np.isnan(values[f, i, j]) or np.isnan(diag[i]) or np.isnan(diag[j]):
                    continue
                denom = np.sqrt(diag[i] * diag[j])
                if denom > 0 and np.isfinite(denom):
                    sims[f, i, j] = values[f, i, j] / denom
    return sims


def plot_family_heatmaps(output_dir: Path, steps: list[int], sims: np.ndarray) -> None:
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for f, label in enumerate(CANONICAL_LABELS):
        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
        im = ax.imshow(np.ma.masked_invalid(sims[f]), vmin=-1.0, vmax=1.0, cmap="coolwarm")
        ax.set_title(f"Family {label}: locally normalized diagonal TN sim")
        ax.set_xlabel("checkpoint step")
        ax.set_ylabel("checkpoint step")
        ax.set_xticks(range(len(steps)))
        ax.set_yticks(range(len(steps)))
        ax.set_xticklabels(steps, rotation=90, fontsize=7)
        ax.set_yticklabels(steps, fontsize=7)
        cbar = fig.colorbar(im, ax=ax, shrink=0.82)
        cbar.set_label("local cosine")
        fig.savefig(image_dir / f"family_{label}_local_heatmap.png", dpi=220)
        plt.close(fig)


def write_summary_csv(output_dir: Path, sims: np.ndarray, steps: list[int]) -> None:
    rows = []
    final_idx = len(steps) - 1
    for f, label in enumerate(CANONICAL_LABELS):
        row = {"family": label}
        for offset in (1, 2, 5):
            vals = []
            for i in range(len(steps) - offset):
                v = sims[f, i, i + offset]
                if not np.isnan(v):
                    vals.append(float(v))
            row[f"mean_window_offset_{offset}"] = sum(vals) / len(vals) if vals else float("nan")
        row["final_self"] = sims[f, final_idx, final_idx]
        rows.append(row)
    with (output_dir / "family_diag_heatmap_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", default="experiments/induction_heads/runs/small-big-experiment-runs")
    parser.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--cache_dir", default=".cache/ctg-paths")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    configure_cache(cache_dir)
    device = choose_device(args.device)
    dtype = torch.float64
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else run_dir / "path_decomp_trajectory" / "family_diag_heatmaps"
    )
    data_path = output_dir / "family_diag_tn_sims.npz"
    values = load_existing(data_path, len(CANONICAL_FAMILIES), args.steps)
    components: dict[int, object] = {}

    print(f"device={device}", flush=True)
    print(f"cache_dir={cache_dir}", flush=True)
    print(f"steps={args.steps}", flush=True)
    print(f"window={args.window}", flush=True)
    for i, j in pair_indices(len(args.steps), args.window):
        if not np.isnan(values[0, i, j]):
            continue
        step_i = args.steps[i]
        step_j = args.steps[j]
        if step_i not in components:
            components[step_i] = load_component(run_dir, step_i, dtype, device)
        if step_j not in components:
            components[step_j] = load_component(run_dir, step_j, dtype, device)
        print(f"computing step {step_i} vs {step_j}", flush=True)
        start = time.perf_counter()
        diag_values = family_diagonal_inner_products_component(components[step_i], components[step_j])
        elapsed = time.perf_counter() - start
        for f, fam in enumerate(CANONICAL_FAMILIES):
            values[f, i, j] = diag_values[fam]
            values[f, j, i] = diag_values[fam]
        sims = local_normalize(values)
        save_data(data_path, args.steps, values, sims)
        print(f"step {step_i} vs {step_j}: family diagonal time={elapsed:.2f}s", flush=True)

    sims = local_normalize(values)
    save_data(data_path, args.steps, values, sims)
    plot_family_heatmaps(output_dir, args.steps, sims)
    write_summary_csv(output_dir, sims, args.steps)
    print(f"Wrote data: {data_path}", flush=True)
    print(f"Wrote images: {output_dir / 'images'}", flush=True)


if __name__ == "__main__":
    main()
