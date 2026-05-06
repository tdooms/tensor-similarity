#!/usr/bin/env python3
"""Compute exact TN similarity heatmaps between layer-1 heads.

This is the cheap exact-TN analogue of ``empirical_l1_head_sims.py``. It only
measures the layer-1 active head contribution after the layer-2 residual branch
and unembed:

    embed -> layer1_head_h -> layer2_residual -> unembed

It deliberately avoids constructing the full 34 x 34 path-pair matrix.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from quimb.tensor import Tensor
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[5]
for _path in (str(ROOT), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from experiments.path_decomp.path_pair_tn_heatmaps import (  # noqa: E402
    cache_stats,
    choose_device,
    configure_runtime_cache,
    load_component_with_n_ctx,
    print_cache_status,
    select_steps,
)
from src.components.base import Term  # noqa: E402
from src.components.similarity import State, _initial_state, _moment, _step  # noqa: E402


warnings.filterwarnings("ignore", message="Trial error: No module named 'kahypar'.*")


def layer1_head_term(layer, head_idx: int, n_ctx: int, **like) -> Term:
    """Return the layer active term with the shared head index fixed."""
    terms = layer.terms(n_ctx, **like)
    if len(terms) != 2:
        raise ValueError(f"Expected residual/active terms, got {len(terms)}")
    active = terms[1]
    if not (0 <= head_idx < layer.n_head):
        raise ValueError(f"head_idx={head_idx} outside [0, {layer.n_head})")

    selector = torch.zeros(layer.n_head, **like)
    selector[head_idx] = 1.0
    tn = active.tn.copy()
    tn &= Tensor(selector, inds=("n",), tags=("HEAD_SELECT",))
    return Term(tn, active.legs, active.symmetries)


@torch.no_grad()
def l1_head_inner_products_component(model_a, model_b, *, show_entries: bool = False) -> np.ndarray:
    """Return an (n_head_a, n_head_b) matrix of exact TN similarities."""
    if model_a.n_layers < 2 or model_b.n_layers < 2:
        raise ValueError("Expected at least two layers.")
    if model_a.n_head != model_b.n_head:
        raise ValueError(f"Head counts differ: {model_a.n_head} vs {model_b.n_head}")

    state = _initial_state(model_a)
    n_ctx = state.s_aa.shape[0]
    like = dict(device=state.s_aa.device, dtype=state.s_aa.dtype)

    state = _step(
        state,
        model_a.embed.terms(n_ctx, **like),
        model_b.embed.terms(n_ctx, **like),
    )

    layer2_resid_a = model_a.layers[1].terms(n_ctx, **like)[0]
    layer2_resid_b = model_b.layers[1].terms(n_ctx, **like)[0]
    unembed_a = model_a.unembed.terms(n_ctx, **like)[0]
    unembed_b = model_b.unembed.terms(n_ctx, **like)[0]

    matrix = np.empty((model_a.n_head, model_b.n_head), dtype=np.float64)
    head_terms_a = [layer1_head_term(model_a.layers[0], h, n_ctx, **like) for h in range(model_a.n_head)]
    head_terms_b = [layer1_head_term(model_b.layers[0], h, n_ctx, **like) for h in range(model_b.n_head)]

    for ha, term_a in enumerate(head_terms_a):
        for hb, term_b in enumerate(head_terms_b):
            start = time.perf_counter()
            s_l1 = _moment(term_a, term_b, 0, 1, state)
            s_l2 = _moment(layer2_resid_a, layer2_resid_b, 0, 1, State(s_l1, s_l1, s_l1))
            s_out = _moment(unembed_a, unembed_b, 0, 1, State(s_l2, s_l2, s_l2))
            matrix[ha, hb] = torch.einsum("ijij->", s_out[:, 1:, :, 1:]).item()
            if show_entries:
                elapsed = time.perf_counter() - start
                print(f"head {ha} vs {hb}: value={matrix[ha, hb]:.12e} time_sec={elapsed:.3f}", flush=True)
    return matrix


def pair_indices(n: int, window: int | None):
    for i in range(n):
        for j in range(i, n):
            if window is None or (j - i) <= window:
                yield i, j


def load_existing(path: Path, steps: list[int], n_head: int) -> np.ndarray:
    values = np.full((len(steps), len(steps), n_head, n_head), np.nan, dtype=np.float64)
    if not path.exists():
        return values
    old = np.load(path, allow_pickle=True)
    old_steps = [int(x) for x in old["steps"]]
    old_values = old["l1_head_tn_values"]
    old_index = {s: i for i, s in enumerate(old_steps)}
    h = min(n_head, old_values.shape[-1])
    for i, step_i in enumerate(steps):
        for j, step_j in enumerate(steps):
            if step_i in old_index and step_j in old_index:
                values[i, j, :h, :h] = old_values[old_index[step_i], old_index[step_j], :h, :h]
    return values


def normalize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, _, h, _ = values.shape
    norms = np.full((n, h), np.nan, dtype=np.float64)
    for i in range(n):
        norms[i] = np.diag(values[i, i])

    sims = np.full_like(values, np.nan)
    for i in range(n):
        for j in range(n):
            denom = np.sqrt(np.outer(norms[i], norms[j]))
            with np.errstate(invalid="ignore", divide="ignore"):
                sim = values[i, j] / denom
            sims[i, j] = np.where(np.isfinite(sim) & np.isfinite(denom) & (denom > 0), sim, np.nan)
    return sims, norms


def save_data(path: Path, steps: list[int], values: np.ndarray) -> None:
    sims, norms = normalize(values)
    labels = np.array([f"head{h}" for h in range(values.shape[-1])], dtype=object)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        steps=np.array(steps, dtype=np.int64),
        head_labels=labels,
        l1_head_tn_values=values,
        l1_head_tn_sims=sims,
        l1_head_tn_norms=norms,
    )


def fill_pair(values: np.ndarray, i: int, j: int, matrix: np.ndarray) -> None:
    values[i, j] = matrix
    if i != j:
        values[j, i] = matrix.T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=None)
    parser.add_argument("--step_interval", type=int, default=500)
    parser.add_argument("--no_step_interval", action="store_true")
    parser.add_argument("--linear_checkpoints", type=int, default=0)
    parser.add_argument("--log_checkpoints", type=int, default=0)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--cache_dir", default=".cache/ctg-l1-heads")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--n_ctx", type=int, default=None)
    parser.add_argument("--show_entries", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    configure_runtime_cache(cache_dir)
    print_cache_status(cache_dir)

    device = choose_device(args.device)
    dtype = torch.float64
    step_interval = None if args.no_step_interval else args.step_interval
    steps = select_steps(run_dir, args.steps, step_interval, args.linear_checkpoints, args.log_checkpoints)

    first_component = load_component_with_n_ctx(run_dir, steps[0], args.n_ctx, dtype, device)
    n_head = first_component.n_head
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "path_decomp_trajectory" / "l1_head_tn"
    data_path = output_dir / "l1_head_tn_sims.npz"
    values = load_existing(data_path, steps, n_head)
    components: dict[int, object] = {steps[0]: first_component}

    pending_pairs = [
        (i, j)
        for i, j in pair_indices(len(steps), args.window)
        if np.any(np.isnan(values[i, j]))
    ]
    print(f"device={device}", flush=True)
    print(f"cache_dir={cache_dir}", flush=True)
    print(f"steps={steps}", flush=True)
    print(f"window={args.window}", flush=True)
    print(f"n_ctx={args.n_ctx}", flush=True)
    print(f"n_head={n_head}", flush=True)
    print(f"output={data_path}", flush=True)
    print(f"pending_pairs={len(pending_pairs)}", flush=True)

    if args.dry_run:
        print("dry_run=true", flush=True)
        return

    with tqdm(total=len(pending_pairs), desc="Layer-1 head TN", unit="pair") as pbar:
        for k, (i, j) in enumerate(pending_pairs, start=1):
            step_i, step_j = steps[i], steps[j]
            if step_i not in components:
                components[step_i] = load_component_with_n_ctx(run_dir, step_i, args.n_ctx, dtype, device)
            if step_j not in components:
                components[step_j] = load_component_with_n_ctx(run_dir, step_j, args.n_ctx, dtype, device)

            tqdm.write(f"computing step {step_i} vs {step_j}")
            start = time.perf_counter()
            matrix = l1_head_inner_products_component(
                components[step_i],
                components[step_j],
                show_entries=args.show_entries,
            )
            elapsed = time.perf_counter() - start
            fill_pair(values, i, j, matrix)
            if args.save_every > 0 and k % args.save_every == 0:
                save_data(data_path, steps, values)
            tqdm.write(f"step {step_i} vs {step_j}: l1-head matrix time={elapsed:.2f}s")
            pbar.update(1)

    save_data(data_path, steps, values)
    n_files, n_bytes = cache_stats(cache_dir)
    print(f"Wrote data: {data_path}", flush=True)
    print(f"cache_files_after={n_files} cache_bytes_after={n_bytes}", flush=True)


if __name__ == "__main__":
    main()
