#!/usr/bin/env python3
"""Store full path-pair TN matrices for checkpoint trajectories.

For each checkpoint pair (i, j), this stores the full 34 x 34 matrix

    M[i, j, rho, sigma] = <F_i,rho, F_j,sigma>

including self checkpoint pairs (i, i). Those self matrices are needed for
local path norms and for later regrouping into layer/block similarities.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[5]
for _path in (str(ROOT), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from experiments.path_decomp.forward import enumerate_families  # noqa: E402
from experiments.path_decomp.moments import (  # noqa: E402
    _family_to_tt_and_src,
    _master_moment,
    _stack_s_split,
)
from experiments.path_decomp.no_sym_tn_similarity import (  # noqa: E402
    configure_cache,
    load_component,
)
from models import AttentionLM  # noqa: E402
from models.components.model import AttentionLMComponent  # noqa: E402
from src.components.base import Term  # noqa: E402
from src.components.similarity import State, _initial_state, _moment, _step  # noqa: E402


DEFAULT_STEPS = list(range(0, 15001, 500))
FAMILIES = list(enumerate_families())
PATH_LABELS = [
    "direct" if fam == "direct" else "layer1" if fam == "layer1" else format(fam[1], "05b")
    for fam in FAMILIES
]


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return device


def load_component_with_n_ctx(
    run_dir: Path,
    step: int,
    n_ctx: int | None,
    dtype: torch.dtype,
    device: torch.device,
):
    """Load a trained checkpoint, optionally rebuilding the model at shorter n_ctx."""
    if n_ctx is None:
        return load_component(run_dir, step, dtype, device)

    with (run_dir / "config.yaml").open() as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg["model"]["n_ctx"] = n_ctx

    model = AttentionLM.from_config(cfg)
    checkpoint = torch.load(run_dir / "checkpoints" / f"step_{step}.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    component = AttentionLMComponent.from_trained_model(model, ignore_norms=True)
    return component.to(device=device, dtype=dtype)


@torch.no_grad()
def path_pair_inner_products_component(model_a, model_b) -> np.ndarray:
    """Compute the full 34 x 34 path-pair matrix for AttentionLMComponent."""
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
    for tta in (0, 1):
        for ttb in (0, 1):
            masters[(tta, ttb)] = _master_moment(ta2[tta], tb2[ttb], 0, 1, S)

    th_a = model_a.unembed.terms(n_ctx, **like)
    th_b = model_b.unembed.terms(n_ctx, **like)
    matrix = np.empty((len(FAMILIES), len(FAMILIES)), dtype=np.float64)
    for ia, fa in enumerate(FAMILIES):
        tta, src_a = _family_to_tt_and_src(fa)
        for ib, fb in enumerate(FAMILIES):
            ttb, src_b = _family_to_tt_and_src(fb)
            master = masters[(tta, ttb)]
            idx = (slice(None),) * 4 + tuple(src_a) + tuple(src_b)
            s_ab_l2 = master[idx]
            proxy = State(s_ab_l2, s_ab_l2, s_ab_l2)
            s_ab_out = _moment(th_a[0], th_b[0], 0, 1, proxy)
            matrix[ia, ib] = torch.einsum("ijij->", s_ab_out[:, 1:, :, 1:]).item()
    return matrix


def pair_indices(n: int, window: int | None):
    for i in range(n):
        for j in range(i, n):
            if window is None or (j - i) <= window:
                yield i, j


def load_existing(path: Path, steps: list[int]) -> np.ndarray:
    n = len(steps)
    values = np.full((n, n, len(FAMILIES), len(FAMILIES)), np.nan, dtype=np.float64)
    if not path.exists():
        return values

    old = np.load(path, allow_pickle=True)
    old_steps = [int(x) for x in old["steps"]]
    old_values = old["path_pair_values"]
    old_index = {s: i for i, s in enumerate(old_steps)}
    for i, step_i in enumerate(steps):
        for j, step_j in enumerate(steps):
            if step_i in old_index and step_j in old_index:
                values[i, j] = old_values[old_index[step_i], old_index[step_j]]
    return values


def local_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize each path pair by path-local self norms from M_ii and M_jj."""
    n = values.shape[0]
    sims = np.full_like(values, np.nan)
    for i in range(n):
        norm_i = np.diag(values[i, i])
        for j in range(n):
            norm_j = np.diag(values[j, j])
            denom = np.sqrt(np.outer(norm_i, norm_j))
            with np.errstate(invalid="ignore", divide="ignore"):
                sim = values[i, j] / denom
            sims[i, j] = np.where(np.isfinite(sim) & np.isfinite(denom) & (denom > 0), sim, np.nan)
    return sims


def save_data(path: Path, steps: list[int], values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        steps=np.array(steps, dtype=np.int64),
        path_labels=np.array(PATH_LABELS, dtype=object),
        path_families=np.array([repr(fam) for fam in FAMILIES], dtype=object),
        path_pair_values=values,
        path_pair_local_sims=local_normalize(values),
    )


def fill_pair(values: np.ndarray, i: int, j: int, matrix: np.ndarray) -> None:
    values[i, j] = matrix
    if i != j:
        values[j, i] = matrix.T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", default="experiments/induction_heads/runs/small-big-experiment-runs")
    parser.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--cache_dir", default=".cache/ctg-paths")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--n_ctx", type=int, default=None, help="Optional checkpoint-load context override.")
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
        else run_dir / "path_decomp_trajectory" / "path_pair_matrices"
    )
    data_path = output_dir / "path_pair_tn_matrices.npz"
    values = load_existing(data_path, args.steps)
    components: dict[int, object] = {}

    print(f"device={device}", flush=True)
    print(f"cache_dir={cache_dir}", flush=True)
    print(f"steps={args.steps}", flush=True)
    print(f"window={args.window}", flush=True)
    print(f"n_ctx={args.n_ctx}", flush=True)
    print(f"output={data_path}", flush=True)

    pending_pairs = [
        (i, j)
        for i, j in pair_indices(len(args.steps), args.window)
        if np.any(np.isnan(values[i, j]))
    ]

    with tqdm(total=len(pending_pairs), desc="Path-pair TN matrices", unit="pair") as pbar:
        for k, (i, j) in enumerate(pending_pairs, start=1):
            step_i = args.steps[i]
            step_j = args.steps[j]
            if step_i not in components:
                components[step_i] = load_component_with_n_ctx(
                    run_dir, step_i, args.n_ctx, dtype, device
                )
            if step_j not in components:
                components[step_j] = load_component_with_n_ctx(
                    run_dir, step_j, args.n_ctx, dtype, device
                )

            tqdm.write(f"computing step {step_i} vs {step_j}")
            start = time.perf_counter()
            matrix = path_pair_inner_products_component(components[step_i], components[step_j])
            elapsed = time.perf_counter() - start
            fill_pair(values, i, j, matrix)

            if args.save_every > 0 and k % args.save_every == 0:
                save_data(data_path, args.steps, values)
            tqdm.write(f"step {step_i} vs {step_j}: path-pair matrix time={elapsed:.2f}s")
            pbar.update(1)

    save_data(data_path, args.steps, values)
    print(f"Wrote data: {data_path}", flush=True)


if __name__ == "__main__":
    main()
