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


DEFAULT_STEP_INTERVAL = 500
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


def cache_stats(cache_dir: Path) -> tuple[int, int]:
    if not cache_dir.exists():
        return 0, 0
    files = [p for p in cache_dir.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def print_cache_status(cache_dir: Path) -> None:
    n_files, n_bytes = cache_stats(cache_dir)
    if n_files:
        print(
            f"cache_found={cache_dir} files={n_files} bytes={n_bytes}",
            flush=True,
        )
    else:
        print(f"cache_found=false cache_dir={cache_dir}", flush=True)


def configure_runtime_cache(cache_dir: Path) -> None:
    """Point both env and already-imported contraction utilities at cache_dir."""
    import cotengra as ctg
    from src.components import utils as comp_utils

    configure_cache(cache_dir)
    comp_utils._CACHE_DIR = cache_dir
    comp_utils._OPT = ctg.ReusableHyperOptimizer(
        directory=str(cache_dir),
        minimize="size",
        methods=("greedy", "kahypar"),
        max_repeats=32,
        parallel=False,
        progbar=False,
    )
    comp_utils._EXPRS.clear()


def checkpoint_steps(run_dir: Path) -> list[int]:
    checkpoint_dir = run_dir / "checkpoints"
    steps = []
    for path in checkpoint_dir.glob("step_*.pt"):
        try:
            steps.append(int(path.stem.removeprefix("step_")))
        except ValueError:
            pass
    if not steps:
        raise FileNotFoundError(f"No step_*.pt checkpoints found in {checkpoint_dir}")
    return sorted(set(steps))


def nearest_available_steps(available: list[int], requested: np.ndarray) -> list[int]:
    """Snap requested numeric steps to the nearest checkpoint files present."""
    available_arr = np.array(available, dtype=np.int64)
    selected = []
    for value in requested:
        idx = int(np.argmin(np.abs(available_arr - value)))
        selected.append(int(available_arr[idx]))
    return selected


def linear_steps(available: list[int], count: int) -> list[int]:
    if count <= 0:
        return []
    requested = np.linspace(available[0], available[-1], num=count)
    return nearest_available_steps(available, requested)


def log_steps(available: list[int], count: int) -> list[int]:
    """Choose log-spaced checkpoint steps, always including 0 when available."""
    if count <= 0:
        return []
    final_step = available[-1]
    if final_step <= 0:
        return [final_step]

    selected = [0] if available[0] == 0 else []
    positive_count = max(count - len(selected), 0)
    if positive_count:
        positive_available = [s for s in available if s > 0]
        min_positive = positive_available[0]
        requested = np.geomspace(min_positive, final_step, num=positive_count)
        selected.extend(nearest_available_steps(available, requested))
    return selected


def select_steps(
    run_dir: Path,
    explicit_steps: list[int] | None,
    step_interval: int | None,
    linear_count: int,
    log_count: int,
) -> list[int]:
    available = checkpoint_steps(run_dir)
    if explicit_steps:
        missing = sorted(set(explicit_steps) - set(available))
        if missing:
            raise FileNotFoundError(f"Requested checkpoint steps not found: {missing}")
        return sorted(dict.fromkeys(explicit_steps))

    selected: list[int] = []
    final_step = available[-1]

    if step_interval is not None:
        if step_interval <= 0:
            raise ValueError("--step_interval must be positive")
        selected.extend(s for s in available if s == 0 or s == final_step or s % step_interval == 0)

    selected.extend(linear_steps(available, linear_count))
    selected.extend(log_steps(available, log_count))

    if not selected:
        selected.extend([available[0], final_step])
    return sorted(dict.fromkeys(selected))


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
    if cfg.get("model", {}).get("norm_type") == "tok_0":
        cfg["model"]["norm_type"] = "tok0"
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
    parser.add_argument("--steps", type=int, nargs="+", default=None)
    parser.add_argument(
        "--step_interval",
        type=int,
        default=DEFAULT_STEP_INTERVAL,
        help=(
            "When --steps is omitted, use checkpoint 0, the final checkpoint, "
            "and checkpoints whose step is a multiple of this value."
        ),
    )
    parser.add_argument(
        "--no_step_interval",
        action="store_true",
        help="Disable interval-based checkpoint selection when using linear/log selection.",
    )
    parser.add_argument(
        "--linear_checkpoints",
        type=int,
        default=0,
        help="Add this many linearly spaced checkpoints, snapped to available checkpoint files.",
    )
    parser.add_argument(
        "--log_checkpoints",
        type=int,
        default=0,
        help="Add this many log-spaced checkpoints, snapped to available checkpoint files.",
    )
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--cache_dir", default=".cache/ctg-paths")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--n_ctx", type=int, default=None, help="Optional checkpoint-load context override.")
    parser.add_argument("--dry_run", action="store_true", help="Print selected checkpoints and pending pair count, then exit.")
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
    steps = select_steps(
        run_dir,
        args.steps,
        step_interval,
        args.linear_checkpoints,
        args.log_checkpoints,
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else run_dir / "path_decomp_trajectory" / "path_pair_matrices"
    )
    data_path = output_dir / "path_pair_tn_matrices.npz"
    values = load_existing(data_path, steps)
    components: dict[int, object] = {}

    print(f"device={device}", flush=True)
    print(f"cache_dir={cache_dir}", flush=True)
    print(f"steps={steps}", flush=True)
    print(f"window={args.window}", flush=True)
    print(f"n_ctx={args.n_ctx}", flush=True)
    print(f"output={data_path}", flush=True)

    pending_pairs = [
        (i, j)
        for i, j in pair_indices(len(steps), args.window)
        if np.any(np.isnan(values[i, j]))
    ]
    print(f"n_checkpoints={len(steps)}", flush=True)
    print(f"pending_pairs={len(pending_pairs)}", flush=True)

    if args.dry_run:
        print("dry_run=true", flush=True)
        return

    with tqdm(total=len(pending_pairs), desc="Path-pair TN matrices", unit="pair") as pbar:
        for k, (i, j) in enumerate(pending_pairs, start=1):
            step_i = steps[i]
            step_j = steps[j]
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
                save_data(data_path, steps, values)
            tqdm.write(f"step {step_i} vs {step_j}: path-pair matrix time={elapsed:.2f}s")
            pbar.update(1)

    save_data(data_path, steps, values)
    print(f"Wrote data: {data_path}", flush=True)


if __name__ == "__main__":
    main()
