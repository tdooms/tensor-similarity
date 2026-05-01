#!/usr/bin/env python3
"""Whole-model TN similarity with term symmetries stripped.

This is the no-symmetry variant used as the trusted reference in the
path-decomposition pytest. For trained `AttentionLM` checkpoints, norms are
explicitly ignored by converting to `AttentionLMComponent(ignore_norms=True)`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def configure_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TENSOR_MARS_CTG_CACHE_DIR", str(cache_dir))


def import_tn_helpers():
    from models import AttentionLM
    from models.components.model import AttentionLMComponent
    from src.components.base import Term
    from src.components.similarity import _initial_state, _step

    return AttentionLM, AttentionLMComponent, Term, _initial_state, _step


def strip_symmetries(terms, Term):
    return [Term(t.tn, t.legs, symmetries=()) for t in terms]


def trace_nonconstant(s: torch.Tensor) -> float:
    return torch.einsum("ijij->", s[:, 1:, :, 1:]).item()


def load_component(run_dir: Path, step: int, dtype: torch.dtype, device: torch.device):
    AttentionLM, AttentionLMComponent, _Term, _initial_state, _step = import_tn_helpers()
    with (run_dir / "config.yaml").open() as f:
        cfg = yaml.safe_load(f)
    model = AttentionLM.from_config(cfg)
    checkpoint = torch.load(run_dir / "checkpoints" / f"step_{step}.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    component = AttentionLMComponent.from_trained_model(model, ignore_norms=True)
    return component.to(device=device, dtype=dtype)


def load_config(run_dir: Path) -> dict:
    with (run_dir / "config.yaml").open() as f:
        return yaml.safe_load(f)


@torch.no_grad()
def similarity_no_sym(model_a, model_b):
    _AttentionLM, _AttentionLMComponent, Term, _initial_state, _step = import_tn_helpers()
    state = _initial_state(model_a)
    for comp_a, comp_b in zip(model_a.components(), model_b.components()):
        n = state.s_aa.shape[0]
        like = dict(device=state.s_aa.device, dtype=state.s_aa.dtype)
        terms_a = strip_symmetries(comp_a.terms(n, **like), Term)
        terms_b = strip_symmetries(comp_b.terms(n, **like), Term)
        state = _step(state, terms_a, terms_b)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", default="experiments/induction_heads/runs/small-big-experiment-runs")
    parser.add_argument("--step_a", type=int, required=True)
    parser.add_argument("--step_b", type=int, required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--cache_dir", default=".cache/ctg-paths")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    configure_cache(cache_dir)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    dtype = torch.float64
    print(f"cache_dir={cache_dir}")
    print(f"device={device}")
    print(f"loading step {args.step_a} and step {args.step_b}")
    model_a = load_component(run_dir, args.step_a, dtype, device)
    model_b = load_component(run_dir, args.step_b, dtype, device)

    start = time.perf_counter()
    state = similarity_no_sym(model_a, model_b)
    elapsed = time.perf_counter() - start

    aa = trace_nonconstant(state.s_aa)
    ab = trace_nonconstant(state.s_ab)
    bb = trace_nonconstant(state.s_bb)
    cos = ab / ((aa * bb) ** 0.5)
    print(f"time_sec={elapsed:.3f}")
    print(f"tr_aa={aa:.12e}")
    print(f"tr_ab={ab:.12e}")
    print(f"tr_bb={bb:.12e}")
    print(f"cos={cos:.12g}")


if __name__ == "__main__":
    main()
