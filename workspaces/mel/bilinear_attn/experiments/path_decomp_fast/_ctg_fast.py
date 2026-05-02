from __future__ import annotations

import os
from pathlib import Path

import cotengra as ctg

from src.components import utils as _comp_utils


def _resolve_cache_dir(cache_dir: str | Path | None = None) -> Path:
    if cache_dir is None:
        cache_dir = os.environ.get("TENSOR_MARS_CTG_CACHE_DIR", ".cache/ctg-paths-fast")
    path = Path(cache_dir)
    if not path.is_absolute():
        path = Path.cwd() / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_fast_opt(cache_dir: Path) -> ctg.ReusableHyperOptimizer:
    return ctg.ReusableHyperOptimizer(
        directory=str(cache_dir),
        minimize="flops",
        methods=("greedy",),
        max_repeats=4,
        parallel=False,
        progbar=False,
    )


def install_fast_optimizer(cache_dir: str | Path | None = None) -> Path:
    resolved = _resolve_cache_dir(cache_dir)
    os.environ["TENSOR_MARS_CTG_CACHE_DIR"] = str(resolved)
    _comp_utils._CACHE_DIR = resolved
    _comp_utils._OPT = _make_fast_opt(resolved)
    _comp_utils._EXPRS.clear()
    return resolved
