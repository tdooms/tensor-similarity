"""Fast family-diagonal TN tooling.

This package is kept side-by-side with ``experiments/path_decomp`` so outputs
can be diffed against the reference implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[5]
for _path in (str(ROOT), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def install_fast_ctg(cache_dir: str | Path | None = None) -> None:
    from ._ctg_fast import install_fast_optimizer

    install_fast_optimizer(cache_dir=cache_dir)
