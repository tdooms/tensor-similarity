"""TN similarity module for bilinear attention models.

This module provides tensor network similarity computation using the main
codebase's exact algorithm (src/components/similarity.py).

Main exports:
- cosine_similarity: Compute cosine similarity between two models
- compute_tn_similarity: Get full State object with second moments
- inner_product: Compute unnormalized inner product
- self_similarity: Convenience function for self-similarity

Monte Carlo baseline:
- mc_similarity: MC similarity using Gaussian residual-stream samples

Limitations:
- Only supports models with norm_type='none' and norm_places=[]
- Only supports 'bilinear' and 'quadratic' attention types
- Does not support use_rmsnorm_qk=True
"""

# --- bridge to tensor-mars main codebase -------------------------------------
# The bilinear_attn workspace is pip-installed as an editable package, but the
# tensor-mars repo root (which owns `src.components.*`) is not. We prepend it
# to sys.path here so `from src.components.similarity import ...` resolves for
# any consumer of tn_sim (tests, scripts, notebooks).
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[4]  # tn_sim/__init__.py -> bilinear_attn -> mel -> workspaces -> tensor-mars
if (_REPO_ROOT / "src" / "components").is_dir() and str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
# -----------------------------------------------------------------------------

from .similarity import (
    compute_tn_similarity,
    cosine_similarity,
    inner_product,
    self_similarity,
)
from .mc_similarity import mc_similarity, random_sim

__all__ = [
    "compute_tn_similarity",
    "cosine_similarity",
    "inner_product",
    "self_similarity",
    "mc_similarity",
    "random_sim",
]
