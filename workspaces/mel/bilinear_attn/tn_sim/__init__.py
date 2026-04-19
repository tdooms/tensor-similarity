"""TN similarity module for bilinear attention models.

This module provides tensor network similarity computation using the main
codebase's exact algorithm (src/components/similarity.py).

Main exports:
- cosine_similarity: Compute cosine similarity between two models
- compute_tn_similarity: Get full State object with second moments
- inner_product: Compute unnormalized inner product
- self_similarity: Convenience function for self-similarity

Monte Carlo baselines:
- mc_similarity: MC similarity using Gaussian residual-stream samples.
- mc_similarity_gaussian_tokens: TN-matched MC (Gaussian over the padded
  vocab axis, propagated through the full model). This is the MC estimator
  that converges to the TN value; use it to validate TN outputs.
- random_sim: discrete-uniform token-sequence MC (for heatmaps, not a TN
  baseline).

Limitations:
- Only supports models with norm_type='none' and norm_places=[]
- Only supports 'bilinear' and 'quadratic' attention types
- Does not support use_rmsnorm_qk=True
"""

# The tensor-mars repo root (owner of ``src.components.*``) is not on
# sys.path by default. ``models.components`` owns that path bridge; import
# it here purely for the side effect so ``from src.components...`` works
# inside ``.similarity`` below.
import models.components as _models_components  # noqa: F401

from .similarity import (
    compute_tn_similarity,
    cosine_similarity,
    inner_product,
    self_similarity,
)
from .mc_similarity import mc_similarity, mc_similarity_gaussian_tokens, random_sim

__all__ = [
    "compute_tn_similarity",
    "cosine_similarity",
    "inner_product",
    "self_similarity",
    "mc_similarity",
    "mc_similarity_gaussian_tokens",
    "random_sim",
]
