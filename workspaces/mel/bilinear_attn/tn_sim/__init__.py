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

from .similarity import (
    compute_tn_similarity,
    cosine_similarity,
    inner_product,
    self_similarity,
)
from .similarity_batch import (
    compute_tn_similarity_batch,
    cosine_similarity_batch,
)
from .mc_similarity import mc_similarity, random_sim

__all__ = [
    "compute_tn_similarity",
    "cosine_similarity",
    "inner_product",
    "self_similarity",
    "compute_tn_similarity_batch",
    "cosine_similarity_batch",
    "mc_similarity",
    "random_sim",
]
