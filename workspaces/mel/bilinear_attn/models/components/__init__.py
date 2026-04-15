"""Component-compatible layers for TN similarity computation.

This module provides Component-compatible wrappers around the mel workspace
model layers, enabling use of the main codebase's exact TN similarity algorithm.

Key classes:
- EmbeddingComponent: Token embedding as a Linear component
- BilinearAttentionComponent: Bilinear attention as a Component
- AttentionLMComponent: Full model implementing Model.components() interface
"""

from .embedding import EmbeddingComponent
from .attention import BilinearAttentionComponent
from .model import AttentionLMComponent

__all__ = [
    "EmbeddingComponent",
    "BilinearAttentionComponent",
    "AttentionLMComponent",
]
