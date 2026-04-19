"""Component-compatible layers for TN similarity computation.

This module provides Component-compatible wrappers around the mel workspace
model layers, enabling use of the main codebase's exact TN similarity algorithm.

Key classes:
- EmbeddingComponent: Token embedding as a Linear component
- BilinearAttentionComponent: Bilinear attention as a Component
- AttentionLMComponent: Full model implementing Model.components() interface
"""

# --- bridge to tensor-mars main codebase (see tn_sim/__init__.py) ------------
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[5]
if (_REPO_ROOT / "src" / "components").is_dir() and str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
# -----------------------------------------------------------------------------

from .embedding import EmbeddingComponent, UnembeddingComponent
from .attention import BilinearAttentionComponent, QuadraticAttentionComponent
from .model import AttentionLMComponent

__all__ = [
    "EmbeddingComponent",
    "UnembeddingComponent",
    "BilinearAttentionComponent",
    "QuadraticAttentionComponent",
    "AttentionLMComponent",
]
