"""Shared test configuration for bilinear_attn.

Adds the tensor-mars repo root to ``sys.path`` at collection time so tests
can import from both ``src.components.*`` (tensor-mars main) and the
workspace's editable packages (``models.*``, ``train.*``, ``tn_sim.*``).

Production code no longer needs the sys.path bridge at import time –
``models.components.__init__`` performs it lazily when actually imported –
but pytest collects every ``test_*.py`` eagerly, and some don't touch
``models.components`` first. Putting the bridge here guarantees it runs
before any test file is imported.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if (_REPO_ROOT / "src" / "components").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
