import sys
from pathlib import Path

# Add repo root (tensor-mars/) to path so `src` package is importable
_repo_root = Path(__file__).parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
