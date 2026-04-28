"""One-shot cold script: populates `~/.cache/tensor-mars/ctg-paths/paths.pkl`.

Run once after the model architecture changes. `mel_run.py` assumes the path
cache is already warm and will raise `uncompiled topology` otherwise.

Run: `uv run python workspaces/thomas/mel_precompile.py`
"""
import time

import torch

from mel_similarity import DEVICE, DTYPE, load_models
from src.components.similarity import precompile


models = load_models(limit=2)  # precompile only uses first two
print(f'Loaded {len(models)} models on {DEVICE} ({DTYPE})', flush=True)

t0 = time.perf_counter()
precompile(*[m for _, m in models])
print(f'precompile: {time.perf_counter() - t0:.1f}s', flush=True)
if torch.cuda.is_available():
    print(f'  peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB', flush=True)
