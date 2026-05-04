import json
from pathlib import Path

import numpy as np
import torch


def build_schedule(max_steps: int, n_log: int = 100, n_lin: int = 100) -> list[int]:
    log = np.unique(np.round(np.geomspace(1, max_steps, num=n_log)).astype(int)).tolist()
    lin = np.unique(np.round(np.linspace(0, max_steps, num=n_lin)).astype(int)).tolist()
    steps = sorted(set(log) | set(lin) | {0, max_steps})
    return [x for x in steps if 0 <= x <= max_steps]


def save_checkpoint(state: dict, step: int, ckpt_dir: Path):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, ckpt_dir / f"step_{step}.pt")


def write_manifest(steps: list[int], path: Path):
    path.write_text(json.dumps({"steps": steps}, indent=2), encoding="utf-8")
