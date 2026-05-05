import math
from pathlib import Path

import torch


class NonFiniteError(RuntimeError):
    pass


def _is_finite_scalar(x) -> bool:
    if isinstance(x, torch.Tensor):
        return torch.isfinite(x).all().item()
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return True


def _record(run_dir: Path, step: int, tag: str, bad: list[str]):
    errs = run_dir / "errors"
    errs.mkdir(parents=True, exist_ok=True)
    (errs / f"nan_{tag}_step{step}.txt").write_text(
        f"step={step}\ntag={tag}\nbad={bad}\n",
        encoding="utf-8",
    )


def check_finite_tensors(tag: str, named_tensors, step: int, run_dir: Path, *, enabled: bool = True):
    if not enabled:
        return
    bad = [n for n, t in named_tensors if t is not None and not torch.isfinite(t).all()]
    if bad:
        _record(run_dir, step, tag, bad)
        suffix = "..." if len(bad) > 8 else ""
        raise NonFiniteError(f"[step {step}] non-finite in {tag}: {bad[:8]}{suffix}")


def check_finite_metrics(tag: str, metrics: dict, step: int, run_dir: Path | None, *, enabled: bool = True):
    if not enabled:
        return
    bad = [k for k, v in metrics.items() if not _is_finite_scalar(v)]
    if bad:
        if run_dir is not None:
            _record(run_dir, step, tag, bad)
        raise NonFiniteError(f"[step {step}] non-finite metrics in {tag}: {bad}")
