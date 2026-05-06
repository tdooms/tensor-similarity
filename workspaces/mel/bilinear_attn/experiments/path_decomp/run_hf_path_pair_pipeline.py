#!/usr/bin/env python3
"""Download a HF run repo, benchmark TN, then store path-pair heatmaps."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_RUNS_DIR = ROOT / "experiments" / "induction_heads" / "runs"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "tensor-mars" / "ctg-paths"


def run_command(cmd: list[str], *, dry_run: bool) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def resolve_run_dir(download_dir: Path) -> Path:
    if (download_dir / "config.yaml").exists() and (download_dir / "checkpoints").exists():
        return download_dir

    candidates = [
        p
        for p in download_dir.rglob("config.yaml")
        if (p.parent / "checkpoints").exists()
    ]
    if len(candidates) == 1:
        return candidates[0].parent
    if not candidates:
        raise FileNotFoundError(
            f"No run directory with config.yaml and checkpoints/ found under {download_dir}"
        )
    raise RuntimeError(
        "Multiple run directories found; pass --local_dir pointing at the desired one: "
        + ", ".join(str(p.parent) for p in candidates)
    )


def checkpoint_steps(run_dir: Path) -> list[int]:
    steps = []
    for path in (run_dir / "checkpoints").glob("step_*.pt"):
        try:
            steps.append(int(path.stem.removeprefix("step_")))
        except ValueError:
            pass
    if not steps:
        raise FileNotFoundError(f"No step_*.pt checkpoints found in {run_dir / 'checkpoints'}")
    return sorted(set(steps))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_id", default="melephant/small-big-experiment-runs")
    parser.add_argument("--repo_type", default="dataset", choices=("dataset", "model"))
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--skip_benchmark", action="store_true")
    parser.add_argument("--skip_heatmap", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without downloading or computing.")
    parser.add_argument("--dry_run_heatmap", action="store_true", help="Run heatmap driver with --dry_run.")
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--cache_dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--n_ctx", type=int, default=None)
    parser.add_argument("--benchmark_step_a", type=int, default=None)
    parser.add_argument("--benchmark_step_b", type=int, default=None)
    parser.add_argument("--linear_checkpoints", type=int, default=20)
    parser.add_argument("--log_checkpoints", type=int, default=20)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    if args.local_dir is not None:
        download_dir = Path(args.local_dir)
    else:
        download_dir = DEFAULT_RUNS_DIR / args.repo_id.split("/")[-1]

    token = os.environ.get("HF_TOKEN")
    if token:
        print("HF_TOKEN found in environment.", flush=True)
    else:
        print("HF_TOKEN not set; download will only work for public repos.", flush=True)

    if not args.skip_download:
        print(f"Downloading {args.repo_id} to {download_dir}", flush=True)
        if not args.dry_run:
            snapshot_download(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                revision=args.revision,
                local_dir=str(download_dir),
                token=token,
                local_dir_use_symlinks=False,
            )
    else:
        print("Skipping download.", flush=True)

    run_dir = resolve_run_dir(download_dir)
    steps = checkpoint_steps(run_dir)
    final_step = steps[-1]
    benchmark_step_a = args.benchmark_step_a if args.benchmark_step_a is not None else steps[0]
    benchmark_step_b = args.benchmark_step_b if args.benchmark_step_b is not None else final_step

    print(f"run_dir={run_dir}", flush=True)
    print(f"available_steps={steps}", flush=True)
    print(f"benchmark_steps={benchmark_step_a},{benchmark_step_b}", flush=True)

    benchmark_script = ROOT / "experiments" / "path_decomp" / "path_pair_tn_benchmark.py"
    heatmap_script = ROOT / "experiments" / "path_decomp" / "path_pair_tn_heatmaps.py"

    if not args.skip_benchmark:
        cmd = [
            sys.executable,
            str(benchmark_script),
            "--run_dir_a",
            str(run_dir),
            "--step_a",
            str(benchmark_step_a),
            "--step_b",
            str(benchmark_step_b),
            "--device",
            args.device,
            "--cache_dir",
            args.cache_dir,
        ]
        if args.n_ctx is not None:
            cmd.extend(["--n_ctx", str(args.n_ctx)])
        run_command(cmd, dry_run=args.dry_run)

    if not args.skip_heatmap:
        cmd = [
            sys.executable,
            str(heatmap_script),
            "--run_dir",
            str(run_dir),
            "--linear_checkpoints",
            str(args.linear_checkpoints),
            "--log_checkpoints",
            str(args.log_checkpoints),
            "--no_step_interval",
            "--device",
            args.device,
            "--cache_dir",
            args.cache_dir,
        ]
        if args.window is not None:
            cmd.extend(["--window", str(args.window)])
        if args.n_ctx is not None:
            cmd.extend(["--n_ctx", str(args.n_ctx)])
        if args.output_dir is not None:
            cmd.extend(["--output_dir", args.output_dir])
        if args.dry_run_heatmap:
            cmd.append("--dry_run")
        run_command(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
