#!/usr/bin/env python3
"""Compute whole-model empirical similarity heatmaps from induction train data.

For each checkpoint, this evaluates the model logits on repeated-token train
data and computes empirical raw dot products plus globally normalized cosine
similarities across checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[5]
for _path in (str(ROOT), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from experiments.induction_heads.data import RepeatedTokenDataset  # noqa: E402
from experiments.path_decomp.path_pair_tn_heatmaps import select_steps  # noqa: E402
from models import AttentionLM  # noqa: E402


def checkpoint_path(run_dir: Path, step: int) -> Path:
    exact = run_dir / "checkpoints" / f"step_{step}.pt"
    if exact.exists():
        return exact
    for path in sorted((run_dir / "checkpoints").glob("step_*.pt")):
        try:
            if int(path.stem.removeprefix("step_")) == step:
                return path
        except ValueError:
            pass
    raise FileNotFoundError(f"No checkpoint for step {step} in {run_dir / 'checkpoints'}")


def load_model(run_dir: Path, step: int, device: torch.device) -> tuple[AttentionLM, dict]:
    with (run_dir / "config.yaml").open() as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    if cfg.get("model", {}).get("norm_type") == "tok_0":
        cfg["model"]["norm_type"] = "tok0"

    model = AttentionLM.from_config(cfg)
    checkpoint = torch.load(checkpoint_path(run_dir, step), map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model.to(device), cfg


def build_train_dataset(cfg: dict, n_samples: int, seed: int) -> RepeatedTokenDataset:
    model_cfg = cfg["model"]
    data_cfg = cfg.get("data", {})
    use_bos = data_cfg.get("use_bos", False)
    bos_token_id = data_cfg.get("bos_token_id")
    if use_bos and bos_token_id is None:
        bos_token_id = model_cfg["vocab_size"] - 1
    if not use_bos:
        bos_token_id = None
    return RepeatedTokenDataset(
        vocab_size=model_cfg["vocab_size"],
        n_ctx=model_cfg["n_ctx"],
        n_samples=n_samples,
        seed=seed,
        bos_token_id=bos_token_id,
    )


@torch.no_grad()
def checkpoint_logits(
    model: AttentionLM,
    dataset: RepeatedTokenDataset,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    chunks = []
    for start in tqdm(range(0, len(dataset), batch_size), desc="Empirical batches", leave=False):
        batch = dataset.data[start : start + batch_size].to(device)
        chunks.append(model(batch).detach().cpu().float())
    return torch.cat(chunks, dim=0)


def compute_global_sims(outputs: list[torch.Tensor]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(outputs)
    flat = [out.reshape(-1).double() for out in outputs]
    dims = [x.numel() for x in flat]
    if len(set(dims)) != 1:
        raise ValueError(f"All checkpoint logits must have same flattened size; got {dims}")

    raw = np.full((n, n), np.nan, dtype=np.float64)
    norms = np.full(n, np.nan, dtype=np.float64)
    denom_dim = dims[0]
    for i in range(n):
        norms[i] = torch.dot(flat[i], flat[i]).item() / denom_dim
    for i in range(n):
        for j in range(n):
            raw[i, j] = torch.dot(flat[i], flat[j]).item() / denom_dim

    sims = np.full_like(raw, np.nan)
    for i in range(n):
        for j in range(n):
            denom = norms[i] * norms[j]
            if np.isfinite(raw[i, j]) and np.isfinite(denom) and denom > 0:
                sims[i, j] = raw[i, j] / np.sqrt(denom)
    return raw, sims, norms


def plot_heatmap(matrix: np.ndarray, steps: np.ndarray, output_path: Path, *, title: str, vmin, vmax, cmap: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    im = ax.imshow(np.ma.masked_invalid(matrix), vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("checkpoint step")
    ax.set_xticks(range(len(steps)))
    ax.set_yticks(range(len(steps)))
    ax.set_xticklabels([str(int(s)) for s in steps], rotation=90, fontsize=7)
    ax.set_yticklabels([str(int(s)) for s in steps], fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.82)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_summary_csv(path: Path, values: np.ndarray, steps: np.ndarray) -> None:
    rows = []
    for i, step_i in enumerate(steps):
        for j, step_j in enumerate(steps):
            rows.append(
                {
                    "step_i": int(step_i),
                    "step_j": int(step_j),
                    "value": float(values[i, j]) if np.isfinite(values[i, j]) else float("nan"),
                }
            )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step_i", "step_j", "value"])
        writer.writeheader()
        writer.writerows(rows)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=None)
    parser.add_argument("--step_interval", type=int, default=500)
    parser.add_argument("--no_step_interval", action="store_true")
    parser.add_argument("--linear_checkpoints", type=int, default=0)
    parser.add_argument("--log_checkpoints", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--plot_dir", default=None)
    parser.add_argument("--vmin", type=float, default=-1.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--cmap", default="coolwarm")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = choose_device(args.device)
    step_interval = None if args.no_step_interval else args.step_interval
    steps = select_steps(run_dir, args.steps, step_interval, args.linear_checkpoints, args.log_checkpoints)

    first_model, cfg = load_model(run_dir, steps[0], device)
    dataset = build_train_dataset(cfg, args.n_samples, args.seed)

    outputs = []
    for idx, step in enumerate(tqdm(steps, desc="Empirical checkpoints", unit="ckpt")):
        model = first_model if idx == 0 else load_model(run_dir, step, device)[0]
        outputs.append(checkpoint_logits(model, dataset, args.batch_size, device))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    raw, sims, norms = compute_global_sims(outputs)
    output_path = Path(args.output) if args.output else run_dir / "empirical_global_sims.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        steps=np.array(steps, dtype=np.int64),
        empirical_global_values=raw,
        empirical_global_sims=sims,
        empirical_global_norms=norms,
        n_samples=args.n_samples,
        split="train",
        seed=args.seed,
        run_dir=str(run_dir),
    )
    print(f"Wrote empirical global data: {output_path}")
    print(f"steps={steps}")
    print(f"finite_global_sims={np.isfinite(sims).sum()}/{sims.size}")

    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        plot_heatmap(raw, np.array(steps), plot_dir / "empirical_global_raw.png", title="Empirical global raw", vmin=None, vmax=None, cmap=args.cmap)
        plot_heatmap(
            sims,
            np.array(steps),
            plot_dir / "empirical_global_sim.png",
            title="Empirical global cosine",
            vmin=args.vmin,
            vmax=args.vmax,
            cmap=args.cmap,
        )
        write_summary_csv(plot_dir / "empirical_global_sim_summary.csv", sims, np.array(steps))
        print(f"Wrote empirical global heatmaps: {plot_dir}")


if __name__ == "__main__":
    main()
