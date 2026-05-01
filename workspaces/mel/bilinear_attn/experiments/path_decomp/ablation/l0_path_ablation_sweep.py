#!/usr/bin/env python3
"""Sweep layer-0 path ablations across checkpoints and plot class metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.induction_heads.score_pattern_accuracy import data_for_format, resolve_run_paths  # noqa: E402
from experiments.path_decomp.ablation.head_ablation import canonical_shape  # noqa: E402
from experiments.path_decomp.ablation.l0_path_ablation import forward_l0_path_ablation  # noqa: E402
from models import AttentionLM  # noqa: E402


DEFAULT_STEPS = list(range(0, 15001, 500))
MODES = ["direct", "layer2", "both"]
HEADS = [0, 1]


def _next_offset(seq: list[int], target_pos: int) -> int | None:
    query_pos = target_pos - 1
    query_tok = seq[query_pos]
    prev = None
    for key_pos in range(query_pos - 1, -1, -1):
        if seq[key_pos] == query_tok:
            prev = key_pos
            break
    if prev is None:
        return None
    offset = query_pos - (prev + 1)
    return offset if offset >= 1 else None


def shape_classes(tokens: torch.Tensor, mask: torch.Tensor, bos_token_id: int | None) -> dict[str, str]:
    offsets_by_shape: dict[str, set[int]] = defaultdict(set)
    for seq_t, mask_t in zip(tokens, mask):
        seq = seq_t.tolist()
        shape = canonical_shape(seq, bos_token_id)
        for target_pos in range(1, len(seq)):
            if not bool(mask_t[target_pos]):
                continue
            offset = _next_offset(seq, target_pos)
            if offset is not None:
                offsets_by_shape[shape].add(offset)
    return {
        shape: "fixed_offset" if len(offsets) == 1 else "variable_offset"
        for shape, offsets in offsets_by_shape.items()
    }


def eval_by_class(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    classes: dict[str, str],
    bos_token_id: int | None,
) -> list[dict]:
    preds = logits[:, :-1].argmax(dim=-1)
    targets = tokens[:, 1:]
    eval_mask = mask[:, 1:]
    losses = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)

    rows = defaultdict(lambda: {"n_eval_tokens": 0, "correct": 0, "loss_sum": 0.0})
    for b in range(tokens.shape[0]):
        cls = classes[canonical_shape(tokens[b].tolist(), bos_token_id)]
        selected = eval_mask[b]
        n = int(selected.sum().item())
        if n == 0:
            continue
        rows[cls]["n_eval_tokens"] += n
        rows[cls]["correct"] += int(((preds[b] == targets[b]) & selected).sum().item())
        rows[cls]["loss_sum"] += float(losses[b][selected].sum().item())

    out = []
    for cls, stats in sorted(rows.items()):
        n = stats["n_eval_tokens"]
        out.append({
            "offset_class": cls,
            "n_eval_tokens": n,
            "correct": stats["correct"],
            "accuracy": stats["correct"] / n if n else float("nan"),
            "loss": stats["loss_sum"] / n if n else float("nan"),
        })
    return out


def load_model(cfg: dict, checkpoint_path: Path) -> AttentionLM:
    model = AttentionLM.from_config(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def run_sweep(cfg: dict, run_dir: Path, steps: list[int], n_samples: int, seed: int, bos_token_id: int | None) -> list[dict]:
    torch.manual_seed(seed)
    tokens, mask = data_for_format(cfg, "variable_gap", n_samples, bos_token_id)
    classes = shape_classes(tokens, mask, bos_token_id)
    rows = []

    for step in steps:
        checkpoint_path = run_dir / "checkpoints" / f"step_{step}.pt"
        model = load_model(cfg, checkpoint_path)

        with torch.no_grad():
            baseline_logits = forward_l0_path_ablation(model, tokens, None, "baseline")
        baseline_rows = eval_by_class(baseline_logits, tokens, mask, classes, bos_token_id)
        baseline_by_class = {r["offset_class"]: r for r in baseline_rows}
        for r in baseline_rows:
            rows.append({
                "step": step,
                "head": "baseline",
                "mode": "baseline",
                **r,
                "delta_accuracy": 0.0,
                "delta_loss": 0.0,
            })

        for head in HEADS:
            for mode in MODES:
                with torch.no_grad():
                    logits = forward_l0_path_ablation(model, tokens, head, mode)
                scored = eval_by_class(logits, tokens, mask, classes, bos_token_id)
                for r in scored:
                    base = baseline_by_class[r["offset_class"]]
                    rows.append({
                        "step": step,
                        "head": f"L0H{head}",
                        "mode": mode,
                        **r,
                        "delta_accuracy": r["accuracy"] - base["accuracy"],
                        "delta_loss": r["loss"] - base["loss"],
                    })
        print(f"finished step {step}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _pair_index(family_pairs: list[str]) -> dict[str, int]:
    return {pair: i for i, pair in enumerate(family_pairs)}


def tn_lines(trajectory_path: Path, top_n: int) -> list[tuple[str, list[int], list[float]]]:
    if not trajectory_path.exists():
        return []
    with trajectory_path.open() as f:
        traj = json.load(f)
    steps = traj["steps"]
    idx = _pair_index(traj["family_pairs"])
    final_values = traj["local_norm"][-1]
    diagonal_pairs = [p for p in traj["family_pairs"] if p.split("|")[0] == p.split("|")[1]]
    diagonal_pairs.sort(key=lambda p: abs(final_values[idx[p]]), reverse=True)
    selected = diagonal_pairs[:top_n]
    return [
        (pair.replace("layer2:", ""), steps, [traj["local_norm"][i][idx[pair]] for i in range(len(steps))])
        for pair in selected
    ]


def plot_head(rows: list[dict], head: str, output_path: Path, trajectory_path: Path, top_tn: int) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False, constrained_layout=True)
    ax_tn, ax_acc, ax_loss = axes

    for label, steps, values in tn_lines(trajectory_path, top_tn):
        ax_tn.plot(steps, values, marker="o", linewidth=1.5, label=label)
    ax_tn.set_title("Top path-family TN similarity over training (local_norm diagonal)")
    ax_tn.set_ylabel("TN sim")
    ax_tn.grid(alpha=0.25)
    ax_tn.legend(fontsize=7, ncol=3)

    colors = {"fixed_offset": "#2f6fbb", "variable_offset": "#b64a3a"}
    styles = {"baseline": ":", "direct": "--", "layer2": "-", "both": "-."}
    markers = {"baseline": "o", "direct": "s", "layer2": "^", "both": "D"}

    head_rows = [r for r in rows if r["head"] in ("baseline", head)]
    for mode in ["baseline", *MODES]:
        for cls in ["fixed_offset", "variable_offset"]:
            series = [
                r for r in head_rows
                if r["mode"] == mode and r["offset_class"] == cls
            ]
            if not series:
                continue
            series.sort(key=lambda r: int(r["step"]))
            label = f"{cls.replace('_', ' ')} / {mode}"
            steps = [int(r["step"]) for r in series]
            ax_acc.plot(
                steps,
                [float(r["accuracy"]) for r in series],
                color=colors[cls],
                linestyle=styles[mode],
                marker=markers[mode],
                linewidth=1.6,
                markersize=4,
                label=label,
            )
            ax_loss.plot(
                steps,
                [float(r["loss"]) for r in series],
                color=colors[cls],
                linestyle=styles[mode],
                marker=markers[mode],
                linewidth=1.6,
                markersize=4,
                label=label,
            )

    ax_acc.set_title(f"{head} path ablation accuracy")
    ax_acc.set_ylabel("masked accuracy")
    ax_acc.set_ylim(-0.05, 1.05)
    ax_acc.grid(alpha=0.25)
    ax_acc.legend(fontsize=7, ncol=2)

    ax_loss.set_title(f"{head} path ablation loss")
    ax_loss.set_ylabel("masked CE loss")
    ax_loss.set_xlabel("checkpoint step")
    ax_loss.grid(alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", default="runs/small-big-experiment-runs")
    parser.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS)
    parser.add_argument("--n_samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--top_tn", type=int, default=5)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    config_path, _, _ = resolve_run_paths(args.checkpoint_dir, args.steps[-1])
    with config_path.open() as f:
        cfg = yaml.safe_load(f)
    run_dir = config_path.parent

    data_cfg = cfg.get("data", {})
    bos_token_id = None
    if data_cfg.get("use_bos", False):
        bos_token_id = data_cfg.get("bos_token_id", cfg["model"]["vocab_size"] - 1)

    rows = run_sweep(cfg, run_dir, args.steps, args.n_samples, args.seed, bos_token_id)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else run_dir / "path_decomp_trajectory" / "l0_path_ablation_sweep"
    )
    csv_path = output_dir / "l0_path_ablation_sweep_metrics.csv"
    write_csv(csv_path, rows)

    trajectory_path = run_dir / "path_decomp_trajectory" / "trajectory.json"
    for head in ["L0H0", "L0H1"]:
        plot_head(
            rows,
            head,
            output_dir / f"{head}_path_ablation_sweep.png",
            trajectory_path,
            args.top_tn,
        )

    print(f"Wrote metrics: {csv_path}")
    print(f"Wrote plots: {output_dir}")


if __name__ == "__main__":
    main()
