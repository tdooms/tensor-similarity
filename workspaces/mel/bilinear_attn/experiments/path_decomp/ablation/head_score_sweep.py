#!/usr/bin/env python3
"""Sweep attention role scores across checkpoints on variable-gap data."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.induction_heads.analyze_sequence_formats import AttentionCapture  # noqa: E402
from experiments.induction_heads.score_pattern_accuracy import data_for_format, resolve_run_paths  # noqa: E402
from experiments.path_decomp.ablation.head_ablation import canonical_shape  # noqa: E402
from models import AttentionLM  # noqa: E402


DEFAULT_STEPS = list(range(0, 15001, 500))


def _prev_and_next_targets(tokens: torch.Tensor, mask: torch.Tensor, bos_token_id: int | None) -> list[dict]:
    """Use only supervised/evaluated target positions from the dataset mask."""
    rows = []
    for b in range(tokens.shape[0]):
        seq = tokens[b].tolist()
        for target_pos in range(1, tokens.shape[1]):
            if not bool(mask[b, target_pos]):
                continue
            query_pos = target_pos - 1
            query_tok = seq[query_pos]
            if query_tok == bos_token_id:
                continue
            prev = None
            for key_pos in range(query_pos - 1, -1, -1):
                if seq[key_pos] == query_tok:
                    prev = key_pos
                    break
            if prev is None or prev + 1 >= query_pos:
                continue
            rows.append({
                "batch": b,
                "query": query_pos,
                "prev_same": prev,
                "next_after_prev": prev + 1,
                "next_offset": query_pos - (prev + 1),
            })
    return rows


def _score_targets(pattern: torch.Tensor, targets: list[dict], key_name: str) -> dict:
    shares = []
    top1s = []
    ranks = []
    scores = []
    for target in targets:
        b = target["batch"]
        q = target["query"]
        k = target[key_name]
        if q <= 0 or k is None or k >= q:
            continue
        row = pattern[b, q, :q].float()
        val = row[k]
        denom = row.sum()
        shares.append(float((val / denom).item()) if denom > 0 else 0.0)
        top1s.append(1.0 if bool(val >= row.max()) else 0.0)
        ranks.append(float((row <= val).float().mean().item()))
        scores.append(float(val.item()))

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "n": len(shares),
        "mean_score": mean(scores),
        "mean_share": mean(shares),
        "top1_rate": mean(top1s),
        "rank_percentile": mean(ranks),
    }


def _score_relative(pattern: torch.Tensor, bos_offset: int, max_offset: int) -> list[dict]:
    rows = []
    for offset in range(1, max_offset + 1):
        shares = []
        top1s = []
        ranks = []
        scores = []
        for b in range(pattern.shape[0]):
            for q in range(max(bos_offset + offset, 1), pattern.shape[1]):
                k = q - offset
                if k < 0 or k >= q:
                    continue
                row = pattern[b, q, :q].float()
                val = row[k]
                denom = row.sum()
                shares.append(float((val / denom).item()) if denom > 0 else 0.0)
                top1s.append(1.0 if bool(val >= row.max()) else 0.0)
                ranks.append(float((row <= val).float().mean().item()))
                scores.append(float(val.item()))

        def mean(xs: list[float]) -> float:
            return sum(xs) / len(xs) if xs else 0.0

        rows.append({
            "offset": offset,
            "n": len(shares),
            "mean_score": mean(scores),
            "mean_share": mean(shares),
            "top1_rate": mean(top1s),
            "rank_percentile": mean(ranks),
        })
    return rows


def _shape_classes(tokens: torch.Tensor, mask: torch.Tensor, bos_token_id: int | None) -> dict[str, str]:
    offsets_by_shape = defaultdict(set)
    for b in range(tokens.shape[0]):
        seq = tokens[b].tolist()
        shape = canonical_shape(seq, bos_token_id)
        for target_pos in range(1, tokens.shape[1]):
            if not bool(mask[b, target_pos]):
                continue
            q = target_pos - 1
            tok = seq[q]
            prev = None
            for k in range(q - 1, -1, -1):
                if seq[k] == tok:
                    prev = k
                    break
            if prev is not None and prev + 1 < q:
                offsets_by_shape[shape].add(q - (prev + 1))
    return {
        shape: "fixed_offset" if len(offsets) == 1 else "variable_offset"
        for shape, offsets in offsets_by_shape.items()
    }


def load_model(cfg: dict, checkpoint_path: Path) -> AttentionLM:
    model = AttentionLM.from_config(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def score_step(model: AttentionLM, cfg: dict, tokens: torch.Tensor, mask: torch.Tensor,
               bos_token_id: int | None) -> tuple[list[dict], list[dict]]:
    targets = _prev_and_next_targets(tokens, mask, bos_token_id)
    capture = AttentionCapture(model, cfg["model"]["attn_type"])
    capture.register_hooks()
    with torch.no_grad():
        _ = model(tokens)
    capture.remove_hooks()

    bos_offset = 1 if bos_token_id is not None else 0
    max_offset = cfg["model"]["n_ctx"] - bos_offset - 1
    role_rows = []
    rel_rows = []
    for (layer, head, circuit), pattern in sorted(capture.patterns.items()):
        if circuit != "combined":
            continue
        prev = _score_targets(pattern, targets, "prev_same")
        nxt = _score_targets(pattern, targets, "next_after_prev")
        role_rows.append({
            "layer": layer,
            "head": head,
            "prev_n": prev["n"],
            "prev_mean_score": prev["mean_score"],
            "prev_mean_share": prev["mean_share"],
            "prev_top1_rate": prev["top1_rate"],
            "prev_rank_percentile": prev["rank_percentile"],
            "next_n": nxt["n"],
            "next_mean_score": nxt["mean_score"],
            "next_mean_share": nxt["mean_share"],
            "next_top1_rate": nxt["top1_rate"],
            "next_rank_percentile": nxt["rank_percentile"],
        })
        for row in _score_relative(pattern, bos_offset, max_offset):
            rel_rows.append({"layer": layer, "head": head, **row})
    return role_rows, rel_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_role_scores(rows: list[dict], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    metrics = [
        ("prev_mean_share", "previous-same token attention share"),
        ("next_mean_share", "next-after-previous token attention share"),
    ]
    colors = {"L0H0": "#2f6fbb", "L0H1": "#b64a3a", "L1H0": "#3c8f4f", "L1H1": "#8a5fbf"}
    for ax, (metric, title) in zip(axes, metrics):
        for head in ["L0H0", "L0H1", "L1H0", "L1H1"]:
            series = [
                r for r in rows
                if f"L{r['layer']}H{r['head']}" == head
            ]
            series.sort(key=lambda r: int(r["step"]))
            ax.plot(
                [int(r["step"]) for r in series],
                [float(r[metric]) for r in series],
                marker="o",
                linewidth=1.8,
                markersize=3,
                color=colors[head],
                label=head,
            )
        ax.set_title(title)
        ax.set_ylabel("mean share")
        ax.grid(alpha=0.25)
        ax.legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("checkpoint step")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_relative_scores(rows: list[dict], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True, constrained_layout=True)
    colors = {1: "#2f6fbb", 2: "#b64a3a", 3: "#3c8f4f", 4: "#8a5fbf", 5: "#cc8b00", 6: "#666666", 7: "#111111"}
    for ax, head in zip(axes.flat, ["L0H0", "L0H1", "L1H0", "L1H1"]):
        head_rows = [r for r in rows if f"L{r['layer']}H{r['head']}" == head]
        for offset in sorted({int(r["offset"]) for r in head_rows}):
            series = [r for r in head_rows if int(r["offset"]) == offset]
            series.sort(key=lambda r: int(r["step"]))
            ax.plot(
                [int(r["step"]) for r in series],
                [float(r["mean_share"]) for r in series],
                marker="o",
                linewidth=1.3,
                markersize=2.5,
                color=colors.get(offset),
                label=f"q-{offset}",
            )
        ax.set_title(head)
        ax.set_ylabel("mean share")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
    for ax in axes[-1]:
        ax.set_xlabel("checkpoint step")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", default="runs/small-big-experiment-runs")
    parser.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS)
    parser.add_argument("--n_samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=12345)
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

    torch.manual_seed(args.seed)
    tokens, mask = data_for_format(cfg, "variable_gap", args.n_samples, bos_token_id)

    role_rows = []
    rel_rows = []
    for step in args.steps:
        model = load_model(cfg, run_dir / "checkpoints" / f"step_{step}.pt")
        step_role_rows, step_rel_rows = score_step(model, cfg, tokens, mask, bos_token_id)
        for row in step_role_rows:
            role_rows.append({"step": step, **row})
        for row in step_rel_rows:
            rel_rows.append({"step": step, **row})
        print(f"finished step {step}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else run_dir / "path_decomp_trajectory" / "head_score_sweep_500"
    )
    write_csv(output_dir / "head_role_scores_over_time.csv", role_rows)
    write_csv(output_dir / "relative_position_scores_over_time.csv", rel_rows)
    plot_role_scores(role_rows, output_dir / "previous_and_next_head_scores.png")
    plot_relative_scores(rel_rows, output_dir / "relative_position_scores.png")
    print(f"Wrote outputs: {output_dir}")


if __name__ == "__main__":
    main()
