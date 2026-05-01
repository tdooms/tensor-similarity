#!/usr/bin/env python3
"""Score heads by relative-position attention, independent of token identity."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.induction_heads.analyze_sequence_formats import (  # noqa: E402
    AttentionCapture,
    generate_format_sequences,
)
from experiments.induction_heads.data import RepeatedTokenDataset  # noqa: E402
from experiments.induction_heads.score_head_roles import (  # noqa: E402
    FORMAT_CHOICES,
    generate_balanced_offset_sequences,
)
from models import AttentionLM  # noqa: E402


EXTRA_FORMATS = ("variable_gap", "balanced_offsets")
ALL_FORMATS = (*FORMAT_CHOICES, *EXTRA_FORMATS)


def resolve_run_paths(checkpoint_dir_arg: str, step: int | None):
    checkpoint_dir = Path(__file__).parent / checkpoint_dir_arg
    if checkpoint_dir.name == "checkpoints":
        run_dir = checkpoint_dir.parent
    else:
        run_dir = checkpoint_dir
        checkpoint_dir = run_dir / "checkpoints"

    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if step is None:
        checkpoint_path = run_dir / "final.pt"
        step_label = "final"
    else:
        checkpoint_path = checkpoint_dir / f"step_{step}.pt"
        step_label = str(step)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return config_path, checkpoint_path, step_label


def tokens_for_format(cfg: Dict, format_type: str, n_samples: int,
                      bos_token_id: int | None) -> torch.Tensor:
    if format_type == "variable_gap":
        return RepeatedTokenDataset(
            vocab_size=cfg["model"]["vocab_size"],
            n_ctx=cfg["model"]["n_ctx"],
            n_samples=n_samples,
            seed=43,
            bos_token_id=bos_token_id,
        ).data
    if format_type == "balanced_offsets":
        return generate_balanced_offset_sequences(
            vocab_size=cfg["model"]["vocab_size"],
            n_ctx=cfg["model"]["n_ctx"],
            n_per_offset=n_samples,
            bos_token_id=bos_token_id,
        )
    tokens, _ = generate_format_sequences(
        format_type,
        cfg["model"]["vocab_size"],
        cfg["model"]["n_ctx"],
        n_samples,
        bos_token_id=bos_token_id,
    )
    return tokens


def score_relative_offset(pattern: torch.Tensor, offset: int, bos_offset: int) -> dict:
    scores = []
    shares = []
    ranks = []
    top1_hits = []

    batch, n_ctx, _ = pattern.shape
    for b in range(batch):
        for q in range(max(bos_offset + offset, 1), n_ctx):
            k = q - offset
            if k < 0 or k >= q:
                continue
            row = pattern[b, q, :q].float()
            val = row[k].item()
            row_sum = row.sum().item()
            scores.append(val)
            shares.append(val / row_sum if row_sum > 0 else 0.0)
            ranks.append((row <= row[k]).float().mean().item())
            top1_hits.append(1.0 if bool(row[k] >= row.max()) else 0.0)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "n": len(scores),
        "mean_score": mean(scores),
        "mean_share": mean(shares),
        "top1_rate": mean(top1_hits),
        "rank_percentile": mean(ranks),
    }


def score_format(model, cfg: Dict, format_type: str, n_samples: int,
                 bos_token_id: int | None) -> list[dict]:
    tokens = tokens_for_format(cfg, format_type, n_samples, bos_token_id)
    capture = AttentionCapture(model, cfg["model"]["attn_type"])
    capture.register_hooks()
    with torch.no_grad():
        _ = model(tokens)
    capture.remove_hooks()

    bos_offset = 1 if bos_token_id is not None else 0
    max_offset = cfg["model"]["n_ctx"] - bos_offset - 1
    rows = []
    for (layer, head, circuit), pattern in sorted(capture.patterns.items()):
        if circuit != "combined":
            continue
        for offset in range(1, max_offset + 1):
            score = score_relative_offset(pattern, offset, bos_offset)
            rows.append(
                {
                    "format": format_type,
                    "layer": layer,
                    "head": head,
                    "offset": offset,
                    "n": score["n"],
                    "mean_score": score["mean_score"],
                    "mean_share": score["mean_share"],
                    "top1_rate": score["top1_rate"],
                    "rank_percentile": score["rank_percentile"],
                }
            )
    return rows


def print_profile(rows: list[dict]) -> None:
    print(f"{'Format':<16} {'Head':<5} {'Top offset':>10} {'Share':>8} {'Top1':>8} {'Rank':>8}")
    print("-" * 66)
    groups = {}
    for row in rows:
        groups.setdefault((row["format"], row["layer"], row["head"]), []).append(row)
    for (fmt, layer, head), group in sorted(groups.items()):
        best = max(group, key=lambda r: r["mean_share"])
        print(
            f"{fmt:<16} L{layer}H{head:<2} {best['offset']:>10} "
            f"{best['mean_share']:>8.3f} {best['top1_rate']:>8.3f} "
            f"{best['rank_percentile']:>8.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=512)
    parser.add_argument("--format", choices=("all", *ALL_FORMATS), default="all")
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    config_path, checkpoint_path, step_label = resolve_run_paths(args.checkpoint_dir, args.step)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model = AttentionLM.from_config(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data_cfg = cfg.get("data", {})
    bos_token_id = None
    if data_cfg.get("use_bos", False):
        bos_token_id = data_cfg.get("bos_token_id", cfg["model"]["vocab_size"] - 1)

    formats = ALL_FORMATS if args.format == "all" else (args.format,)
    rows = []
    for fmt in formats:
        rows.extend(score_format(model, cfg, fmt, args.n_samples, bos_token_id))

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Step: {step_label}")
    print(f"Samples per format: {args.n_samples}")
    print_profile(rows)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote CSV: {output_csv}")


if __name__ == "__main__":
    main()
