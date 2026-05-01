#!/usr/bin/env python3
"""Evaluate next-token accuracy by sequence pattern."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.induction_heads.data import RepeatedTokenDataset  # noqa: E402
from experiments.induction_heads.score_head_roles import (  # noqa: E402
    FORMAT_CHOICES,
    _previous_same_targets,
    _tokens_for_format,
)
from models import AttentionLM  # noqa: E402


PATTERN_CHOICES = (*FORMAT_CHOICES, "variable_gap")


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


def eval_mask_for_format(tokens: torch.Tensor, format_type: str,
                         bos_token_id: int | None) -> torch.Tensor:
    """Return next-token positions whose target is implied by induction.

    A target position ``t`` is evaluated when the query position ``p=t-1``
    contains a token that appeared earlier at ``prev``, and the target token is
    exactly the token after that previous occurrence: ``tokens[t] == tokens[prev+1]``.
    """
    if format_type == "variable_gap":
        raise ValueError("variable_gap mask should come from RepeatedTokenDataset")

    mask = torch.zeros_like(tokens, dtype=torch.bool)
    batch, n_ctx = tokens.shape
    for b in range(batch):
        seq = tokens[b].tolist()
        for target_pos in range(1, n_ctx):
            query_pos = target_pos - 1
            query_tok = seq[query_pos]
            if query_tok == bos_token_id:
                continue
            prev = None
            for k in range(query_pos - 1, -1, -1):
                if seq[k] == query_tok:
                    prev = k
                    break
            if prev is None or prev + 1 >= query_pos:
                continue
            if seq[target_pos] == seq[prev + 1]:
                mask[b, target_pos] = True
    return mask


def data_for_format(cfg: Dict, format_type: str, n_samples: int,
                    bos_token_id: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    if format_type == "variable_gap":
        ds = RepeatedTokenDataset(
            vocab_size=cfg["model"]["vocab_size"],
            n_ctx=cfg["model"]["n_ctx"],
            n_samples=n_samples,
            seed=43,
            bos_token_id=bos_token_id,
        )
        return ds.data, ds.repeat_masks

    tokens = _tokens_for_format(cfg, format_type, n_samples, bos_token_id)
    return tokens, eval_mask_for_format(tokens, format_type, bos_token_id)


def score_accuracy(model, cfg: Dict, format_type: str, n_samples: int,
                   bos_token_id: int | None) -> dict:
    tokens, mask = data_for_format(cfg, format_type, n_samples, bos_token_id)
    with torch.no_grad():
        logits = model(tokens)
    preds = logits.argmax(dim=-1)

    eval_mask = mask.clone()
    eval_mask[:, 0] = False
    shifted_preds = preds[:, :-1]
    shifted_targets = tokens[:, 1:]
    shifted_mask = eval_mask[:, 1:]

    correct = (shifted_preds == shifted_targets) & shifted_mask
    total = int(shifted_mask.sum().item())
    total_correct = int(correct.sum().item())

    # Offset here is the attention offset from query position ``t-1`` to
    # next-after-previous position ``prev+1``.
    offset_totals = {}
    offset_correct = {}
    batch, n_ctx = tokens.shape
    for b in range(batch):
        seq = tokens[b].tolist()
        for target_pos in range(1, n_ctx):
            if not bool(mask[b, target_pos]):
                continue
            query_pos = target_pos - 1
            query_tok = seq[query_pos]
            prev = None
            for k in range(query_pos - 1, -1, -1):
                if seq[k] == query_tok:
                    prev = k
                    break
            if prev is None:
                continue
            next_after_prev = prev + 1
            offset = query_pos - next_after_prev
            if offset < 1:
                continue
            offset_totals[offset] = offset_totals.get(offset, 0) + 1
            is_correct = preds[b, query_pos].item() == tokens[b, target_pos].item()
            offset_correct[offset] = offset_correct.get(offset, 0) + int(is_correct)

    return {
        "format": format_type,
        "n_samples": tokens.shape[0],
        "n_eval_tokens": total,
        "correct": total_correct,
        "accuracy": total_correct / total if total else 0.0,
        "offset_accuracy": {
            offset: offset_correct.get(offset, 0) / count
            for offset, count in sorted(offset_totals.items())
        },
        "offset_counts": dict(sorted(offset_totals.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=2048)
    parser.add_argument("--format", choices=("all", *PATTERN_CHOICES), default="all")
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

    formats = PATTERN_CHOICES if args.format == "all" else (args.format,)
    rows = [score_accuracy(model, cfg, fmt, args.n_samples, bos_token_id) for fmt in formats]

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Step: {step_label}")
    print(f"{'Format':<16} {'Samples':>8} {'EvalTok':>8} {'Correct':>8} {'Acc':>8} {'Offset acc':<30}")
    print("-" * 92)
    for r in rows:
        offset_acc = ", ".join(f"{k}:{v:.3f}" for k, v in r["offset_accuracy"].items())
        print(
            f"{r['format']:<16} {r['n_samples']:>8} {r['n_eval_tokens']:>8} "
            f"{r['correct']:>8} {r['accuracy']:>8.3f} {offset_acc:<30}"
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = []
    for r in rows:
        base = {k: v for k, v in r.items() if k not in ("offset_accuracy", "offset_counts")}
        for offset, acc in r["offset_accuracy"].items():
            row = dict(base)
            row["next_offset"] = offset
            row["offset_count"] = r["offset_counts"][offset]
            row["offset_accuracy"] = acc
            flat_rows.append(row)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"\nWrote CSV: {output_csv}")


if __name__ == "__main__":
    main()
PATTERN_CHOICES = (*FORMAT_CHOICES, "variable_gap")
