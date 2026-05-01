#!/usr/bin/env python3
"""Ablate individual AttentionLM heads and measure induction accuracy.

The intervention zeros a selected head's active value stream `z = pattern @ v`
before the output projection. This leaves the residual part of the attention
layer intact and asks how much accuracy depends on that head's active path.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

import torch
import yaml
from einops import rearrange, einsum

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.induction_heads.score_pattern_accuracy import (  # noqa: E402
    PATTERN_CHOICES,
    data_for_format,
    resolve_run_paths,
)
from models import AttentionLM  # noqa: E402


Ablation = tuple[int, int]


def _parse_head(spec: str) -> Ablation:
    """Parse `L1H0`, `1,0`, or `1:0` into `(layer, head)`."""
    s = spec.strip().upper()
    if s.startswith("L") and "H" in s:
        left, right = s[1:].split("H", 1)
        return int(left), int(right)
    if "," in s:
        left, right = s.split(",", 1)
        return int(left), int(right)
    if ":" in s:
        left, right = s.split(":", 1)
        return int(left), int(right)
    raise argparse.ArgumentTypeError(f"Could not parse head spec: {spec!r}")


def _quadratic_attention_with_head_mask(layer, x: torch.Tensor, ablated_heads: set[int]) -> torch.Tensor:
    _, t, _ = x.shape
    q = rearrange(layer.q(x), "b t (n d) -> b t n d", n=layer.n_head)
    k = rearrange(layer.k(x), "b t (n d) -> b t n d", n=layer.n_head)
    v = rearrange(layer.v(x), "b t (n d) -> b t n d", n=layer.n_head)

    q = layer.rotary(layer.norm_qk(q))
    k = layer.rotary(layer.norm_qk(k))
    scores = einsum(q, k, "b tq n d, b tk n d -> b n tq tk")
    pattern = (scores / layer.d_head).square()
    pattern = pattern * layer.causal_mask[None, None, :t, :t]
    z = einsum(pattern, v, "b n tq tk, b tk n d -> b tq n d")
    if ablated_heads:
        z = z.clone()
        z[:, :, sorted(ablated_heads), :] = 0

    z_merge = rearrange(z, "b t n d -> b t (n d)")
    return x + layer.scale * (layer.o(z_merge) - x)


def _bilinear_attention_with_head_mask(layer, x: torch.Tensor, ablated_heads: set[int]) -> torch.Tensor:
    _, t, _ = x.shape
    q1 = rearrange(layer.q1(x), "b t (n d) -> b t n d", n=layer.n_head)
    k1 = rearrange(layer.k1(x), "b t (n d) -> b t n d", n=layer.n_head)
    q2 = rearrange(layer.q2(x), "b t (n d) -> b t n d", n=layer.n_head)
    k2 = rearrange(layer.k2(x), "b t (n d) -> b t n d", n=layer.n_head)
    v = rearrange(layer.v(x), "b t (n d) -> b t n d", n=layer.n_head)

    q1 = layer.rotary(layer.norm_qk(q1))
    k1 = layer.rotary(layer.norm_qk(k1))
    q2 = layer.rotary(layer.norm_qk(q2))
    k2 = layer.rotary(layer.norm_qk(k2))
    scores1 = einsum(q1, k1, "b tq n d, b tk n d -> b n tq tk")
    scores2 = einsum(q2, k2, "b tq n d, b tk n d -> b n tq tk")
    pattern = (scores1 * scores2) / layer.d_head**2
    pattern = pattern * layer.causal_mask[None, None, :t, :t]
    z = einsum(pattern, v, "b n tq tk, b tk n d -> b tq n d")
    if ablated_heads:
        z = z.clone()
        z[:, :, sorted(ablated_heads), :] = 0

    z_merge = rearrange(z, "b t n d -> b t (n d)")
    return x + layer.scale * (layer.o(z_merge) - x)


def _layer_with_head_mask(layer, x: torch.Tensor, ablated_heads: set[int]) -> torch.Tensor:
    if hasattr(layer, "q") and hasattr(layer, "k"):
        return _quadratic_attention_with_head_mask(layer, x, ablated_heads)
    if hasattr(layer, "q1") and hasattr(layer, "k1"):
        return _bilinear_attention_with_head_mask(layer, x, ablated_heads)
    raise NotImplementedError(f"Head masking is not implemented for {type(layer).__name__}")


@torch.no_grad()
def forward_with_ablations(model: AttentionLM, input_ids: torch.Tensor, ablations: set[Ablation]) -> torch.Tensor:
    by_layer: dict[int, set[int]] = {}
    for layer_idx, head_idx in ablations:
        by_layer.setdefault(layer_idx, set()).add(head_idx)

    x = model.embed(input_ids)
    if model.embed_norm is not None:
        x = model.embed_norm(x)

    if model.layer_norms is not None:
        for layer_idx, (norm, layer) in enumerate(zip(model.layer_norms, model.layers)):
            x_normed = norm(x)
            heads = by_layer.get(layer_idx)
            if heads:
                out = _layer_with_head_mask(layer, x_normed, heads)
            else:
                out = layer(x_normed)
            x = x + (out - x_normed)
    else:
        for layer_idx, layer in enumerate(model.layers):
            heads = by_layer.get(layer_idx)
            if heads:
                x = _layer_with_head_mask(layer, x, heads)
            else:
                x = layer(x)

    return model.unembed(model.final_norm(x))


def _offset_for_target(seq: list[int], target_pos: int) -> int | None:
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


def canonical_shape(seq: list[int], bos_token_id: int | None) -> str:
    """Map token ids to first-occurrence labels, excluding BOS from the shape."""
    names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    mapping: dict[int, str] = {}
    parts = []
    for tok in seq:
        if bos_token_id is not None and tok == bos_token_id:
            continue
        if tok not in mapping:
            mapping[tok] = names[len(mapping)] if len(mapping) < len(names) else f"T{len(mapping)}"
        parts.append(mapping[tok])
    return "".join(parts)


def score_logits(tokens: torch.Tensor, mask: torch.Tensor, logits: torch.Tensor, format_type: str) -> tuple[dict, list[dict]]:
    preds = logits.argmax(dim=-1)
    shifted_mask = mask[:, 1:].clone()
    correct = (preds[:, :-1] == tokens[:, 1:]) & shifted_mask
    total = int(shifted_mask.sum().item())
    total_correct = int(correct.sum().item())

    summary = {
        "format": format_type,
        "n_samples": int(tokens.shape[0]),
        "n_eval_tokens": total,
        "correct": total_correct,
        "accuracy": total_correct / total if total else 0.0,
    }

    by_offset: dict[int, dict[str, int]] = {}
    for b in range(tokens.shape[0]):
        seq = tokens[b].tolist()
        for target_pos in range(1, tokens.shape[1]):
            if not bool(mask[b, target_pos]):
                continue
            offset = _offset_for_target(seq, target_pos)
            if offset is None:
                continue
            row = by_offset.setdefault(offset, {"n_eval_tokens": 0, "correct": 0})
            row["n_eval_tokens"] += 1
            row["correct"] += int(preds[b, target_pos - 1].item() == tokens[b, target_pos].item())

    offset_rows = []
    for offset, counts in sorted(by_offset.items()):
        n_eval = counts["n_eval_tokens"]
        offset_rows.append({
            "format": format_type,
            "next_offset": offset,
            "n_eval_tokens": n_eval,
            "correct": counts["correct"],
            "accuracy": counts["correct"] / n_eval if n_eval else 0.0,
        })
    return summary, offset_rows


def score_logits_by_shape(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    logits: torch.Tensor,
    format_type: str,
    bos_token_id: int | None,
) -> list[dict]:
    preds = logits.argmax(dim=-1)
    by_shape: dict[str, dict[str, int]] = {}
    for b in range(tokens.shape[0]):
        shape = canonical_shape(tokens[b].tolist(), bos_token_id)
        row = by_shape.setdefault(shape, {"n_sequences": 0, "n_eval_tokens": 0, "correct": 0})
        row["n_sequences"] += 1
        for target_pos in range(1, tokens.shape[1]):
            if not bool(mask[b, target_pos]):
                continue
            row["n_eval_tokens"] += 1
            row["correct"] += int(preds[b, target_pos - 1].item() == tokens[b, target_pos].item())

    rows = []
    for shape, counts in sorted(by_shape.items(), key=lambda kv: (-kv[1]["n_eval_tokens"], kv[0])):
        n_eval = counts["n_eval_tokens"]
        rows.append({
            "format": format_type,
            "shape": shape,
            "n_sequences": counts["n_sequences"],
            "n_eval_tokens": n_eval,
            "correct": counts["correct"],
            "accuracy": counts["correct"] / n_eval if n_eval else 0.0,
        })
    return rows


def score_logits_by_shape_position(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    logits: torch.Tensor,
    format_type: str,
    bos_token_id: int | None,
) -> list[dict]:
    preds = logits.argmax(dim=-1)
    by_shape_pos: dict[tuple[str, int], dict[str, int]] = {}
    for b in range(tokens.shape[0]):
        seq = tokens[b].tolist()
        shape = canonical_shape(seq, bos_token_id)
        content_pos_offset = 1 if bos_token_id is not None else 0
        for target_pos in range(1, tokens.shape[1]):
            if not bool(mask[b, target_pos]):
                continue
            content_target_pos = target_pos - content_pos_offset
            key = (shape, content_target_pos)
            row = by_shape_pos.setdefault(key, {"n_eval_tokens": 0, "correct": 0})
            row["n_eval_tokens"] += 1
            row["correct"] += int(preds[b, target_pos - 1].item() == tokens[b, target_pos].item())

    rows = []
    for (shape, content_target_pos), counts in sorted(
        by_shape_pos.items(), key=lambda kv: (kv[0][0], kv[0][1])
    ):
        n_eval = counts["n_eval_tokens"]
        rows.append({
            "format": format_type,
            "shape": shape,
            "content_target_pos": content_target_pos,
            "n_eval_tokens": n_eval,
            "correct": counts["correct"],
            "accuracy": counts["correct"] / n_eval if n_eval else 0.0,
        })
    return rows


def _condition_label(ablation: Ablation | None) -> str:
    if ablation is None:
        return "baseline"
    return f"L{ablation[0]}H{ablation[1]}_ablated"


@torch.no_grad()
def evaluate_conditions(
    model: AttentionLM,
    cfg: dict,
    formats: Iterable[str],
    n_samples: int,
    bos_token_id: int | None,
    ablations: list[Ablation],
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    summary_rows: list[dict] = []
    offset_rows: list[dict] = []
    shape_rows: list[dict] = []
    shape_position_rows: list[dict] = []

    for format_idx, format_type in enumerate(formats):
        torch.manual_seed(seed + format_idx)
        tokens, mask = data_for_format(cfg, format_type, n_samples, bos_token_id)
        conditions: list[Ablation | None] = [None, *ablations]
        baseline_summary = None
        baseline_offsets: dict[int, float] = {}

        for ablation in conditions:
            label = _condition_label(ablation)
            logits = (
                model(tokens)
                if ablation is None
                else forward_with_ablations(model, tokens, {ablation})
            )
            summary, offsets = score_logits(tokens, mask, logits, format_type)
            shapes = score_logits_by_shape(tokens, mask, logits, format_type, bos_token_id)
            shape_positions = score_logits_by_shape_position(tokens, mask, logits, format_type, bos_token_id)
            summary["condition"] = label

            if ablation is None:
                baseline_summary = summary
                baseline_offsets = {int(r["next_offset"]): float(r["accuracy"]) for r in offsets}
                baseline_shapes = {str(r["shape"]): float(r["accuracy"]) for r in shapes}
                baseline_shape_positions = {
                    (str(r["shape"]), int(r["content_target_pos"])): float(r["accuracy"])
                    for r in shape_positions
                }
                summary["delta_accuracy"] = 0.0
                summary["error_increase"] = 0.0
            else:
                assert baseline_summary is not None
                summary["delta_accuracy"] = summary["accuracy"] - baseline_summary["accuracy"]
                summary["error_increase"] = (
                    (1.0 - summary["accuracy"]) - (1.0 - baseline_summary["accuracy"])
                )
            summary_rows.append(summary)

            for row in offsets:
                row["condition"] = label
                base_acc = baseline_offsets.get(int(row["next_offset"]))
                row["baseline_accuracy"] = base_acc
                row["delta_accuracy"] = 0.0 if base_acc is None else row["accuracy"] - base_acc
                offset_rows.append(row)

            for row in shapes:
                row["condition"] = label
                base_acc = baseline_shapes.get(str(row["shape"]))
                row["baseline_accuracy"] = base_acc
                row["delta_accuracy"] = 0.0 if base_acc is None else row["accuracy"] - base_acc
                shape_rows.append(row)

            for row in shape_positions:
                row["condition"] = label
                key = (str(row["shape"]), int(row["content_target_pos"]))
                base_acc = baseline_shape_positions.get(key)
                row["baseline_accuracy"] = base_acc
                row["delta_accuracy"] = 0.0 if base_acc is None else row["accuracy"] - base_acc
                shape_position_rows.append(row)

    return summary_rows, offset_rows, shape_rows, shape_position_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: list[dict]) -> None:
    print(f"{'format':<14} {'condition':<15} {'eval':>7} {'correct':>7} {'acc':>8} {'delta':>8}")
    print("-" * 70)
    for row in rows:
        print(
            f"{row['format']:<14} {row['condition']:<15} {row['n_eval_tokens']:>7} "
            f"{row['correct']:>7} {row['accuracy']:>8.3f} {row['delta_accuracy']:>8.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--step", type=int, default=15000)
    parser.add_argument("--n_samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--format", choices=("all", *PATTERN_CHOICES), default="all")
    parser.add_argument("--ablate", type=_parse_head, action="append", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    config_path, checkpoint_path, step_label = resolve_run_paths(args.checkpoint_dir, args.step)
    with config_path.open() as f:
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
    ablations = args.ablate if args.ablate is not None else [(1, 0), (1, 1)]
    summary_rows, offset_rows, shape_rows, shape_position_rows = evaluate_conditions(
        model=model,
        cfg=cfg,
        formats=formats,
        n_samples=args.n_samples,
        bos_token_id=bos_token_id,
        ablations=ablations,
        seed=args.seed,
    )

    if args.output_dir is None:
        run_dir = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent
        output_dir = ROOT / "experiments" / "path_decomp" / "ablation_runs" / f"{run_dir.name}_step_{step_label}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = output_dir / "head_ablation_summary.csv"
    offset_csv = output_dir / "head_ablation_by_offset.csv"
    shape_csv = output_dir / "head_ablation_by_shape.csv"
    shape_position_csv = output_dir / "head_ablation_by_shape_position.csv"
    _write_csv(summary_csv, summary_rows)
    _write_csv(offset_csv, offset_rows)
    _write_csv(shape_csv, shape_rows)
    _write_csv(shape_position_csv, shape_position_rows)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Step: {step_label}")
    print(f"Ablations: {', '.join(_condition_label(a) for a in ablations)}")
    print()
    _print_summary(summary_rows)
    print()
    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote offset CSV:  {offset_csv}")
    print(f"Wrote shape CSV:   {shape_csv}")
    print(f"Wrote shape-pos CSV: {shape_position_csv}")


if __name__ == "__main__":
    main()
