#!/usr/bin/env python3
"""Score heads for previous-same-token and next-token-after-previous roles."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.induction_heads.analyze_sequence_formats import (  # noqa: E402
    AttentionCapture,
    generate_format_sequences,
)
from experiments.induction_heads.data import RepeatedTokenDataset  # noqa: E402
from models import AttentionLM  # noqa: E402


FORMAT_CHOICES = ("ABCABC", "ABCDAB", "ABABAB", "ABCDBC")
ALL_FORMAT_CHOICES = (*FORMAT_CHOICES, "variable_gap", "balanced_offsets")


def _resolve_run_paths(checkpoint_dir_arg: str, step: int | None):
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

    return run_dir, checkpoint_dir, config_path, checkpoint_path, step_label


def _previous_same_targets(tokens: torch.Tensor, bos_token_id: int | None) -> list[dict]:
    """Return query-target rows for previous-same and next-after-previous keys."""
    rows = []
    batch, n_ctx = tokens.shape
    for b in range(batch):
        seq = tokens[b].tolist()
        for q in range(n_ctx):
            tok = seq[q]
            if tok == bos_token_id:
                continue
            prev = None
            for k in range(q - 1, -1, -1):
                if seq[k] == tok:
                    prev = k
                    break
            if prev is None:
                continue
            next_after_prev = prev + 1
            if next_after_prev >= q:
                next_after_prev = None
            rows.append(
                {
                    "batch": b,
                    "query": q,
                    "prev_same": prev,
                    "next_after_prev": next_after_prev,
                    "prev_same_offset": q - prev,
                    "next_after_prev_offset": q - next_after_prev if next_after_prev is not None else None,
                    "prev_pos_1": q - 1 if q - 1 >= 0 else None,
                    "prev_pos_2": q - 2 if q - 2 >= 0 else None,
                    "prev_pos_3": q - 3 if q - 3 >= 0 else None,
                    "prev_pos_4": q - 4 if q - 4 >= 0 else None,
                }
            )
    return rows


def _offset_summary(targets: list[dict], offset_name: str) -> dict:
    counts = {}
    for target in targets:
        offset = target[offset_name]
        if offset is None:
            continue
        counts[offset] = counts.get(offset, 0) + 1
    total = sum(counts.values())
    if not counts:
        return {"n_offsets": 0, "dominant_offset": None, "dominant_frac": 0.0, "counts": {}}
    dominant_offset, dominant_count = max(counts.items(), key=lambda x: x[1])
    return {
        "n_offsets": len(counts),
        "dominant_offset": dominant_offset,
        "dominant_frac": dominant_count / total,
        "counts": dict(sorted(counts.items())),
    }


def _score_target(pattern: torch.Tensor, targets: Iterable[dict], target_name: str) -> dict:
    """Score a head against one target definition."""
    scores = []
    shares = []
    ranks = []
    top1_hits = []
    margins = []
    n_valid = 0

    for target in targets:
        k_target = target[target_name]
        if k_target is None:
            continue
        b = target["batch"]
        q = target["query"]
        if q <= 0:
            continue

        row = pattern[b, q, :q].float()
        if row.numel() == 0:
            continue

        val = row[k_target].item()
        row_sum = row.sum().item()
        other_mask = torch.ones_like(row, dtype=torch.bool)
        other_mask[k_target] = False
        others = row[other_mask]

        # Rank percentile: 1.0 means target is tied for highest earlier-key score.
        num_less_equal = (row <= row[k_target]).float().mean().item()
        top1 = bool(row[k_target] >= row.max())
        margin = val - others.mean().item() if others.numel() else 0.0

        scores.append(val)
        shares.append(val / row_sum if row_sum > 0 else 0.0)
        ranks.append(num_less_equal)
        top1_hits.append(1.0 if top1 else 0.0)
        margins.append(margin)
        n_valid += 1

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "n": n_valid,
        "mean_score": mean(scores),
        "mean_share": mean(shares),
        "rank_percentile": mean(ranks),
        "top1_rate": mean(top1_hits),
        "mean_margin": mean(margins),
    }


def _tokens_for_format(cfg: Dict, format_type: str, n_samples: int,
                       bos_token_id: int | None) -> torch.Tensor:
    if format_type == "balanced_offsets":
        return generate_balanced_offset_sequences(
            vocab_size=cfg["model"]["vocab_size"],
            n_ctx=cfg["model"]["n_ctx"],
            n_per_offset=n_samples,
            bos_token_id=bos_token_id,
        )

    if format_type == "variable_gap":
        dataset = RepeatedTokenDataset(
            vocab_size=cfg["model"]["vocab_size"],
            n_ctx=cfg["model"]["n_ctx"],
            n_samples=n_samples,
            seed=43,
            bos_token_id=bos_token_id,
        )
        return dataset.data

    tokens, _desc = generate_format_sequences(
        format_type,
        cfg["model"]["vocab_size"],
        cfg["model"]["n_ctx"],
        n_samples,
        bos_token_id=bos_token_id,
    )
    return tokens


def generate_balanced_offset_sequences(vocab_size: int, n_ctx: int, n_per_offset: int,
                                       bos_token_id: int | None) -> torch.Tensor:
    """Generate balanced probes across every feasible previous-same offset.

    For each offset d, place token A at q-d and q. Place a distinct token B at
    q-d+1 so ``next_after_prev`` is also well-defined. The remaining content
    positions are filled with unique tokens disjoint from A and B, so the only
    repeated token is A and each generated sequence contributes one probe.
    """
    rng = torch.Generator().manual_seed(12345)
    bos_offset = 1 if bos_token_id is not None else 0
    content_tokens = [t for t in range(vocab_size) if t != bos_token_id]
    rows = []

    for offset in range(2, n_ctx - bos_offset):
        q_values = list(range(bos_offset + offset, n_ctx))
        for sample_idx in range(n_per_offset):
            q = q_values[sample_idx % len(q_values)]
            prev = q - offset
            perm = torch.randperm(len(content_tokens), generator=rng).tolist()
            a = content_tokens[perm[0]]
            b = content_tokens[perm[1]]
            filler_pool = [t for t in content_tokens if t not in (a, b)]

            seq = torch.empty(n_ctx, dtype=torch.long)
            if bos_token_id is not None:
                seq[0] = bos_token_id

            filler_idx = 0
            for pos in range(bos_offset, n_ctx):
                if pos in (prev, prev + 1, q):
                    continue
                seq[pos] = filler_pool[filler_idx]
                filler_idx += 1

            seq[prev] = a
            seq[prev + 1] = b
            seq[q] = a
            rows.append(seq)

    return torch.stack(rows)


def score_format(model, cfg: Dict, format_type: str, n_samples: int,
                 bos_token_id: int | None) -> list[dict]:
    tokens = _tokens_for_format(cfg, format_type, n_samples, bos_token_id)
    targets = _previous_same_targets(tokens, bos_token_id)
    prev_offsets = _offset_summary(targets, "prev_same_offset")
    next_offsets = _offset_summary(targets, "next_after_prev_offset")

    capture = AttentionCapture(model, cfg["model"]["attn_type"])
    capture.register_hooks()
    with torch.no_grad():
        _ = model(tokens)
    capture.remove_hooks()

    rows = []
    for (layer, head, circuit), pattern in sorted(capture.patterns.items()):
        if circuit != "combined":
            continue
        prev = _score_target(pattern, targets, "prev_same")
        nxt = _score_target(pattern, targets, "next_after_prev")
        prev_non_dominant = _score_target(
            pattern,
            [t for t in targets if t["prev_same_offset"] != prev_offsets["dominant_offset"]],
            "prev_same",
        )
        next_non_dominant = _score_target(
            pattern,
            [t for t in targets if t["next_after_prev_offset"] != next_offsets["dominant_offset"]],
            "next_after_prev",
        )
        pos1 = _score_target(pattern, targets, "prev_pos_1")
        pos2 = _score_target(pattern, targets, "prev_pos_2")
        pos3 = _score_target(pattern, targets, "prev_pos_3")
        pos4 = _score_target(pattern, targets, "prev_pos_4")
        rows.append(
            {
                "format": format_type,
                "layer": layer,
                "head": head,
                "prev_n_offsets": prev_offsets["n_offsets"],
                "prev_dominant_offset": prev_offsets["dominant_offset"],
                "prev_dominant_frac": prev_offsets["dominant_frac"],
                "prev_n": prev["n"],
                "prev_mean_score": prev["mean_score"],
                "prev_mean_share": prev["mean_share"],
                "prev_rank_percentile": prev["rank_percentile"],
                "prev_top1_rate": prev["top1_rate"],
                "prev_mean_margin": prev["mean_margin"],
                "prev_non_dominant_n": prev_non_dominant["n"],
                "prev_non_dominant_share": prev_non_dominant["mean_share"],
                "prev_non_dominant_top1_rate": prev_non_dominant["top1_rate"],
                "next_n_offsets": next_offsets["n_offsets"],
                "next_dominant_offset": next_offsets["dominant_offset"],
                "next_dominant_frac": next_offsets["dominant_frac"],
                "next_n": nxt["n"],
                "next_mean_score": nxt["mean_score"],
                "next_mean_share": nxt["mean_share"],
                "next_rank_percentile": nxt["rank_percentile"],
                "next_top1_rate": nxt["top1_rate"],
                "next_mean_margin": nxt["mean_margin"],
                "next_non_dominant_n": next_non_dominant["n"],
                "next_non_dominant_share": next_non_dominant["mean_share"],
                "next_non_dominant_top1_rate": next_non_dominant["top1_rate"],
                "pos_q_minus_1_share": pos1["mean_share"],
                "pos_q_minus_1_top1_rate": pos1["top1_rate"],
                "pos_q_minus_2_share": pos2["mean_share"],
                "pos_q_minus_2_top1_rate": pos2["top1_rate"],
                "pos_q_minus_3_share": pos3["mean_share"],
                "pos_q_minus_3_top1_rate": pos3["top1_rate"],
                "pos_q_minus_4_share": pos4["mean_share"],
                "pos_q_minus_4_top1_rate": pos4["top1_rate"],
            }
        )
    return rows


def score_format_by_offset(model, cfg: Dict, format_type: str, n_samples: int,
                           bos_token_id: int | None) -> list[dict]:
    tokens = _tokens_for_format(cfg, format_type, n_samples, bos_token_id)
    targets = _previous_same_targets(tokens, bos_token_id)

    capture = AttentionCapture(model, cfg["model"]["attn_type"])
    capture.register_hooks()
    with torch.no_grad():
        _ = model(tokens)
    capture.remove_hooks()

    rows = []
    for (layer, head, circuit), pattern in sorted(capture.patterns.items()):
        if circuit != "combined":
            continue
        for role, target_name, offset_name in (
            ("prev", "prev_same", "prev_same_offset"),
            ("next", "next_after_prev", "next_after_prev_offset"),
        ):
            offsets = sorted({t[offset_name] for t in targets if t[offset_name] is not None})
            for offset in offsets:
                offset_targets = [t for t in targets if t[offset_name] == offset]
                score = _score_target(pattern, offset_targets, target_name)
                rows.append(
                    {
                        "format": format_type,
                        "layer": layer,
                        "head": head,
                        "role": role,
                        "offset": offset,
                        "n": score["n"],
                        "mean_share": score["mean_share"],
                        "top1_rate": score["top1_rate"],
                        "rank_percentile": score["rank_percentile"],
                        "mean_margin": score["mean_margin"],
                    }
                )
    return rows


def score_format_by_position(model, cfg: Dict, format_type: str, n_samples: int,
                             bos_token_id: int | None) -> list[dict]:
    """Score targets stratified by offset and absolute target/query positions."""
    tokens = _tokens_for_format(cfg, format_type, n_samples, bos_token_id)
    targets = _previous_same_targets(tokens, bos_token_id)

    capture = AttentionCapture(model, cfg["model"]["attn_type"])
    capture.register_hooks()
    with torch.no_grad():
        _ = model(tokens)
    capture.remove_hooks()

    rows = []
    for (layer, head, circuit), pattern in sorted(capture.patterns.items()):
        if circuit != "combined":
            continue
        for role, target_name, offset_name in (
            ("prev", "prev_same", "prev_same_offset"),
            ("next", "next_after_prev", "next_after_prev_offset"),
        ):
            groups = sorted({
                (t[offset_name], t[target_name], t["query"])
                for t in targets
                if t[offset_name] is not None and t[target_name] is not None
            })
            for offset, target_pos, query_pos in groups:
                group_targets = [
                    t for t in targets
                    if t[offset_name] == offset
                    and t[target_name] == target_pos
                    and t["query"] == query_pos
                ]
                score = _score_target(pattern, group_targets, target_name)
                rows.append(
                    {
                        "format": format_type,
                        "layer": layer,
                        "head": head,
                        "role": role,
                        "offset": offset,
                        "target_pos": target_pos,
                        "query_pos": query_pos,
                        "n": score["n"],
                        "mean_share": score["mean_share"],
                        "top1_rate": score["top1_rate"],
                        "rank_percentile": score["rank_percentile"],
                        "mean_margin": score["mean_margin"],
                    }
                )
    return rows


def print_summary(rows: list[dict]) -> None:
    print(
        f"{'Format':<8} {'Head':<5} "
        f"{'PrevShare':>9} {'PrevTop1':>8} {'PrevRank':>8} "
        f"{'NextShare':>9} {'NextTop1':>8} {'NextRank':>8} "
        f"{'q-1':>7} {'q-2':>7} {'q-3':>7} {'q-4':>7}"
    )
    print("-" * 116)
    for r in rows:
        head = f"L{r['layer']}H{r['head']}"
        print(
            f"{r['format']:<8} {head:<5} "
            f"{r['prev_mean_share']:>9.4f} {r['prev_top1_rate']:>8.3f} "
            f"{r['prev_rank_percentile']:>8.3f} "
            f"{r['next_mean_share']:>9.4f} {r['next_top1_rate']:>8.3f} "
            f"{r['next_rank_percentile']:>8.3f} "
            f"{r['pos_q_minus_1_share']:>7.3f} {r['pos_q_minus_2_share']:>7.3f} "
            f"{r['pos_q_minus_3_share']:>7.3f} {r['pos_q_minus_4_share']:>7.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=512)
    parser.add_argument("--format", choices=("all", *ALL_FORMAT_CHOICES), default="all")
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--output_by_offset_csv", type=str, default=None)
    parser.add_argument("--output_by_position_csv", type=str, default=None)
    args = parser.parse_args()

    _run_dir, _checkpoint_dir, config_path, checkpoint_path, step_label = _resolve_run_paths(
        args.checkpoint_dir, args.step,
    )
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

    formats = ALL_FORMAT_CHOICES if args.format == "all" else (args.format,)
    all_rows = []
    all_offset_rows = []
    all_position_rows = []
    for format_type in formats:
        all_rows.extend(score_format(model, cfg, format_type, args.n_samples, bos_token_id))
        all_offset_rows.extend(
            score_format_by_offset(model, cfg, format_type, args.n_samples, bos_token_id)
        )
        all_position_rows.extend(
            score_format_by_position(model, cfg, format_type, args.n_samples, bos_token_id)
        )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Step: {step_label}")
    print(f"Samples per format: {args.n_samples}")
    print_summary(all_rows)

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nWrote CSV: {output_csv}")

    if args.output_by_offset_csv:
        output_by_offset_csv = Path(args.output_by_offset_csv)
        output_by_offset_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_by_offset_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_offset_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_offset_rows)
        print(f"Wrote by-offset CSV: {output_by_offset_csv}")

    if args.output_by_position_csv:
        output_by_position_csv = Path(args.output_by_position_csv)
        output_by_position_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_by_position_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_position_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_position_rows)
        print(f"Wrote by-position CSV: {output_by_position_csv}")


if __name__ == "__main__":
    main()
