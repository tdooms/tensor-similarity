#!/usr/bin/env python3
"""Analyze correct vs incorrect supervised induction targets by offset."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.induction_heads.analyze_sequence_formats import AttentionCapture  # noqa: E402
from experiments.induction_heads.data import RepeatedTokenDataset  # noqa: E402
from models import AttentionLM  # noqa: E402


def resolve_run_paths(checkpoint_dir_arg: str, step: int | None):
    checkpoint_dir = Path(__file__).parent / checkpoint_dir_arg
    if checkpoint_dir.name == "checkpoints":
        run_dir = checkpoint_dir.parent
    else:
        run_dir = checkpoint_dir
        checkpoint_dir = run_dir / "checkpoints"
    config_path = run_dir / "config.yaml"
    checkpoint_path = checkpoint_dir / f"step_{step}.pt" if step is not None else run_dir / "final.pt"
    return config_path, checkpoint_path, str(step) if step is not None else "final"


def supervised_cases(tokens: torch.Tensor, mask: torch.Tensor, bos_token_id: int | None, offset: int):
    cases = []
    for b, seq_t in enumerate(tokens):
        seq = seq_t.tolist()
        for target_pos in range(1, len(seq)):
            if not bool(mask[b, target_pos]):
                continue
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
            next_key = prev + 1
            if query_pos - next_key != offset:
                continue
            cases.append(
                {
                    "batch": b,
                    "seq": seq,
                    "prev_same": prev,
                    "next_key": next_key,
                    "query_pos": query_pos,
                    "target_pos": target_pos,
                    "target_token": seq[target_pos],
                    "query_token": query_tok,
                }
            )
    return cases


def summarize_head(pattern: torch.Tensor, cases: list[dict], correct_flags: list[bool]):
    rows = []
    for label, want_correct in (("correct", True), ("incorrect", False), ("all", None)):
        selected = [
            case for case, ok in zip(cases, correct_flags)
            if want_correct is None or ok == want_correct
        ]
        if not selected:
            continue

        key_shares = []
        key_ranks = []
        key_top1 = []
        q_minus_1 = []
        prev_same = []
        argmaxes = []
        for case in selected:
            b = case["batch"]
            q = case["query_pos"]
            row = pattern[b, q, :q].float()
            row_sum = row.sum().item()
            key = case["next_key"]
            prev = case["prev_same"]
            val = row[key]
            key_shares.append(val.item() / row_sum if row_sum > 0 else 0.0)
            key_ranks.append((row <= val).float().mean().item())
            key_top1.append(1.0 if bool(val >= row.max()) else 0.0)
            q_minus_1.append(row[q - 1].item() / row_sum if row_sum > 0 else 0.0)
            prev_same.append(row[prev].item() / row_sum if row_sum > 0 else 0.0)
            argmaxes.append(int(torch.argmax(row).item()))

        def mean(xs):
            return sum(xs) / len(xs) if xs else 0.0

        counts = Counter(argmaxes)
        rows.append(
            {
                "split": label,
                "n": len(selected),
                "next_key_share": mean(key_shares),
                "next_key_rank": mean(key_ranks),
                "next_key_top1": mean(key_top1),
                "q_minus_1_share": mean(q_minus_1),
                "prev_same_share": mean(prev_same),
                "argmax_counts": " ".join(f"{k}:{v}" for k, v in sorted(counts.items())),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--offset", type=int, default=5)
    parser.add_argument("--n_samples", type=int, default=4096)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    config_path, checkpoint_path, step_label = resolve_run_paths(args.checkpoint_dir, args.step)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    bos_token_id = None
    if cfg.get("data", {}).get("use_bos", False):
        bos_token_id = cfg.get("data", {}).get("bos_token_id", cfg["model"]["vocab_size"] - 1)

    ds = RepeatedTokenDataset(
        vocab_size=cfg["model"]["vocab_size"],
        n_ctx=cfg["model"]["n_ctx"],
        n_samples=args.n_samples,
        seed=43,
        bos_token_id=bos_token_id,
    )
    cases = supervised_cases(ds.data, ds.repeat_masks, bos_token_id, args.offset)

    model = AttentionLM.from_config(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    capture = AttentionCapture(model, cfg["model"]["attn_type"])
    capture.register_hooks()
    with torch.no_grad():
        logits = model(ds.data)
    capture.remove_hooks()
    preds = logits.argmax(dim=-1)

    correct_flags = [
        int(preds[c["batch"], c["query_pos"]].item()) == c["target_token"]
        for c in cases
    ]

    rows = []
    for (layer, head, circuit), pattern in sorted(capture.patterns.items()):
        if circuit != "combined":
            continue
        for row in summarize_head(pattern, cases, correct_flags):
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "offset": args.offset,
                    **row,
                }
            )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total = len(cases)
    correct = sum(correct_flags)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Offset {args.offset}: {correct}/{total} correct = {correct / total if total else 0:.3f}")
    print(
        f"{'Head':<5} {'Split':<10} {'N':>5} {'NextShare':>10} {'NextTop1':>9} "
        f"{'NextRank':>9} {'q-1':>8} {'PrevSame':>9} {'Argmax counts'}"
    )
    print("-" * 100)
    for r in rows:
        print(
            f"L{r['layer']}H{r['head']:<2} {r['split']:<10} {r['n']:>5} "
            f"{r['next_key_share']:>10.3f} {r['next_key_top1']:>9.3f} "
            f"{r['next_key_rank']:>9.3f} {r['q_minus_1_share']:>8.3f} "
            f"{r['prev_same_share']:>9.3f} {r['argmax_counts']}"
        )
    print(f"\nWrote CSV: {output_csv}")


if __name__ == "__main__":
    main()
