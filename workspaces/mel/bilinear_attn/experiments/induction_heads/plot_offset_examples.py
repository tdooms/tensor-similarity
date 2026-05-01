#!/usr/bin/env python3
"""Plot per-example attention patterns for variable-gap induction cases."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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


def find_cases(tokens: torch.Tensor, bos_token_id: int | None, offset: int, limit: int):
    cases = []
    for b in range(tokens.shape[0]):
        seq = tokens[b].tolist()
        for target_pos in range(1, tokens.shape[1]):
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
            next_after_prev = prev + 1
            if query_pos - next_after_prev != offset:
                continue
            if seq[target_pos] != seq[next_after_prev]:
                continue
            cases.append(
                {
                    "batch": b,
                    "seq": seq,
                    "query_pos": query_pos,
                    "target_pos": target_pos,
                    "prev_same": prev,
                    "next_after_prev": next_after_prev,
                    "query_token": query_tok,
                    "target_token": seq[target_pos],
                }
            )
            if len(cases) >= limit:
                return cases
    return cases


def annotate_sequence(case: dict) -> str:
    parts = []
    for i, tok in enumerate(case["seq"]):
        label = str(tok)
        marks = []
        if i == case["prev_same"]:
            marks.append("prev")
        if i == case["next_after_prev"]:
            marks.append("next")
        if i == case["query_pos"]:
            marks.append("query")
        if i == case["target_pos"]:
            marks.append("target")
        if marks:
            label = f"{label}<{','.join(marks)}>"
        parts.append(label)
    return " ".join(parts)


def plot_case(patterns: dict, case: dict, n_layers: int, n_heads: int, output_path: Path):
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(4.2 * n_heads, 4.2 * n_layers))
    if n_layers == 1 and n_heads == 1:
        axes = [[axes]]
    elif n_layers == 1:
        axes = [axes]
    elif n_heads == 1:
        axes = [[ax] for ax in axes]

    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer][head]
            mat = patterns[(layer, head, "combined")][case["batch"]].numpy()
            im = ax.imshow(mat, cmap="viridis", interpolation="nearest")
            ax.scatter([case["prev_same"]], [case["query_pos"]], marker="s", s=90,
                       facecolors="none", edgecolors="white", linewidths=1.8, label="prev")
            ax.scatter([case["next_after_prev"]], [case["query_pos"]], marker="o", s=90,
                       facecolors="none", edgecolors="red", linewidths=1.8, label="next")
            ax.axhline(case["query_pos"], color="white", linewidth=0.7, alpha=0.5)
            ax.set_title(f"L{layer}H{head}")
            ax.set_xlabel("key")
            ax.set_ylabel("query")
            ax.set_xticks(range(len(case["seq"])))
            ax.set_yticks(range(len(case["seq"])))
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(annotate_sequence(case), fontsize=10)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--offset", type=int, default=5)
    parser.add_argument("--n_dataset", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--output_dir", required=True)
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
        n_samples=args.n_dataset,
        seed=43,
        bos_token_id=bos_token_id,
    )
    cases = find_cases(ds.data, bos_token_id, args.offset, args.limit)

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

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / f"offset_{args.offset}_cases_step_{step_label}.csv"
    with open(summary_path, "w", newline="") as f:
        fields = [
            "case_id", "batch", "sequence", "prev_same", "next_after_prev",
            "query_pos", "target_pos", "query_token", "target_token",
            "prediction", "correct",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, case in enumerate(cases):
            pred = int(preds[case["batch"], case["query_pos"]].item())
            case["prediction"] = pred
            case["correct"] = pred == case["target_token"]
            writer.writerow(
                {
                    "case_id": i,
                    "batch": case["batch"],
                    "sequence": annotate_sequence(case),
                    "prev_same": case["prev_same"],
                    "next_after_prev": case["next_after_prev"],
                    "query_pos": case["query_pos"],
                    "target_pos": case["target_pos"],
                    "query_token": case["query_token"],
                    "target_token": case["target_token"],
                    "prediction": pred,
                    "correct": case["correct"],
                }
            )
            plot_case(
                capture.patterns,
                case,
                cfg["model"]["n_layers"],
                cfg["model"]["n_head"],
                out / f"offset_{args.offset}_case_{i:02d}.png",
            )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Found/plotted {len(cases)} offset-{args.offset} cases")
    print(f"Summary: {summary_path}")
    for i, case in enumerate(cases):
        status = "OK" if case["correct"] else "MISS"
        print(
            f"{i:02d} {status} pred={case['prediction']} target={case['target_token']} "
            f"q={case['query_pos']} prev={case['prev_same']} next={case['next_after_prev']} "
            f"| {annotate_sequence(case)}"
        )


if __name__ == "__main__":
    main()
