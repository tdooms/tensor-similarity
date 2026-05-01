#!/usr/bin/env python3
"""Split layer-0 head ablations into direct and layer-1-active paths."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from einops import rearrange, einsum

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.induction_heads.score_pattern_accuracy import data_for_format, resolve_run_paths  # noqa: E402
from experiments.path_decomp.ablation.head_ablation import canonical_shape  # noqa: E402
from models import AttentionLM  # noqa: E402


def _split_linear(linear, x, n_head):
    return rearrange(linear(x), "b t (n d) -> b t n d", n=n_head)


def _attention_z(layer, x):
    t = x.shape[1]
    if hasattr(layer, "q") and hasattr(layer, "k"):
        q = layer.rotary(layer.norm_qk(_split_linear(layer.q, x, layer.n_head)))
        k = layer.rotary(layer.norm_qk(_split_linear(layer.k, x, layer.n_head)))
        v = _split_linear(layer.v, x, layer.n_head)
        scores = einsum(q, k, "b tq n d, b tk n d -> b n tq tk")
        pattern = (scores / layer.d_head).square()
    elif hasattr(layer, "q1") and hasattr(layer, "k1"):
        q1 = layer.rotary(layer.norm_qk(_split_linear(layer.q1, x, layer.n_head)))
        k1 = layer.rotary(layer.norm_qk(_split_linear(layer.k1, x, layer.n_head)))
        q2 = layer.rotary(layer.norm_qk(_split_linear(layer.q2, x, layer.n_head)))
        k2 = layer.rotary(layer.norm_qk(_split_linear(layer.k2, x, layer.n_head)))
        v = _split_linear(layer.v, x, layer.n_head)
        scores1 = einsum(q1, k1, "b tq n d, b tk n d -> b n tq tk")
        scores2 = einsum(q2, k2, "b tq n d, b tk n d -> b n tq tk")
        pattern = (scores1 * scores2) / layer.d_head**2
    else:
        raise NotImplementedError(f"Unsupported attention layer: {type(layer).__name__}")
    pattern = pattern * layer.causal_mask[None, None, :t, :t]
    return einsum(pattern, v, "b n tq tk, b tk n d -> b tq n d")


def _active_output(layer, x):
    z = _attention_z(layer, x)
    z_merge = rearrange(z, "b t n d -> b t (n d)")
    return layer.scale * layer.o(z_merge)


def _layer0_parts(layer, x):
    z = _attention_z(layer, x)
    o_w = rearrange(layer.o.weight, "d_model (n d_head) -> n d_model d_head", n=layer.n_head)
    per_head = einsum(z, o_w, "b t n d_head, n d_model d_head -> b t n d_model")
    per_head = layer.scale * per_head
    base = (1.0 - layer.scale) * x
    return base, per_head


@torch.no_grad()
def forward_l0_path_ablation(model: AttentionLM, input_ids: torch.Tensor, head: int | None, mode: str) -> torch.Tensor:
    if model.layer_norms is not None:
        raise NotImplementedError("This path split is implemented for post-layer/residual mode, not pre_layer norm mode.")
    if model.n_layers != 2:
        raise ValueError(f"Expected 2 layers, got {model.n_layers}")

    x0 = model.embed(input_ids)
    if model.embed_norm is not None:
        x0 = model.embed_norm(x0)

    base0, l0_heads = _layer0_parts(model.layers[0], x0)
    x1_full = base0 + l0_heads.sum(dim=2)

    if head is None or mode == "baseline":
        residual_input = x1_full
        active_input = x1_full
    else:
        x1_without = x1_full - l0_heads[:, :, head, :]
        if mode == "direct":
            residual_input = x1_without
            active_input = x1_full
        elif mode == "layer2":
            residual_input = x1_full
            active_input = x1_without
        elif mode == "both":
            residual_input = x1_without
            active_input = x1_without
        else:
            raise ValueError(f"Unknown mode: {mode}")

    layer1 = model.layers[1]
    x2 = (1.0 - layer1.scale) * residual_input + _active_output(layer1, active_input)
    return model.unembed(model.final_norm(x2))


def _next_offset(seq: list[int], target_pos: int) -> int | None:
    query_pos = target_pos - 1
    query_tok = seq[query_pos]
    prev = None
    for k in range(query_pos - 1, -1, -1):
        if seq[k] == query_tok:
            prev = k
            break
    if prev is None:
        return None
    offset = query_pos - (prev + 1)
    return offset if offset >= 1 else None


def _shape_classes(tokens, mask, bos_token_id):
    offsets_by_shape = defaultdict(set)
    examples = {}
    for seq_t, mask_t in zip(tokens, mask):
        seq = seq_t.tolist()
        shape = canonical_shape(seq, bos_token_id)
        rows = []
        bos_shift = 1 if bos_token_id is not None else 0
        for target_pos in range(1, len(seq)):
            if not bool(mask_t[target_pos]):
                continue
            offset = _next_offset(seq, target_pos)
            if offset is None:
                continue
            offsets_by_shape[shape].add(offset)
            query_pos = target_pos - 1
            next_key = query_pos - offset
            rows.append((target_pos - bos_shift, query_pos - bos_shift, next_key - bos_shift, offset))
        examples.setdefault(shape, rows)
    return {
        shape: "constant" if len(offsets) == 1 else "variable"
        for shape, offsets in offsets_by_shape.items()
    }, examples


def score_by_class(tokens, mask, logits, bos_token_id, shape_class):
    preds = logits.argmax(dim=-1)
    rows = defaultdict(lambda: {"n_eval_tokens": 0, "correct": 0})
    for b in range(tokens.shape[0]):
        shape = canonical_shape(tokens[b].tolist(), bos_token_id)
        cls = shape_class[shape]
        for target_pos in range(1, tokens.shape[1]):
            if not bool(mask[b, target_pos]):
                continue
            rows[cls]["n_eval_tokens"] += 1
            rows[cls]["correct"] += int(preds[b, target_pos - 1].item() == tokens[b, target_pos].item())
    out = []
    for cls, counts in sorted(rows.items()):
        n = counts["n_eval_tokens"]
        out.append({
            "offset_class": cls,
            "n_eval_tokens": n,
            "correct": counts["correct"],
            "accuracy": counts["correct"] / n if n else 0.0,
        })
    return out


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--step", type=int, default=15000)
    parser.add_argument("--n_samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=12345)
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

    torch.manual_seed(args.seed)
    tokens, mask = data_for_format(cfg, "variable_gap", args.n_samples, bos_token_id)
    shape_class, examples = _shape_classes(tokens, mask, bos_token_id)

    rows = []
    baseline_logits = forward_l0_path_ablation(model, tokens, None, "baseline")
    baseline = score_by_class(tokens, mask, baseline_logits, bos_token_id, shape_class)
    baseline_acc = {r["offset_class"]: r["accuracy"] for r in baseline}
    for r in baseline:
        r.update({"head": "baseline", "mode": "baseline", "delta_accuracy": 0.0})
        rows.append(r)

    for head in (0, 1):
        for mode in ("direct", "layer2", "both"):
            logits = forward_l0_path_ablation(model, tokens, head, mode)
            scored = score_by_class(tokens, mask, logits, bos_token_id, shape_class)
            for r in scored:
                r.update({
                    "head": f"L0H{head}",
                    "mode": mode,
                    "delta_accuracy": r["accuracy"] - baseline_acc[r["offset_class"]],
                })
                rows.append(r)

    if args.output_dir is None:
        output_dir = ROOT / "experiments" / "path_decomp" / "ablation_runs" / f"small-big-experiment-runs_step_{step_label}_L0_paths"
    else:
        output_dir = Path(args.output_dir)
    out_csv = output_dir / "l0_path_ablation_by_offset_class.csv"
    write_csv(out_csv, rows)

    examples_csv = output_dir / "offset_class_examples.csv"
    example_rows = [
        {
            "shape": shape,
            "offset_class": shape_class[shape],
            "eval_rows_target_query_key_offset": repr(examples[shape]),
        }
        for shape in sorted(shape_class)
    ]
    write_csv(examples_csv, example_rows)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Step: {step_label}")
    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote examples: {examples_csv}")
    print(f"{'class':<10} {'head':<8} {'mode':<9} {'eval':>6} {'acc':>8} {'drop':>8}")
    print("-" * 58)
    for r in rows:
        print(
            f"{r['offset_class']:<10} {r['head']:<8} {r['mode']:<9} "
            f"{r['n_eval_tokens']:>6} {r['accuracy']:>8.3f} {r['delta_accuracy']:>8.3f}"
        )


if __name__ == "__main__":
    main()
