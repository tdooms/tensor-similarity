#!/usr/bin/env python3
"""Evaluate a pretrained SimpleStories LLaMA on repeated *real* stories.

Each sample is a story tokenized and then concatenated with itself:
    [story_tokens | story_tokens]
We measure next-token accuracy on the second (repeated) half only.
Stories use their full tokenized length (variable per sample).

Usage (from the bilinear_attn directory):
    python -m experiments.induction_heads.eval_pretrained_stories
    python -m experiments.induction_heads.eval_pretrained_stories --model-size 5M --n-samples 200
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaForCausalLM


def tokenize_and_repeat_stories(
    dataset,
    tokenizer,
    n_samples: int,
    max_seq_len: int | None = None,
    start_idx: int = 0,
) -> list[torch.Tensor]:
    """Tokenize stories and repeat each one: [story | story].

    Args:
        dataset: HuggingFace dataset with a 'story' field.
        tokenizer: Tokenizer to use.
        n_samples: Number of samples to collect.
        max_seq_len: If set, cap each half to this many tokens (to stay
                     within model context length). None = use full story.
        start_idx: Index to start from in the dataset.

    Returns:
        List of tensors, each of shape (2 * story_len,).
    """
    sequences = []
    idx = start_idx
    while len(sequences) < n_samples and idx < len(dataset):
        text = dataset[idx]["story"]
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) < 2:
            idx += 1
            continue
        if max_seq_len is not None:
            ids = ids[:max_seq_len]
        half = torch.tensor(ids, dtype=torch.long)
        sequences.append(torch.cat([half, half], dim=0))
        idx += 1

    if not sequences:
        raise RuntimeError("No valid stories found")

    return sequences


@torch.no_grad()
def evaluate_single(model, input_ids_1d):
    """Evaluate a single repeated-story sequence.

    Args:
        model: Causal LM.
        input_ids_1d: 1-D tensor of shape (2 * story_len,).

    Returns:
        Dict with first/second half loss, accuracy, preds, targets.
    """
    seq_len = input_ids_1d.shape[0] // 2
    input_ids = input_ids_1d.unsqueeze(0)  # (1, 2*seq_len)
    logits = model(input_ids).logits       # (1, 2*seq_len, V)
    V = logits.shape[-1]

    # ── first half: predicting positions 1..seq_len-1 ──
    first_pred_logits = logits[0, 0 : seq_len - 1, :]   # (seq_len-1, V)
    first_targets = input_ids_1d[1 : seq_len]             # (seq_len-1,)
    first_loss = torch.nn.functional.cross_entropy(
        first_pred_logits, first_targets
    ).item()
    first_preds = first_pred_logits.argmax(dim=-1)
    first_acc = (first_preds == first_targets).float().mean().item()

    # ── second half: predicting positions seq_len..2*seq_len-1 ──
    second_pred_logits = logits[0, seq_len - 1 : 2 * seq_len - 1, :]  # (seq_len, V)
    second_targets = input_ids_1d[seq_len : 2 * seq_len]               # (seq_len,)
    second_loss = torch.nn.functional.cross_entropy(
        second_pred_logits, second_targets
    ).item()
    second_preds = second_pred_logits.argmax(dim=-1)
    second_acc = (second_preds == second_targets).float().mean().item()

    return {
        "seq_len": seq_len,
        "first_loss": first_loss,
        "first_accuracy": first_acc,
        "second_loss": second_loss,
        "second_accuracy": second_acc,
        "second_preds": second_preds,
        "second_targets": second_targets,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained LLaMA on repeated SimpleStories text"
    )
    parser.add_argument("--model-size", type=str, default="5M")
    parser.add_argument("--max-seq-len", type=int, default=None,
                        help="Cap each half to this many tokens (default: no cap, use full story)")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--n-print", type=int, default=5,
                        help="Number of examples to print (default: 5)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── load model ───────────────────────────────────────────────────────
    model_path = f"SimpleStories/SimpleStories-V2-{args.model_size}"
    print(f"Loading {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path).to(device).eval()
    print(f"Vocab size: {model.config.vocab_size}")

    # ── load stories ─────────────────────────────────────────────────────
    print("Loading SimpleStories dataset ...")
    ds = load_dataset("SimpleStories/SimpleStories", split="test")

    cap_str = str(args.max_seq_len) if args.max_seq_len else "full"
    print(f"Tokenizing {args.n_samples} stories (half length: {cap_str}) ...")
    sequences = tokenize_and_repeat_stories(
        ds, tokenizer, args.n_samples, max_seq_len=args.max_seq_len
    )
    actual_n = len(sequences)
    lengths = [s.shape[0] // 2 for s in sequences]
    print(f"Got {actual_n} sequences  "
          f"(story lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={sum(lengths)/len(lengths):.0f})")

    # ── print examples ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"First {args.n_print} examples:")
    print(f"{'=' * 60}")

    for i in range(min(args.n_print, actual_n)):
        seq = sequences[i].to(device)
        result = evaluate_single(model, seq)
        sl = result["seq_len"]

        first_half = seq[:sl].tolist()
        second_half = seq[sl:].tolist()
        predicted = result["second_preds"].tolist()

        first_text = tokenizer.decode(first_half)
        pred_text = tokenizer.decode(predicted)

        matches = sum(1 for p, t in zip(predicted, second_half) if p == t)

        print(f"\n--- Sample {i}  (story_len={sl} tokens) ---")
        print(f"  Story text:     {first_text!r}")
        print(f"  Predicted text: {pred_text!r}")
        print(f"  Token accuracy: {matches}/{sl}")
        print(f"  1st half ids:   {first_half[:15]}{'...' if sl > 15 else ''}")
        print(f"  Target ids:     {second_half[:15]}{'...' if sl > 15 else ''}")
        print(f"  Predicted ids:  {predicted[:15]}{'...' if sl > 15 else ''}")

    # ── full evaluation ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Evaluating on all {actual_n} samples ...")

    total_first_loss = 0.0
    total_first_acc = 0.0
    total_second_loss = 0.0
    total_second_acc = 0.0

    from tqdm import tqdm
    for seq in tqdm(sequences, desc="Evaluating"):
        seq = seq.to(device)
        result = evaluate_single(model, seq)
        total_first_loss += result["first_loss"]
        total_first_acc += result["first_accuracy"]
        total_second_loss += result["second_loss"]
        total_second_acc += result["second_accuracy"]

    avg_first_loss = total_first_loss / actual_n
    avg_first_acc = total_first_acc / actual_n
    avg_second_loss = total_second_loss / actual_n
    avg_second_acc = total_second_acc / actual_n

    delta_loss = avg_second_loss - avg_first_loss
    delta_acc = avg_second_acc - avg_first_acc

    print(f"{'=' * 60}")
    print(f"Model:                {model_path}")
    print(f"N samples:            {actual_n}")
    print(f"Story lengths:        min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.0f}")
    print(f"{'─' * 60}")
    print(f"First-half  loss:     {avg_first_loss:.4f}")
    print(f"Second-half loss:     {avg_second_loss:.4f}")
    print(f"Delta loss (2nd-1st): {delta_loss:+.4f}")
    print(f"{'─' * 60}")
    print(f"First-half  accuracy: {avg_first_acc:.4f}")
    print(f"Second-half accuracy: {avg_second_acc:.4f}")
    print(f"Delta acc  (2nd-1st): {delta_acc:+.4f}")
    print(f"{'=' * 60}")

    # ── save ─────────────────────────────────────────────────────────────
    run_dir = Path("experiments/induction_heads/runs")
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_file = run_dir / f"{timestamp}_pretrained_stories_{args.model_size}.json"

    results = {
        "model": model_path,
        "model_size": args.model_size,
        "max_seq_len": args.max_seq_len,
        "n_samples": actual_n,
        "story_len_min": min(lengths),
        "story_len_max": max(lengths),
        "story_len_mean": sum(lengths) / len(lengths),
        "first_half_loss": avg_first_loss,
        "first_half_accuracy": avg_first_acc,
        "second_half_loss": avg_second_loss,
        "second_half_accuracy": avg_second_acc,
        "delta_loss": delta_loss,
        "delta_accuracy": delta_acc,
    }
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
