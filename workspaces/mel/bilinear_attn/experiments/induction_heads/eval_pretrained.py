#!/usr/bin/env python3
"""Evaluate a pretrained SimpleStories LLaMA model on repeated-token (induction) data.

Generates sequences of the form [random_half | random_half] using the model's
own vocabulary, then measures next-token accuracy and loss on the second
(repeated) half only.

Usage (from the bilinear_attn directory):
    python -m experiments.induction_heads.eval_pretrained
    python -m experiments.induction_heads.eval_pretrained --model-size 1.25M --seq-len 50 --n-samples 1000
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaForCausalLM


def generate_repeated_tokens(
    vocab_size: int,
    seq_len: int,
    batch: int,
    seed: int = 42,
) -> torch.Tensor:
    """Generate repeated-token sequences from a uniform distribution.

    Returns:
        Tensor of shape (batch, 2 * seq_len) where the second half
        copies the first half exactly.
    """
    gen = torch.Generator().manual_seed(seed)
    half = torch.randint(0, vocab_size, (batch, seq_len), generator=gen)
    return torch.cat([half, half], dim=-1)


@torch.no_grad()
def evaluate_induction(
    model,
    input_ids: torch.Tensor,
    seq_len: int,
) -> dict:
    """Run the model on repeated-token data and measure second-half metrics.

    Args:
        model: A causal LM (e.g. LlamaForCausalLM).
        input_ids: (B, 2*seq_len) token ids.
        seq_len: Length of each half.

    Returns:
        Dict with loss, accuracy, and per-position accuracy.
    """
    outputs = model(input_ids)
    logits = outputs.logits  # (B, T, V)

    # Predictions for the repeated half: positions seq_len .. 2*seq_len-1
    # The prediction for position t is logits[:, t-1, :]
    pred_logits = logits[:, seq_len - 1 : 2 * seq_len - 1, :]  # (B, seq_len, V)
    targets = input_ids[:, seq_len : 2 * seq_len]                # (B, seq_len)

    B, T, V = pred_logits.shape

    # Loss
    loss = torch.nn.functional.cross_entropy(
        pred_logits.reshape(B * T, V),
        targets.reshape(B * T),
    ).item()

    # Accuracy
    preds = pred_logits.argmax(dim=-1)  # (B, seq_len)
    correct = (preds == targets).float()
    accuracy = correct.mean().item()

    # Per-position accuracy (averaged over batch)
    per_pos_acc = correct.mean(dim=0).tolist()  # list of length seq_len

    return {
        "loss": loss,
        "accuracy": accuracy,
        "per_position_accuracy": per_pos_acc,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained SimpleStories LLaMA on induction data"
    )
    parser.add_argument("--model-size", type=str, default="1.25M",
                        help="SimpleStories model size (default: 1.25M)")
    parser.add_argument("--seq-len", type=int, default=50,
                        help="Half-sequence length (default: 50)")
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Number of test sequences (default: 1000)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for evaluation (default: 64)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── load model ───────────────────────────────────────────────────────
    model_path = f"SimpleStories/SimpleStories-V2-{args.model_size}"
    print(f"Loading tokenizer and model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()

    vocab_size = model.config.vocab_size
    print(f"Vocab size: {vocab_size}")

    # ── generate data ────────────────────────────────────────────────────
    print(f"Generating {args.n_samples} repeated-token sequences "
          f"(half={args.seq_len}, full={2 * args.seq_len}) ...")
    all_ids = generate_repeated_tokens(
        vocab_size=vocab_size,
        seq_len=args.seq_len,
        batch=args.n_samples,
        seed=args.seed,
    ).to(device)

    # ── evaluate in batches ──────────────────────────────────────────────
    total_loss = 0.0
    total_acc = 0.0
    per_pos_acc_sum = torch.zeros(args.seq_len)
    n_batches = 0

    bs = args.batch_size
    for start in range(0, args.n_samples, bs):
        end = min(start + bs, args.n_samples)
        batch_ids = all_ids[start:end]

        result = evaluate_induction(model, batch_ids, args.seq_len)
        total_loss += result["loss"]
        total_acc += result["accuracy"]
        per_pos_acc_sum += torch.tensor(result["per_position_accuracy"])
        n_batches += 1

    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches
    avg_per_pos = (per_pos_acc_sum / n_batches).tolist()

    print(f"\n{'=' * 50}")
    print(f"Model:       {model_path}")
    print(f"Seq len:     {args.seq_len} (full: {2 * args.seq_len})")
    print(f"N samples:   {args.n_samples}")
    print(f"{'=' * 50}")
    print(f"Second-half loss:     {avg_loss:.4f}")
    print(f"Second-half accuracy: {avg_acc:.4f}")
    print(f"{'=' * 50}")

    # ── save results ─────────────────────────────────────────────────────
    run_dir = Path("experiments/induction_heads/runs")
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_file = run_dir / f"{timestamp}_pretrained_{args.model_size}.json"

    results = {
        "model": model_path,
        "model_size": args.model_size,
        "vocab_size": vocab_size,
        "seq_len": args.seq_len,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "second_half_loss": avg_loss,
        "second_half_accuracy": avg_acc,
        "per_position_accuracy": avg_per_pos,
    }
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
