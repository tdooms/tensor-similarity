"""Induction head test set and evaluation.

Induction heads learn the pattern: [A] [B] ... [A] → predict [B]
i.e., "if I've seen A followed by B before, and now I see A again, predict B."

We create synthetic sequences with repeated bigrams and measure whether the model
assigns higher probability to the correct induction target vs a random baseline.
"""

import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from lib import AttentionLM


def make_induction_sequences(
    vocab_size: int = 4096,
    n_ctx: int = 512,
    n_sequences: int = 100,
    prefix_len: int = 50,
    repeat_len: int = 20,
    seed: int = 42,
) -> dict:
    """Generate sequences that test for induction heads.

    Each sequence has the structure:
        [random prefix] [A1 B1 A2 B2 ... An Bn] [random filler] [A1 B1 A2 B2 ... An ?]

    The model should predict Bn at the ? position (induction).

    Returns dict with:
        input_ids: (N, seq_len) tensor
        target_positions: (N,) tensor — index of the position where induction should fire
        target_tokens: (N,) tensor — the correct induction target at each position
        trigger_tokens: (N,) tensor — the trigger token A that precedes the target
    """
    rng = np.random.RandomState(seed)
    n_bigrams = repeat_len // 2

    all_input_ids = []
    all_target_pos = []
    all_target_tok = []
    all_trigger_tok = []

    for _ in range(n_sequences):
        # Random prefix
        prefix = rng.randint(1, vocab_size, size=prefix_len)

        # Generate n_bigrams unique bigrams [A_i, B_i]
        bigram_tokens = rng.randint(1, vocab_size, size=(n_bigrams, 2))

        # First occurrence of bigrams
        first_occurrence = bigram_tokens.flatten()  # [A1, B1, A2, B2, ...]

        # Random filler between first and second occurrence
        filler_len = n_ctx - prefix_len - len(first_occurrence) * 2 - 1
        if filler_len < 10:
            filler_len = 10
        filler = rng.randint(1, vocab_size, size=filler_len)

        # Second occurrence: all bigrams except the last B (which is the target)
        second_occurrence = bigram_tokens.flatten()[:-1]  # [A1, B1, ..., An] (no final Bn)

        # Build full sequence
        seq = np.concatenate([prefix, first_occurrence, filler, second_occurrence])

        # Truncate or pad to n_ctx
        if len(seq) > n_ctx:
            # Trim filler to fit
            excess = len(seq) - n_ctx
            filler = filler[:-excess] if excess < len(filler) else filler[:10]
            seq = np.concatenate([prefix, first_occurrence, filler, second_occurrence])

        if len(seq) < n_ctx:
            pad = rng.randint(1, vocab_size, size=n_ctx - len(seq))
            seq = np.concatenate([seq, pad])

        seq = seq[:n_ctx]

        # The target position is the last position in the sequence
        # (where we want to predict the last B_n)
        target_pos = len(seq) - 1
        target_tok = bigram_tokens[-1, 1]  # B_n
        trigger_tok = bigram_tokens[-1, 0]  # A_n

        all_input_ids.append(torch.tensor(seq, dtype=torch.long))
        all_target_pos.append(target_pos)
        all_target_tok.append(target_tok)
        all_trigger_tok.append(trigger_tok)

    return {
        "input_ids": torch.stack(all_input_ids),
        "target_positions": torch.tensor(all_target_pos, dtype=torch.long),
        "target_tokens": torch.tensor(all_target_tok, dtype=torch.long),
        "trigger_tokens": torch.tensor(all_trigger_tok, dtype=torch.long),
    }


@torch.no_grad()
def evaluate_induction(
    model: torch.nn.Module,
    test_data: dict,
    device: str = "cuda",
    batch_size: int = 32,
) -> dict:
    """Evaluate induction head performance.

    Returns:
        Dict with metrics:
            - induction_accuracy: fraction where argmax prediction == target
            - induction_loss: average CE loss at target positions
            - mean_target_rank: average rank of the target token
            - mean_target_prob: average probability assigned to target
            - baseline_loss: loss if predicting uniform over vocab
    """
    model.eval()
    input_ids = test_data["input_ids"].to(device)
    target_pos = test_data["target_positions"].to(device)
    target_tok = test_data["target_tokens"].to(device)

    n = input_ids.shape[0]
    all_correct = []
    all_loss = []
    all_rank = []
    all_prob = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_ids = input_ids[start:end]
        batch_pos = target_pos[start:end]
        batch_tok = target_tok[start:end]

        logits = model(batch_ids)  # (B, T, V)

        # Extract logits at target positions
        B = logits.shape[0]
        target_logits = logits[torch.arange(B, device=device), batch_pos]  # (B, V)

        # Accuracy
        preds = target_logits.argmax(dim=-1)
        all_correct.append((preds == batch_tok).float())

        # Loss
        loss = F.cross_entropy(target_logits, batch_tok, reduction="none")
        all_loss.append(loss)

        # Rank of target token
        sorted_indices = target_logits.argsort(dim=-1, descending=True)
        ranks = (sorted_indices == batch_tok.unsqueeze(-1)).nonzero(as_tuple=True)[1]
        all_rank.append(ranks.float())

        # Probability of target
        probs = F.softmax(target_logits, dim=-1)
        target_probs = probs[torch.arange(B, device=device), batch_tok]
        all_prob.append(target_probs)

    vocab_size = logits.shape[-1]
    return {
        "induction_accuracy": torch.cat(all_correct).mean().item(),
        "induction_loss": torch.cat(all_loss).mean().item(),
        "mean_target_rank": torch.cat(all_rank).mean().item(),
        "mean_target_prob": torch.cat(all_prob).mean().item(),
        "baseline_loss": np.log(vocab_size),
        "n_sequences": n,
    }


def main():
    parser = argparse.ArgumentParser(description="Induction head test")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n-sequences", type=int, default=200)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = AttentionLM.from_config(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint (step {ckpt['step']})")

    vocab_size = cfg["model"]["vocab_size"]
    n_ctx = cfg["model"]["n_ctx"]

    print(f"Generating {args.n_sequences} induction test sequences...")
    test_data = make_induction_sequences(
        vocab_size=vocab_size,
        n_ctx=n_ctx,
        n_sequences=args.n_sequences,
    )

    print("Evaluating induction heads...")
    results = evaluate_induction(model, test_data, device=device)

    print(f"\n{'='*40}")
    print(f"Induction Head Test Results")
    print(f"{'='*40}")
    print(f"Accuracy (argmax = target):  {results['induction_accuracy']:.1%}")
    print(f"Mean target probability:     {results['mean_target_prob']:.4f}")
    print(f"Mean target rank:            {results['mean_target_rank']:.1f} / {vocab_size}")
    print(f"Induction CE loss:           {results['induction_loss']:.3f}")
    print(f"Uniform baseline CE loss:    {results['baseline_loss']:.3f}")
    print(f"N sequences:                 {results['n_sequences']}")


if __name__ == "__main__":
    main()
