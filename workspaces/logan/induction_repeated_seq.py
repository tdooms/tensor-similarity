"""Canonical induction head test: repeated random sequences.

From Olsson et al. (2022) "In-context Learning and Induction Heads":
  - Generate random token sequence of length L
  - Repeat it: [seq][seq] (total length 2L)
  - Feed to model, measure loss on first half vs second half
  - A model with induction heads will have much lower loss on the 2nd half

The key insight: on the second half, the model has seen each bigram before,
so if it has induction heads it can predict every token (except the first
repeated token) by copying from the first occurrence.
"""
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from lib import AttentionLM


def make_repeated_sequences(
    vocab_size: int = 4096,
    half_len: int = 256,
    n_sequences: int = 100,
    seed: int = 42,
):
    """Generate [random_seq][random_seq] sequences."""
    rng = np.random.RandomState(seed)
    # Use tokens 1..vocab_size-1 (avoid 0 which might be pad)
    seqs = rng.randint(1, vocab_size, size=(n_sequences, half_len))
    # Repeat: [seq, seq]
    doubled = np.concatenate([seqs, seqs], axis=1)  # (N, 2*half_len)
    return torch.tensor(doubled, dtype=torch.long), half_len


@torch.no_grad()
def evaluate_repeated_seq(
    model: torch.nn.Module,
    sequences: torch.Tensor,
    half_len: int,
    device: str = "cuda",
    batch_size: int = 32,
):
    """Evaluate per-position loss on repeated sequences."""
    model.eval()
    n = sequences.shape[0]
    total_len = sequences.shape[1]

    # Collect per-position losses
    all_losses = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = sequences[start:end].to(device)

        logits = model(batch)  # (B, T, V)

        # Per-position CE loss (no reduction)
        # logits[:, t, :] predicts token at position t+1
        shift_logits = logits[:, :-1, :]
        shift_labels = batch[:, 1:]
        B, T, V = shift_logits.shape
        loss = F.cross_entropy(
            shift_logits.reshape(B * T, V),
            shift_labels.reshape(B * T),
            reduction="none",
        ).reshape(B, T)

        all_losses.append(loss.cpu())

    all_losses = torch.cat(all_losses, dim=0)  # (N, 2*half_len - 1)
    mean_loss = all_losses.mean(dim=0).numpy()  # (2*half_len - 1,)

    # Split into first half and second half
    # Position 0..half_len-2 = first half predictions
    # Position half_len-1..2*half_len-2 = second half predictions
    first_half_loss = mean_loss[:half_len - 1].mean()
    second_half_loss = mean_loss[half_len:].mean()  # skip the boundary token
    boundary_loss = mean_loss[half_len - 1]  # predicting first token of repeat

    return {
        "per_position_loss": mean_loss,
        "first_half_loss": float(first_half_loss),
        "second_half_loss": float(second_half_loss),
        "boundary_loss": float(boundary_loss),
        "induction_score": float(first_half_loss - second_half_loss),
        "half_len": half_len,
        "uniform_baseline": float(np.log(sequences.max().item())),
    }


def main():
    parser = argparse.ArgumentParser(description="Repeated sequence induction test")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n-sequences", type=int, default=200)
    parser.add_argument("--half-len", type=int, default=256)
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

    print(f"Generating {args.n_sequences} repeated random sequences (half_len={args.half_len})...")
    sequences, half_len = make_repeated_sequences(
        vocab_size=vocab_size,
        half_len=args.half_len,
        n_sequences=args.n_sequences,
    )
    print(f"Sequence shape: {sequences.shape}")

    print("Evaluating...")
    results = evaluate_repeated_seq(model, sequences, half_len, device=device)

    print(f"\n{'='*50}")
    print(f"Repeated Sequence Induction Test")
    print(f"{'='*50}")
    print(f"First half mean loss:    {results['first_half_loss']:.4f}")
    print(f"Second half mean loss:   {results['second_half_loss']:.4f}")
    print(f"Boundary token loss:     {results['boundary_loss']:.4f}")
    print(f"Induction score:         {results['induction_score']:.4f}")
    print(f"  (positive = model predicts 2nd half better)")
    print(f"Uniform baseline:        {results['uniform_baseline']:.4f}")


if __name__ == "__main__":
    main()
