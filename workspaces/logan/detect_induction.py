#!/usr/bin/env python3
"""Detect induction heads by measuring per-head attention on the induction diagonal.

For repeated sequences [s₀...s_{L-1} s₀...s_{L-1}], an induction head at position
L+k attends to position k+1 (the token that followed the previous occurrence).
"""
import torch
import numpy as np
from lib import AttentionLM


@torch.no_grad()
def compute_per_head_induction_scores(
    model,
    vocab_size: int = 32,
    half_len: int = 16,
    n_sequences: int = 200,
    device: str = "cuda",
    seed: int = 42,
):
    """Compute per-head induction scores on repeated random sequences.

    Args:
        model: AttentionLM model (already on device).
        vocab_size: Token vocabulary size.
        half_len: Length of each half-sequence (L).
        n_sequences: Number of random sequences to average over.
        device: Device string.
        seed: Random seed for reproducibility.

    Returns:
        List of dicts, one per layer. Each dict maps head_idx to:
            {"raw": float, "normalized": float}
    """
    model.eval()
    rng = np.random.RandomState(seed)

    # Generate repeated sequences: [s, s]
    seqs_np = rng.randint(1, vocab_size, size=(n_sequences, half_len))
    seqs_np = np.concatenate([seqs_np, seqs_np], axis=1)  # (N, 2L)
    seqs = torch.tensor(seqs_np, dtype=torch.long, device=device)

    L = half_len

    # Forward with debug to get per-head attention patterns
    logits, debug_list = model(seqs, return_debug=True)

    results = []
    for layer_idx, debug in enumerate(debug_list):
        pattern = debug["pattern"]  # (B, n_head, 2L, 2L)
        n_head = pattern.shape[1]

        # Build induction diagonal indices:
        # At position L+k, induction target is position k+1, for k = 0..L-2
        query_positions = torch.arange(L, 2 * L - 1, device=device)  # L, L+1, ..., 2L-2
        key_positions = torch.arange(1, L, device=device)             # 1, 2, ..., L-1

        # Extract attention on induction diagonal: (B, n_head, L-1)
        induction_attn = pattern[:, :, query_positions, key_positions]

        # Normalize pattern per row for normalized score
        # Use abs sum for polynomial attention (patterns can be negative)
        row_abs_sums = pattern[:, :, query_positions, :].abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
        normalized_induction_attn = induction_attn / row_abs_sums.squeeze(-1)

        layer_results = {}
        for h in range(n_head):
            raw_score = induction_attn[:, h, :].mean().item()
            norm_score = normalized_induction_attn[:, h, :].mean().item()
            layer_results[h] = {"raw": raw_score, "normalized": norm_score}

        results.append(layer_results)

    return results


def is_induction_head(normalized_score: float, threshold: float = 0.2) -> bool:
    """Classify a head as inductiony based on its normalized induction score."""
    return normalized_score > threshold


def print_induction_report(results, threshold=0.3):
    """Print a formatted table of per-head induction scores."""
    print(f"\n{'Layer':>5} {'Head':>4} {'Raw':>10} {'Normalized':>10} {'Induction?':>10}")
    print("-" * 45)
    for layer_idx, layer_results in enumerate(results):
        for head_idx in sorted(layer_results.keys()):
            scores = layer_results[head_idx]
            is_ind = is_induction_head(scores["normalized"], threshold)
            marker = "  YES" if is_ind else ""
            print(f"  L{layer_idx:>2}   H{head_idx:>2}  {scores['raw']:>10.4f} {scores['normalized']:>10.4f} {marker:>10}")


def load_toy_model(checkpoint_path, attn_type, n_head=2, n_layers=2, d_model=128,
                   vocab_size=32, half_len=16, use_rmsnorm_qk=False, device="cuda"):
    """Load a toy induction model from checkpoint."""
    n_ctx = 2 * half_len
    d_model = max(d_model, n_head * 64)
    model = AttentionLM(
        vocab_size=vocab_size,
        n_ctx=n_ctx,
        d_model=d_model,
        n_head=n_head,
        n_layers=n_layers,
        attn_scale=1.0,
        attn_type=attn_type,
        norm_type="layernorm",
        norm_place="pre_unembed",
        use_rmsnorm_qk=use_rmsnorm_qk,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect induction heads in toy models")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to final.pt")
    parser.add_argument("--attn-type", type=str, required=True, choices=["quadratic", "bilinear", "softmax"])
    parser.add_argument("--qk-norm", action="store_true")
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=32)
    parser.add_argument("--half-len", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--n-sequences", type=int, default=200)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {args.attn_type} model (qk_norm={args.qk_norm}) from {args.checkpoint}")
    model = load_toy_model(
        args.checkpoint,
        attn_type=args.attn_type,
        n_head=args.n_head,
        n_layers=args.n_layers,
        d_model=args.d_model,
        vocab_size=args.vocab_size,
        half_len=args.half_len,
        use_rmsnorm_qk=args.qk_norm,
        device=device,
    )

    print(f"Computing induction scores on {args.n_sequences} repeated sequences (half_len={args.half_len})...")
    results = compute_per_head_induction_scores(
        model,
        vocab_size=args.vocab_size,
        half_len=args.half_len,
        n_sequences=args.n_sequences,
        device=device,
    )

    print_induction_report(results, threshold=args.threshold)
