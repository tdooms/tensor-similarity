#!/usr/bin/env python3
"""Robust induction evaluation across checkpoints with plotting.

Two tests:
1. In-distribution: 100 real sequences from DSIR Pile val, measure loss at first vs
   second occurrence of repeated bigrams.
2. Toy repeated: 100 samples of [16 random tokens][16 random tokens], measure loss
   on first half vs second half.

For both, the key metric is the average loss *difference* (first - second occurrence).
Positive = model predicts better the second time = induction signal.

Usage:
    python eval_induction_robust.py --run-dir runs/2026-02-13_192817_hoogland_replication \
        --d-model 256 --n-head 8 --n-ctx 1024
"""
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, einsum
from pathlib import Path
from collections import defaultdict

# =============================================================================
# Model (must match training scripts)
# =============================================================================

class Rotary(nn.Module):
    def __init__(self, dim, n_ctx, base=10000):
        super().__init__()
        freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        ctx = torch.arange(n_ctx).float()
        freqs = torch.einsum("i,j->ij", ctx, freq)
        cos = freqs.cos()
        sin = freqs.sin()
        self.register_buffer("cos_cached", torch.cat([cos, cos], dim=-1)[None, :, None, :], persistent=False)
        self.register_buffer("sin_cached", torch.cat([sin, sin], dim=-1)[None, :, None, :], persistent=False)

    def forward(self, x):
        seq_len = x.size(1)
        a, b = x.chunk(2, dim=-1)
        y = torch.cat((-b, a), dim=-1)
        return (x * self.cos_cached[:, :seq_len]) + (y * self.sin_cached[:, :seq_len])


class SoftmaxAttention(nn.Module):
    def __init__(self, d_model, n_head, n_ctx, scale=1.0, use_rmsnorm_qk=False,
                 use_bias_qkv=True, use_bias_o=True, rope_base=10000):
        super().__init__()
        self.d_head = d_model // n_head
        self.n_head = n_head
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.scale = scale
        self.rotary = Rotary(self.d_head, n_ctx, base=rope_base)
        self.norm_qk = nn.RMSNorm(self.d_head) if use_rmsnorm_qk else nn.Identity()
        causal_mask = torch.triu(torch.full((n_ctx, n_ctx), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)
        self.q = nn.Linear(d_model, d_model, bias=use_bias_qkv)
        self.k = nn.Linear(d_model, d_model, bias=use_bias_qkv)
        self.v = nn.Linear(d_model, d_model, bias=use_bias_qkv)
        self.o = nn.Linear(d_model, d_model, bias=use_bias_o)

    def forward(self, x, return_debug=False):
        B, T, _ = x.shape
        q = rearrange(self.q(x), "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)
        k = rearrange(self.k(x), "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)
        v = rearrange(self.v(x), "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)
        q = self.rotary(self.norm_qk(q))
        k = self.rotary(self.norm_qk(k))
        scores = einsum(q, k, "b sq nh dh, b sk nh dh -> b nh sq sk")
        scores = scores / (self.d_head ** 0.5)
        scores = scores + self.causal_mask[None, None, :T, :T]
        pattern = torch.softmax(scores, dim=-1)
        z = einsum(pattern, v, "b nh sq sk, b sk nh dh -> b sq nh dh")
        z_merge = rearrange(z, "b seq n_head d_head -> b seq (n_head d_head)")
        out = x + self.scale * self.o(z_merge)
        return out


class AttentionLM(nn.Module):
    def __init__(self, vocab_size, n_ctx, d_model, n_head, n_layers,
                 attn_scale=1.0, rope_base=10000, use_rmsnorm_qk=False,
                 use_bias_qkv=True, use_bias_o=True, norm_type="layernorm"):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        if norm_type == "layernorm":
            self.final_norm = nn.LayerNorm(d_model)
        elif norm_type == "rmsnorm":
            self.final_norm = nn.RMSNorm(d_model)
        else:
            self.final_norm = nn.Identity()
        self.layers = nn.ModuleList([
            SoftmaxAttention(d_model=d_model, n_head=n_head, n_ctx=n_ctx,
                             scale=attn_scale, use_rmsnorm_qk=use_rmsnorm_qk,
                             use_bias_qkv=use_bias_qkv, use_bias_o=use_bias_o,
                             rope_base=rope_base)
            for _ in range(n_layers)
        ])
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.unembed(x)


# =============================================================================
# Test 1: In-distribution repeated bigrams
# =============================================================================

def find_all_repeated_bigrams(data, n_ctx):
    """For each sequence, find all positions where a bigram repeats.

    Returns list of (seq_idx, first_pos, second_pos, token_to_predict) for each
    repeated bigram occurrence. first_pos/second_pos are the position of the first
    token of the bigram.
    """
    results = []
    for seq_idx in range(data.shape[0]):
        seq = data[seq_idx, :n_ctx].numpy()
        bigram_first_seen = {}  # (tok_a, tok_b) -> first position
        for i in range(len(seq) - 1):
            bg = (int(seq[i]), int(seq[i + 1]))
            if bg in bigram_first_seen:
                first_pos = bigram_first_seen[bg]
                if i - first_pos > 2:  # not overlapping
                    results.append({
                        "seq_idx": seq_idx,
                        "first_pos": first_pos,
                        "second_pos": i,
                        "target_token": bg[1],
                        "distance": i - first_pos,
                    })
            else:
                bigram_first_seen[bg] = i
    return results


@torch.no_grad()
def eval_indistribution_bigrams(model, data, n_ctx, device, max_seqs=100):
    """Evaluate loss at first vs second occurrence of repeated bigrams."""
    model.eval()
    data_subset = data[:max_seqs, :n_ctx]

    # Find all repeated bigrams
    all_bigrams = find_all_repeated_bigrams(data_subset, n_ctx)
    if not all_bigrams:
        return {"n_bigrams": 0}

    # Get per-token losses for all sequences
    # Process in small batches
    batch_size = 16
    all_losses = []
    for start in range(0, data_subset.shape[0], batch_size):
        end = min(start + batch_size, data_subset.shape[0])
        batch = data_subset[start:end].to(device)
        logits = model(batch)
        # loss at position i = cross_entropy(logits[i], token[i+1])
        shift_logits = logits[:, :-1, :]
        shift_labels = batch[:, 1:]
        B, T, V = shift_logits.shape
        loss = F.cross_entropy(
            shift_logits.reshape(B * T, V), shift_labels.reshape(B * T),
            reduction="none"
        ).reshape(B, T)
        all_losses.append(loss.cpu())
    all_losses = torch.cat(all_losses, dim=0)  # (n_seqs, n_ctx-1)

    # Gather losses at first and second bigram positions
    first_losses = []
    second_losses = []
    for bg in all_bigrams:
        seq_idx = bg["seq_idx"]
        # Loss at position p means: loss of predicting token[p+1] given tokens[0..p]
        # For a bigram (a,b) at position p, the "induction" prediction is at position p:
        # given we just saw token 'a', predict token 'b'
        first_p = bg["first_pos"]  # loss[first_p] = loss of predicting b after a (first time)
        second_p = bg["second_pos"]  # loss[second_p] = loss of predicting b after a (second time)
        if first_p < all_losses.shape[1] and second_p < all_losses.shape[1]:
            first_losses.append(all_losses[seq_idx, first_p].item())
            second_losses.append(all_losses[seq_idx, second_p].item())

    first_losses = np.array(first_losses)
    second_losses = np.array(second_losses)
    diff = first_losses - second_losses  # positive = induction

    # Also track accuracy
    n_correct_first = 0
    n_correct_second = 0
    for bg in all_bigrams:
        seq_idx = bg["seq_idx"]
        first_p = bg["first_pos"]
        second_p = bg["second_pos"]
        if first_p < all_losses.shape[1] and second_p < all_losses.shape[1]:
            # We can't easily get accuracy from loss alone, but we tracked it
            pass

    return {
        "n_bigrams": len(first_losses),
        "mean_first_loss": float(first_losses.mean()),
        "mean_second_loss": float(second_losses.mean()),
        "mean_loss_diff": float(diff.mean()),
        "std_loss_diff": float(diff.std()),
        "median_loss_diff": float(np.median(diff)),
        "frac_positive": float((diff > 0).mean()),
    }


# =============================================================================
# Test 2: Toy repeated (16 random tokens x2, 100 samples)
# =============================================================================

@torch.no_grad()
def eval_toy_repeated(model, vocab_size, device, n_samples=100, half_len=16):
    """100 samples of [16 random tokens][16 random tokens]."""
    model.eval()
    rng = np.random.RandomState(42)
    seqs = rng.randint(1, vocab_size, size=(n_samples, half_len))
    doubled = np.concatenate([seqs, seqs], axis=1)
    sequences = torch.tensor(doubled, dtype=torch.long)

    # Process in small batches
    batch_size = 32
    all_losses = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = sequences[start:end].to(device)
        logits = model(batch)
        shift_logits = logits[:, :-1, :]
        shift_labels = batch[:, 1:]
        B, T, V = shift_logits.shape
        loss = F.cross_entropy(
            shift_logits.reshape(B * T, V), shift_labels.reshape(B * T),
            reduction="none"
        ).reshape(B, T)
        all_losses.append(loss.cpu())
    all_losses = torch.cat(all_losses, dim=0)  # (n_samples, 2*half_len - 1)

    # First half: positions 0..half_len-2 (predicting tokens 1..half_len-1)
    first_half_loss = all_losses[:, :half_len - 1].mean().item()
    # Second half: positions half_len..2*half_len-2 (predicting tokens half_len+1..2*half_len-1)
    second_half_loss = all_losses[:, half_len:].mean().item()
    loss_diff = first_half_loss - second_half_loss

    # Per-position loss curve (averaged over samples)
    per_pos_loss = all_losses.mean(dim=0).numpy()

    # Accuracy on second half
    second_half_logits_all = []
    second_half_targets_all = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = sequences[start:end].to(device)
        logits = model(batch)
        pred = logits[:, half_len:2*half_len-1, :].argmax(dim=-1)
        tgt = batch[:, half_len+1:2*half_len]
        second_half_logits_all.append(pred.cpu())
        second_half_targets_all.append(tgt.cpu())
    preds = torch.cat(second_half_logits_all, dim=0)
    targets = torch.cat(second_half_targets_all, dim=0)
    accuracy = (preds == targets).float().mean().item()

    return {
        "first_half_loss": first_half_loss,
        "second_half_loss": second_half_loss,
        "loss_diff": loss_diff,
        "accuracy": accuracy,
        "per_pos_loss": per_pos_loss.tolist(),
    }


# =============================================================================
# Main: sweep checkpoints and plot
# =============================================================================

def load_model(checkpoint_path, d_model, n_head, n_ctx, device="cpu"):
    model = AttentionLM(
        vocab_size=5000, n_ctx=n_ctx, d_model=d_model,
        n_head=n_head, n_layers=2,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = {}
    for k, v in ckpt["model_state_dict"].items():
        state_dict[k.replace("_orig_mod.", "")] = v
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, ckpt.get("step", 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--n-head", type=int, required=True)
    parser.add_argument("--n-ctx", type=int, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for results (default: induction_results/)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    output_dir = Path(args.output) if args.output else Path(__file__).parent / "induction_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load validation data
    if args.n_ctx == 1024:
        val_path = Path(__file__).parent / "cached_tokens" / "dsir_pile_val.pt"
    else:
        val_path = Path(__file__).parent / "cached_tokens" / f"dsir_pile_val_ctx{args.n_ctx}.pt"
    val_data = torch.load(val_path, weights_only=True).to(torch.long)[:, :args.n_ctx]
    print(f"Loaded val data: {val_data.shape}")

    # Find all checkpoints, sorted by step
    ckpt_files = sorted(ckpt_dir.glob("step_*.pt"),
                        key=lambda p: int(p.stem.split("_")[1]))
    print(f"Found {len(ckpt_files)} checkpoints: {[f.stem for f in ckpt_files]}")

    all_results = []
    for ckpt_path in ckpt_files:
        print(f"\n{'='*60}")
        print(f"Evaluating {ckpt_path.name}...")
        model, step = load_model(ckpt_path, args.d_model, args.n_head, args.n_ctx, args.device)

        # Test 1: In-distribution bigrams
        id_result = eval_indistribution_bigrams(model, val_data, args.n_ctx, args.device)
        print(f"  In-dist bigrams: n={id_result['n_bigrams']}, "
              f"loss_diff={id_result.get('mean_loss_diff', 'N/A'):.4f}, "
              f"frac_positive={id_result.get('frac_positive', 'N/A'):.3f}")

        # Test 2: Toy repeated
        toy_result = eval_toy_repeated(model, 5000, args.device, n_samples=100, half_len=16)
        print(f"  Toy repeated: loss_diff={toy_result['loss_diff']:.4f}, "
              f"acc={toy_result['accuracy']:.4f}")

        all_results.append({
            "step": step,
            "indist": id_result,
            "toy": toy_result,
        })

        del model
        torch.cuda.empty_cache() if args.device != "cpu" else None

    # Save raw results
    run_name = run_dir.name
    results_path = output_dir / f"eval_results_{run_name}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [r["step"] for r in all_results]
        tokens_B = [s * 64 * 1024 / 1e9 for s in steps]  # approximate

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Induction Evaluation: {run_name}", fontsize=14)

        # Plot 1: Loss difference over training
        ax = axes[0, 0]
        id_diffs = [r["indist"].get("mean_loss_diff", 0) for r in all_results]
        toy_diffs = [r["toy"]["loss_diff"] for r in all_results]
        ax.plot(steps, id_diffs, "o-", label="In-dist bigrams", color="blue")
        ax.plot(steps, toy_diffs, "s-", label="Toy repeated (half=16)", color="red")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss Diff (1st - 2nd occurrence)")
        ax.set_title("Induction Signal: Loss Difference")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Raw losses (in-distribution)
        ax = axes[0, 1]
        id_first = [r["indist"].get("mean_first_loss", 0) for r in all_results]
        id_second = [r["indist"].get("mean_second_loss", 0) for r in all_results]
        ax.plot(steps, id_first, "o-", label="1st occurrence", color="blue")
        ax.plot(steps, id_second, "s-", label="2nd occurrence", color="green")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("In-Distribution Bigram Losses")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Toy per-position loss curve (latest checkpoint)
        ax = axes[1, 0]
        latest = all_results[-1]
        per_pos = latest["toy"]["per_pos_loss"]
        half_len = 16
        positions = list(range(len(per_pos)))
        ax.plot(positions, per_pos, "-", color="purple")
        ax.axvline(x=half_len - 1, color="red", linestyle="--", alpha=0.7, label="Repeat boundary")
        ax.set_xlabel("Position")
        ax.set_ylabel("Loss")
        ax.set_title(f"Toy Per-Position Loss (step {latest['step']})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Fraction of bigrams with positive diff & toy accuracy
        ax = axes[1, 1]
        frac_pos = [r["indist"].get("frac_positive", 0) for r in all_results]
        toy_acc = [r["toy"]["accuracy"] for r in all_results]
        ax.plot(steps, frac_pos, "o-", label="Frac bigrams w/ positive diff", color="blue")
        ax.plot(steps, toy_acc, "s-", label="Toy accuracy", color="red")
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Fraction / Accuracy")
        ax.set_title("Induction Indicators")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / f"induction_eval_{run_name}.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Saved plot to {plot_path}")
        plt.close()

    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
