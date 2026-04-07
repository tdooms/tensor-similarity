#!/usr/bin/env python3
"""Quick induction tests on saved checkpoints.

Test 1: 16 short repeated sequences (half_len=8, total len=16)
Test 2: Real repeated bigrams found in the DSIR Pile validation data

Usage:
    python test_induction_quick.py --checkpoint runs/.../checkpoints/step_20000.pt --d-model 256 --n-head 8 --n-ctx 1024
    python test_induction_quick.py --checkpoint runs/.../checkpoints/step_30000.pt --d-model 512 --n-head 16 --n-ctx 512
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, einsum
from pathlib import Path
from collections import defaultdict


# =============================================================================
# Model (must match training script exactly)
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
        if return_debug:
            return out, {"pattern": pattern}
        return out


class AttentionLM(nn.Module):
    def __init__(self, vocab_size, n_ctx, d_model, n_head, n_layers,
                 attn_scale=1.0, rope_base=10000, use_rmsnorm_qk=False,
                 use_bias_qkv=True, use_bias_o=True,
                 std_embed=0.02, std_qkv=0.02, std_o=0.01,
                 norm_type="layernorm"):
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

    def forward(self, input_ids, return_debug=False):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.unembed(x)
        return logits


# =============================================================================
# Test 1: Short repeated sequences (16 seqs, half_len=8)
# =============================================================================

def test_short_repeated(model, vocab_size, device):
    """16 sequences of form [a b c d e f g h a b c d e f g h]"""
    rng = np.random.RandomState(42)
    n_seq = 16
    half_len = 8
    seqs = rng.randint(1, vocab_size, size=(n_seq, half_len))
    doubled = np.concatenate([seqs, seqs], axis=1)  # shape (16, 16)
    sequences = torch.tensor(doubled, dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(sequences)

    # Accuracy on second half (predicting positions half_len..2*half_len-1)
    pred_logits = logits[:, half_len:2*half_len-1, :]  # predict tokens at pos half_len+1..2*half_len
    targets = sequences[:, half_len+1:2*half_len]
    preds = pred_logits.argmax(dim=-1)
    correct = (preds == targets).float()
    accuracy = correct.mean().item()

    # Per-position accuracy in second half
    per_pos_acc = correct.mean(dim=0).cpu().numpy()

    # Loss on first vs second half
    shift_logits = logits[:, :-1, :]
    shift_labels = sequences[:, 1:]
    B, T, V = shift_logits.shape
    loss = F.cross_entropy(
        shift_logits.reshape(B * T, V), shift_labels.reshape(B * T),
        reduction="none"
    ).reshape(B, T)
    mean_loss = loss.mean(dim=0).cpu().numpy()
    first_half_loss = mean_loss[:half_len - 1].mean()
    second_half_loss = mean_loss[half_len:].mean()
    induction_score = float(first_half_loss - second_half_loss)

    print(f"\n  TEST 1: Short repeated sequences (16 seqs, half_len=8)")
    print(f"    Induction score: {induction_score:.4f}")
    print(f"    1st half loss:   {first_half_loss:.4f}")
    print(f"    2nd half loss:   {second_half_loss:.4f}")
    print(f"    2nd half accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"    Per-position acc (2nd half): {[f'{a:.3f}' for a in per_pos_acc]}")
    return induction_score, accuracy


# =============================================================================
# Test 2: Real repeated bigrams from the dataset
# =============================================================================

def find_repeated_bigrams(data_path, n_ctx, vocab_size, n_examples=10):
    """Find sequences in the cached data that contain repeated bigrams."""
    data = torch.load(data_path, weights_only=True).to(torch.long)[:, :n_ctx]
    print(f"\n  Searching {data.shape[0]} sequences for repeated bigrams...")

    examples = []
    for seq_idx in range(data.shape[0]):
        seq = data[seq_idx].numpy()
        # Find bigrams and their positions
        bigram_positions = defaultdict(list)
        for i in range(len(seq) - 1):
            bg = (int(seq[i]), int(seq[i+1]))
            bigram_positions[bg].append(i)

        # Find bigrams that repeat (appear at 2+ distinct positions)
        for bg, positions in bigram_positions.items():
            if len(positions) >= 2:
                # Take first and second occurrence
                pos1, pos2 = positions[0], positions[1]
                if pos2 - pos1 > 2:  # not overlapping/adjacent
                    examples.append({
                        "seq_idx": seq_idx,
                        "bigram": bg,
                        "pos1": pos1,
                        "pos2": pos2,
                        "distance": pos2 - pos1,
                    })
                    if len(examples) >= n_examples:
                        break
        if len(examples) >= n_examples:
            break

    print(f"  Found {len(examples)} repeated bigram examples")
    for i, ex in enumerate(examples):
        print(f"    [{i}] bigram={ex['bigram']} pos1={ex['pos1']} pos2={ex['pos2']} dist={ex['distance']}")

    return examples, data


def test_repeated_bigrams(model, examples, data, n_ctx, device):
    """Test if model predicts the 2nd token of repeated bigrams correctly."""
    if not examples:
        print("  No repeated bigram examples found!")
        return 0.0, 0.0

    model.eval()
    total_correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for ex in examples:
            seq = data[ex["seq_idx"], :n_ctx].unsqueeze(0).to(device)
            logits = model(seq)  # (1, n_ctx, vocab)

            # At position pos2, the model should predict token bigram[1]
            # (given that it just saw bigram[0] at pos2, and saw bigram[0],bigram[1] at pos1,pos1+1)
            target_pos = ex["pos2"]  # position of first token of repeated bigram
            target_token = ex["bigram"][1]

            # logits at pos2 predict token at pos2+1
            pred_logits = logits[0, target_pos, :]
            pred_token = pred_logits.argmax().item()
            loss = F.cross_entropy(pred_logits.unsqueeze(0),
                                   torch.tensor([target_token], device=device)).item()

            is_correct = pred_token == target_token
            total_correct += int(is_correct)
            total_loss += loss

    accuracy = total_correct / len(examples)
    avg_loss = total_loss / len(examples)

    print(f"\n  TEST 2: Real repeated bigrams ({len(examples)} examples)")
    print(f"    Accuracy: {total_correct}/{len(examples)} = {accuracy:.2%}")
    print(f"    Avg loss at bigram prediction: {avg_loss:.4f}")
    print(f"    Random baseline: {1/5000:.4%}")
    return accuracy, avg_loss


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--n-head", type=int, required=True)
    parser.add_argument("--n-ctx", type=int, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    vocab_size = 5000
    n_layers = 2

    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Config: d_model={args.d_model}, n_head={args.n_head}, n_ctx={args.n_ctx}")

    model = AttentionLM(
        vocab_size=vocab_size, n_ctx=args.n_ctx, d_model=args.d_model,
        n_head=args.n_head, n_layers=n_layers,
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    # Handle compiled model state dict (strip _orig_mod. prefix)
    state_dict = {}
    for k, v in ckpt["model_state_dict"].items():
        new_k = k.replace("_orig_mod.", "")
        state_dict[new_k] = v
    model.load_state_dict(state_dict)
    model = model.to(args.device)
    model.eval()

    step = ckpt.get("step", "?")
    print(f"Loaded step {step}")

    # Test 1: Short repeated sequences
    test_short_repeated(model, vocab_size, args.device)

    # Test 2: Real repeated bigrams
    # Use the cached validation data
    if args.n_ctx == 1024:
        val_path = Path(__file__).parent / "cached_tokens" / "dsir_pile_val.pt"
    else:
        val_path = Path(__file__).parent / "cached_tokens" / f"dsir_pile_val_ctx{args.n_ctx}.pt"

    if val_path.exists():
        examples, data = find_repeated_bigrams(val_path, args.n_ctx, vocab_size, n_examples=10)
        test_repeated_bigrams(model, examples, data, args.n_ctx, args.device)
    else:
        print(f"\n  Skipping bigram test — no cached val data at {val_path}")


if __name__ == "__main__":
    main()
