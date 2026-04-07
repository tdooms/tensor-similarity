#!/usr/bin/env python3
"""Bilinear attention + BatchNorm, matching Run B config (d=512, ctx=512).

Architecture: 2L attention-only, d_model=512, 16 heads, bilinear with BatchNorm on Q1,K1,Q2,K2
Data: DSIR-filtered Pile (streaming), GPT-2 tokenizer truncated to vocab=5000
Hypothesis: Can bilinear+batchnorm learn induction on real data?

Usage:
    python train_bilinear_bn_ctx512_d512.py
    python train_bilinear_bn_ctx512_d512.py --batch-size 64
    python train_bilinear_bn_ctx512_d512.py --batch-size 64 --no-compile
"""
import math
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import IterableDataset, Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, einsum


# =============================================================================
# Config — matches Run B
# =============================================================================
N_CTX = 512
VOCAB_SIZE = 5000
TARGET_TOKENS = 5_000_000_000


# =============================================================================
# Model components
# =============================================================================

class Rotary(nn.Module):
    """Rotary Position Embedding (RoPE)."""
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


class BilinearBatchNormAttention(nn.Module):
    """Bilinear attention with BatchNorm on Q1,K1,Q2,K2 projections and causal masking.

    pattern = (scores1 * scores2) / d_head^2 * causal_mask
    where scores1 = q1 @ k1.T, scores2 = q2 @ k2.T
    BatchNorm1d applied to each projection (flattened B*T dimension).
    """
    def __init__(self, d_model, n_head, n_ctx, scale=1.0, rope_base=10000):
        super().__init__()
        self.d_head = d_model // n_head
        self.n_head = n_head
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.scale = scale

        self.rotary = Rotary(self.d_head, n_ctx, base=rope_base)

        # Multiplicative causal mask (0/1, not -inf/0)
        causal_mask = torch.tril(torch.ones(n_ctx, n_ctx))
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        # Two QK circuits
        self.q1 = nn.Linear(d_model, d_model, bias=True)
        self.k1 = nn.Linear(d_model, d_model, bias=True)
        self.q2 = nn.Linear(d_model, d_model, bias=True)
        self.k2 = nn.Linear(d_model, d_model, bias=True)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

        # BatchNorm on each QK projection
        self.bn_q1 = nn.BatchNorm1d(d_model)
        self.bn_k1 = nn.BatchNorm1d(d_model)
        self.bn_q2 = nn.BatchNorm1d(d_model)
        self.bn_k2 = nn.BatchNorm1d(d_model)

    def forward(self, x):
        B, T, _ = x.shape

        # Project then BatchNorm (flattened B*T)
        q1 = self.bn_q1(self.q1(x).reshape(B * T, -1)).reshape(B, T, -1)
        k1 = self.bn_k1(self.k1(x).reshape(B * T, -1)).reshape(B, T, -1)
        q2 = self.bn_q2(self.q2(x).reshape(B * T, -1)).reshape(B, T, -1)
        k2 = self.bn_k2(self.k2(x).reshape(B * T, -1)).reshape(B, T, -1)

        q1 = rearrange(q1, "b t (nh dh) -> b t nh dh", nh=self.n_head)
        k1 = rearrange(k1, "b t (nh dh) -> b t nh dh", nh=self.n_head)
        q2 = rearrange(q2, "b t (nh dh) -> b t nh dh", nh=self.n_head)
        k2 = rearrange(k2, "b t (nh dh) -> b t nh dh", nh=self.n_head)
        v = rearrange(self.v(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)

        q1 = self.rotary(q1)
        k1 = self.rotary(k1)
        q2 = self.rotary(q2)
        k2 = self.rotary(k2)

        scores1 = einsum(q1, k1, "b sq nh dh, b sk nh dh -> b nh sq sk")
        scores2 = einsum(q2, k2, "b sq nh dh, b sk nh dh -> b nh sq sk")

        pattern = (scores1 * scores2) / (self.d_head ** 2)
        pattern = pattern * self.causal_mask[None, None, :T, :T]

        z = einsum(pattern, v, "b nh sq sk, b sk nh dh -> b sq nh dh")
        z_merge = rearrange(z, "b seq nh dh -> b seq (nh dh)")
        out = x + self.scale * self.o(z_merge)
        return out


class AttentionLM(nn.Module):
    """Autoregressive attention-only LM with bilinear+batchnorm attention."""
    def __init__(self, vocab_size, n_ctx, d_model, n_head, n_layers,
                 attn_scale=1.0, rope_base=10000,
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
            BilinearBatchNormAttention(
                d_model=d_model, n_head=n_head, n_ctx=n_ctx,
                scale=attn_scale, rope_base=rope_base,
            )
            for _ in range(n_layers)
        ])

        self.unembed = nn.Linear(d_model, vocab_size, bias=False)
        self._init_weights(std_embed, std_qkv, std_o)

    def _init_weights(self, std_embed, std_qkv, std_o):
        nn.init.normal_(self.embed.weight, mean=0.0, std=std_embed)
        nn.init.normal_(self.unembed.weight, mean=0.0, std=std_embed)
        for layer in self.layers:
            nn.init.normal_(layer.q1.weight, mean=0.0, std=std_qkv)
            nn.init.normal_(layer.k1.weight, mean=0.0, std=std_qkv)
            nn.init.normal_(layer.q2.weight, mean=0.0, std=std_qkv)
            nn.init.normal_(layer.k2.weight, mean=0.0, std=std_qkv)
            nn.init.normal_(layer.v.weight, mean=0.0, std=std_qkv)
            nn.init.normal_(layer.o.weight, mean=0.0, std=std_o)
            if layer.q1.bias is not None:
                nn.init.zeros_(layer.q1.bias)
                nn.init.zeros_(layer.k1.bias)
                nn.init.zeros_(layer.q2.bias)
                nn.init.zeros_(layer.k2.bias)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.unembed(x)
        return logits


# =============================================================================
# Optimizer: Muon + AdamW
# =============================================================================

def _is_muon_param(name, param):
    if param.ndim < 2:
        return False
    for prefix in ("layers.",):
        if name.startswith(prefix) and name.endswith(".weight"):
            if ".bn_" not in name and ".norm" not in name:
                return True
    return False


def create_optimizer(model, lr=3e-4, muon_lr=0.02, weight_decay=0.1,
                     betas=(0.9, 0.95), use_muon=True):
    if not use_muon:
        from torch.optim import AdamW
        decay, nodecay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            if "bias" in n or "norm" in n or "bn_" in n:
                nodecay.append(p)
            else:
                decay.append(p)
        return AdamW([
            {"params": decay, "weight_decay": weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ], lr=lr, betas=betas)

    from muon import SingleDeviceMuonWithAuxAdam
    muon_params, adam_decay, adam_nodecay = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if _is_muon_param(name, param):
            muon_params.append(param)
        elif "bias" in name or "norm" in name or "bn_" in name:
            adam_nodecay.append(param)
        else:
            adam_decay.append(param)

    param_groups = [
        dict(params=muon_params, use_muon=True, lr=muon_lr, weight_decay=weight_decay),
        dict(params=adam_decay, use_muon=False, lr=lr, betas=betas, weight_decay=weight_decay),
        dict(params=adam_nodecay, use_muon=False, lr=lr, betas=betas, weight_decay=0.0),
    ]
    return SingleDeviceMuonWithAuxAdam(param_groups)


def create_scheduler(optimizer, warmup_steps, max_steps, lr_decay_frac=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(progress, 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_decay_frac + coeff * (1.0 - lr_decay_frac)
    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Loss & eval
# =============================================================================

def compute_loss(logits, input_ids, label_smoothing=0.0):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    B, T, V = shift_logits.shape
    return F.cross_entropy(shift_logits.view(B * T, V), shift_labels.view(B * T),
                           label_smoothing=label_smoothing)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss, n = 0.0, 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        logits = model(input_ids)
        total_loss += compute_loss(logits, input_ids).item()
        n += 1
    return total_loss / max(1, n)


# =============================================================================
# Data
# =============================================================================

class DSIRPileStreaming(IterableDataset):
    def __init__(self, n_ctx=N_CTX, vocab_size=VOCAB_SIZE):
        self.n_ctx = n_ctx
        self.vocab_size = vocab_size

    def __iter__(self):
        from datasets import load_dataset
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        ds = load_dataset("stanford-crfm/DSIR-filtered-pile-50M", split="train", streaming=True)
        token_buffer = []
        for example in ds:
            tokens = tokenizer.encode(example["contents"])
            tokens = [t % self.vocab_size for t in tokens]
            token_buffer.extend(tokens)
            while len(token_buffer) >= self.n_ctx:
                chunk = token_buffer[:self.n_ctx]
                token_buffer = token_buffer[self.n_ctx:]
                yield {"input_ids": torch.tensor(chunk, dtype=torch.long)}


class CachedDataset(Dataset):
    def __init__(self, path, n_ctx=N_CTX, max_samples=None):
        data = torch.load(path, weights_only=True).to(torch.long)[:, :n_ctx]
        if max_samples:
            data = data[:max_samples]
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return {"input_ids": self.data[idx]}


def cache_pile_val(n_ctx=N_CTX, vocab_size=VOCAB_SIZE, n_val=500, cache_dir=None):
    if cache_dir is None:
        cache_dir = Path(__file__).parent / "cached_tokens"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    val_path = cache_dir / f"dsir_pile_val_ctx{n_ctx}.pt"
    if val_path.exists():
        print(f"Pile val cache exists at {val_path}")
        return val_path
    from datasets import load_dataset
    from transformers import GPT2Tokenizer
    print(f"Caching {n_val} Pile val windows (n_ctx={n_ctx})...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    ds = load_dataset("stanford-crfm/DSIR-filtered-pile-50M", split="train", streaming=True)
    all_chunks, token_buffer = [], []
    for example in ds:
        tokens = tokenizer.encode(example["contents"])
        tokens = [t % vocab_size for t in tokens]
        token_buffer.extend(tokens)
        while len(token_buffer) >= n_ctx:
            chunk = token_buffer[:n_ctx]
            token_buffer = token_buffer[n_ctx:]
            all_chunks.append(chunk)
            if len(all_chunks) >= n_val:
                break
        if len(all_chunks) >= n_val:
            break
    val_data = torch.tensor(all_chunks, dtype=torch.int16)
    torch.save(val_data, val_path)
    print(f"Saved: {val_path} ({val_data.shape})")
    return val_path


# =============================================================================
# Induction evaluation (robust: toy repeated + in-distribution bigrams)
# =============================================================================

from collections import defaultdict


@torch.no_grad()
def eval_toy_repeated(model, vocab_size, device, n_samples=100, half_len=16):
    """100 samples of [16 random tokens][16 random tokens]."""
    model.eval()
    rng = np.random.RandomState(42)
    seqs = rng.randint(1, vocab_size, size=(n_samples, half_len))
    doubled = np.concatenate([seqs, seqs], axis=1)
    sequences = torch.tensor(doubled, dtype=torch.long)
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
    all_losses = torch.cat(all_losses, dim=0)
    first_half_loss = all_losses[:, :half_len - 1].mean().item()
    second_half_loss = all_losses[:, half_len:].mean().item()

    # accuracy on second half
    total_correct = 0
    total_tokens = 0
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = sequences[start:end].to(device)
        logits = model(batch)
        preds = logits[:, half_len:2*half_len-1, :].argmax(dim=-1)
        tgt = batch[:, half_len+1:2*half_len]
        total_correct += (preds == tgt).sum().item()
        total_tokens += tgt.numel()

    return {
        "toy_loss_diff": first_half_loss - second_half_loss,
        "toy_first_loss": first_half_loss,
        "toy_second_loss": second_half_loss,
        "toy_accuracy": total_correct / total_tokens,
    }


def _find_repeated_bigrams(data, n_ctx):
    """Pre-compute repeated bigram positions for all sequences."""
    results = []
    for seq_idx in range(data.shape[0]):
        seq = data[seq_idx, :n_ctx].numpy()
        seen = {}
        for i in range(len(seq) - 1):
            bg = (int(seq[i]), int(seq[i + 1]))
            if bg in seen:
                if i - seen[bg] > 2:
                    results.append((seq_idx, seen[bg], i, bg[1]))
            else:
                seen[bg] = i
    return results


@torch.no_grad()
def eval_indist_bigrams(model, val_data, n_ctx, bigram_positions, device):
    """Measure loss at first vs second occurrence of repeated bigrams."""
    model.eval()
    batch_size = 16
    n_seqs = val_data.shape[0]
    all_losses = []
    for start in range(0, n_seqs, batch_size):
        end = min(start + batch_size, n_seqs)
        batch = val_data[start:end, :n_ctx].to(device)
        logits = model(batch)
        shift_logits = logits[:, :-1, :]
        shift_labels = batch[:, 1:]
        B, T, V = shift_logits.shape
        loss = F.cross_entropy(
            shift_logits.reshape(B * T, V), shift_labels.reshape(B * T),
            reduction="none"
        ).reshape(B, T)
        all_losses.append(loss.cpu())
    all_losses = torch.cat(all_losses, dim=0)

    first_losses, second_losses = [], []
    for seq_idx, pos1, pos2, _ in bigram_positions:
        if pos1 < all_losses.shape[1] and pos2 < all_losses.shape[1]:
            first_losses.append(all_losses[seq_idx, pos1].item())
            second_losses.append(all_losses[seq_idx, pos2].item())

    first_losses = np.array(first_losses)
    second_losses = np.array(second_losses)
    diff = first_losses - second_losses

    return {
        "indist_n_bigrams": len(first_losses),
        "indist_loss_diff": float(diff.mean()),
        "indist_first_loss": float(first_losses.mean()),
        "indist_second_loss": float(second_losses.mean()),
        "indist_frac_positive": float((diff > 0).mean()),
    }


# =============================================================================
# Training loop
# =============================================================================

def main(batch_size=64, use_compile=True):
    d_model = 512
    n_head = 16
    n_layers = 2

    tokens_per_step = batch_size * N_CTX
    max_steps = TARGET_TOKENS // tokens_per_step

    print(f"=== Bilinear + BatchNorm: ctx={N_CTX}, d={d_model} ===")
    print(f"2L attn-only, d={d_model}, {n_head} heads (d_head={d_model//n_head}), bilinear+BN")
    print(f"Data: DSIR-filtered Pile (streaming), GPT-2 tokenizer, vocab={VOCAB_SIZE}")
    print(f"Batch: {batch_size}, n_ctx: {N_CTX}, tokens/step: {tokens_per_step:,}")
    print(f"Max steps: {max_steps:,} ({max_steps * tokens_per_step / 1e9:.1f}B tokens)")

    val_path = cache_pile_val(n_ctx=N_CTX, vocab_size=VOCAB_SIZE, n_val=500)

    train_ds = DSIRPileStreaming(n_ctx=N_CTX, vocab_size=VOCAB_SIZE)
    val_ds = CachedDataset(val_path, n_ctx=N_CTX, max_samples=500)
    train_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.init()
        torch.cuda.empty_cache()

    model = AttentionLM(
        vocab_size=VOCAB_SIZE, n_ctx=N_CTX, d_model=d_model, n_head=n_head,
        n_layers=n_layers, attn_scale=1.0, rope_base=10000,
        std_embed=0.02, std_qkv=0.02, std_o=0.01, norm_type="layernorm",
    )
    model = model.to(device)
    if use_compile:
        model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({'compiled' if use_compile else 'eager'})")

    # Pre-compute induction eval data
    val_data = torch.load(val_path, weights_only=True).to(torch.long)[:, :N_CTX]
    bigram_positions = _find_repeated_bigrams(val_data[:100], N_CTX)
    print(f"Induction eval: {len(bigram_positions)} repeated bigrams in 100 val seqs")
    print(f"Toy eval: 100 samples of [16 random tokens] x 2")

    optimizer = create_optimizer(model, lr=3e-4, muon_lr=0.02, weight_decay=0.1,
                                 betas=(0.9, 0.95), use_muon=True)
    scheduler = create_scheduler(optimizer, warmup_steps=1000, max_steps=max_steps, lr_decay_frac=0.1)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(f"runs/{timestamp}_bilinear_bn_ctx512_d512")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    metrics_file = run_dir / "metrics.jsonl"

    def log(d, step):
        d["step"] = step
        with open(metrics_file, "a") as f:
            f.write(json.dumps(d) + "\n")

    step = 0
    t0 = time.time()

    for batch in train_dl:
        if step >= max_steps:
            break

        input_ids = batch["input_ids"].to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = compute_loss(logits, input_ids)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % 10 == 0:
            elapsed = time.time() - t0
            tokens_so_far = step * tokens_per_step
            log({"train_loss": loss.item(), "lr": scheduler.get_last_lr()[0]}, step)
            if step % 100 == 0:
                print(f"  step {step}/{max_steps}  loss={loss.item():.4f}  "
                      f"tokens={tokens_so_far/1e9:.3f}B  elapsed={elapsed:.0f}s")

        # Robust induction eval every 1k steps
        if step % 1000 == 0:
            tokens_seen = step * tokens_per_step

            # Toy repeated: 100 samples of [16 random][16 random]
            toy = eval_toy_repeated(model, VOCAB_SIZE, device, n_samples=100, half_len=16)
            log({**toy, "tokens_seen_B": tokens_seen / 1e9}, step)

            # In-distribution bigrams
            indist = eval_indist_bigrams(model, val_data[:100], N_CTX, bigram_positions, device)
            log({**indist, "tokens_seen_B": tokens_seen / 1e9}, step)

            print(f"\n  [step {step}, {tokens_seen/1e9:.2f}B tokens] "
                  f"toy_loss_diff={toy['toy_loss_diff']:.4f} toy_acc={toy['toy_accuracy']:.4f} | "
                  f"indist_loss_diff={indist['indist_loss_diff']:.4f} frac+={indist['indist_frac_positive']:.3f}")

            model.train()

        # Val loss every 2500 steps
        if step % 2500 == 0:
            val_loss = evaluate(model, val_dl, device)
            log({"val_loss": val_loss}, step)
            model.train()

        if step % 10000 == 0:
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
            }, run_dir / "checkpoints" / f"step_{step}.pt")

    # Final eval
    val_loss = evaluate(model, val_dl, device)
    toy = eval_toy_repeated(model, VOCAB_SIZE, device, n_samples=100, half_len=16)
    indist = eval_indist_bigrams(model, val_data[:100], N_CTX, bigram_positions, device)
    log({"val_loss": val_loss, **toy, **indist, "tokens_seen_B": step * tokens_per_step / 1e9}, step)
    print(f"Final: toy_loss_diff={toy['toy_loss_diff']:.4f} indist_loss_diff={indist['indist_loss_diff']:.4f}")

    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
    }, run_dir / "checkpoints" / "final.pt")

    print(f"\nDone at step {step}. Run dir: {run_dir}")
    print(f"Trained on {step * tokens_per_step / 1e9:.1f}B tokens")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    main(args.batch_size, use_compile=not args.no_compile)
