#!/usr/bin/env python3
"""Self-contained Hoogland et al. replication for induction head formation.

Architecture: 2L attention-only, d_model=256, 8 heads, standard softmax
Data: DSIR-filtered Pile (streaming), GPT-2 tokenizer truncated to vocab=5000
Target: ~5B tokens, but stop once induction is clearly learned
Induction eval every 2500 steps to catch onset (reported at 6.5k-17k steps).

This script is fully self-contained — no external model/training imports needed.
Requires: torch, einops, transformers, datasets, muon

Usage:
    python train_hoogland_standalone.py
    python train_hoogland_standalone.py --batch-size 128
    python train_hoogland_standalone.py --batch-size 128 --no-compile
"""
import math
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import IterableDataset, Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, einsum
from tqdm import tqdm


# =============================================================================
# Config
# =============================================================================
N_CTX = 1024
VOCAB_SIZE = 5000
TARGET_TOKENS = 5_000_000_000


# =============================================================================
# Model components (inlined from mel's codebase)
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


class SoftmaxAttention(nn.Module):
    """Standard softmax attention with causal masking and RoPE."""
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
            return out, {"q": q, "k": k, "v": v, "scores": scores, "pattern": pattern, "z": z}
        return out


class AttentionLM(nn.Module):
    """Autoregressive attention-only language model.

    Architecture: embed -> n_layers x SoftmaxAttention -> LayerNorm -> unembed
    """
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
            SoftmaxAttention(
                d_model=d_model, n_head=n_head, n_ctx=n_ctx,
                scale=attn_scale, use_rmsnorm_qk=use_rmsnorm_qk,
                use_bias_qkv=use_bias_qkv, use_bias_o=use_bias_o,
                rope_base=rope_base,
            )
            for _ in range(n_layers)
        ])

        self.unembed = nn.Linear(d_model, vocab_size, bias=False)
        self._init_weights(std_embed, std_qkv, std_o)

    def _init_weights(self, std_embed, std_qkv, std_o):
        nn.init.normal_(self.embed.weight, mean=0.0, std=std_embed)
        nn.init.normal_(self.unembed.weight, mean=0.0, std=std_embed)
        for layer in self.layers:
            nn.init.normal_(layer.q.weight, mean=0.0, std=std_qkv)
            nn.init.normal_(layer.k.weight, mean=0.0, std=std_qkv)
            nn.init.normal_(layer.v.weight, mean=0.0, std=std_qkv)
            nn.init.normal_(layer.o.weight, mean=0.0, std=std_o)
            if layer.q.bias is not None:
                nn.init.zeros_(layer.q.bias)
                nn.init.zeros_(layer.k.bias)
                nn.init.zeros_(layer.v.bias)
            if layer.o.bias is not None:
                nn.init.zeros_(layer.o.bias)

    def forward(self, input_ids, return_debug=False):
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
            if ".norm." not in name:
                return True
    return False


def create_optimizer(model, lr=3e-4, muon_lr=0.02, weight_decay=0.1,
                     betas=(0.9, 0.95), use_muon=True):
    if not use_muon:
        from torch.optim import AdamW
        decay, nodecay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            if "bias" in n or "norm" in n:
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
        elif "bias" in name or "norm" in name:
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
# Data: streaming DSIR-filtered Pile with GPT-2 tokenizer, vocab truncated to 5000
# =============================================================================

class DSIRPileStreaming(IterableDataset):
    """Stream DSIR-filtered Pile, tokenize with GPT-2, truncate vocab to 5000."""
    def __init__(self, n_ctx=N_CTX, vocab_size=VOCAB_SIZE):
        self.n_ctx = n_ctx
        self.vocab_size = vocab_size

    def __iter__(self):
        from datasets import load_dataset
        from transformers import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        ds = load_dataset(
            "stanford-crfm/DSIR-filtered-pile-50M",
            split="train", streaming=True,
        )

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
    """Pre-cached token windows."""
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
    """Cache a validation set from the Pile for periodic eval."""
    if cache_dir is None:
        cache_dir = Path(__file__).parent / "cached_tokens"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    val_path = cache_dir / "dsir_pile_val.pt"

    if val_path.exists():
        print(f"Pile val cache exists at {val_path}")
        return val_path

    from datasets import load_dataset
    from transformers import GPT2Tokenizer

    print(f"Caching {n_val} Pile val windows...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    ds = load_dataset(
        "stanford-crfm/DSIR-filtered-pile-50M",
        split="train", streaming=True,
    )

    all_chunks = []
    token_buffer = []
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
# Induction evaluation
# =============================================================================

def make_repeated_sequences(vocab_size, half_len=512, n_sequences=100, seed=42):
    rng = np.random.RandomState(seed)
    seqs = rng.randint(1, vocab_size, size=(n_sequences, half_len))
    doubled = np.concatenate([seqs, seqs], axis=1)
    return torch.tensor(doubled, dtype=torch.long), half_len


@torch.no_grad()
def eval_induction(model, sequences, half_len, device, batch_size=16):
    model.eval()
    n = sequences.shape[0]
    all_losses = []
    total_correct = 0
    total_tokens = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = sequences[start:end].to(device)
        logits = model(batch)
        shift_logits = logits[:, :-1, :]
        shift_labels = batch[:, 1:]
        B, T, V = shift_logits.shape
        loss = F.cross_entropy(
            shift_logits.reshape(B * T, V), shift_labels.reshape(B * T),
            reduction="none",
        ).reshape(B, T)
        all_losses.append(loss.cpu())

        pred_logits = logits[:, half_len:2*half_len-1, :]
        targets = batch[:, half_len+1:2*half_len]
        preds = pred_logits.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_tokens += targets.numel()

    all_losses = torch.cat(all_losses, dim=0)
    mean_loss = all_losses.mean(dim=0).numpy()
    first_half = mean_loss[:half_len - 1].mean()
    second_half = mean_loss[half_len:].mean()
    induction_score = float(first_half - second_half)
    accuracy = total_correct / total_tokens
    return induction_score, float(first_half), float(second_half), accuracy


# =============================================================================
# Training loop
# =============================================================================

def main(batch_size=64, use_compile=True):
    d_model = 256
    n_head = 8
    n_layers = 2

    tokens_per_step = batch_size * N_CTX
    max_steps = TARGET_TOKENS // tokens_per_step

    print(f"=== Hoogland et al. Replication (self-contained) ===")
    print(f"2L attn-only, d={d_model}, {n_head} heads, softmax, no QK norm")
    print(f"Data: DSIR-filtered Pile (streaming), GPT-2 tokenizer, vocab={VOCAB_SIZE}")
    print(f"Batch: {batch_size}, n_ctx: {N_CTX}, tokens/step: {tokens_per_step:,}")
    print(f"Max steps: {max_steps:,} ({max_steps * tokens_per_step / 1e9:.1f}B tokens)")

    # Cache val set
    val_path = cache_pile_val(n_ctx=N_CTX, vocab_size=VOCAB_SIZE, n_val=500)

    # Streaming train, cached val
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
        use_rmsnorm_qk=False, use_bias_qkv=True, use_bias_o=True,
        std_embed=0.02, std_qkv=0.02, std_o=0.01, norm_type="layernorm",
    )
    model = model.to(device)
    if use_compile:
        model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({'compiled' if use_compile else 'eager'})")

    # Induction test sequences
    sequences, half_len = make_repeated_sequences(VOCAB_SIZE, half_len=512, n_sequences=100)
    print(f"Induction test: {sequences.shape[0]} seqs, half_len={half_len}, vocab={VOCAB_SIZE}")

    # Optimizer & scheduler
    optimizer = create_optimizer(model, lr=3e-4, muon_lr=0.02, weight_decay=0.1,
                                 betas=(0.9, 0.95), use_muon=True)
    scheduler = create_scheduler(optimizer, warmup_steps=1000, max_steps=max_steps, lr_decay_frac=0.1)

    # Run dir
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(f"runs/{timestamp}_hoogland_replication")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    metrics_file = run_dir / "metrics.jsonl"

    def log(d, step):
        d["step"] = step
        with open(metrics_file, "a") as f:
            f.write(json.dumps(d) + "\n")

    # Training
    induction_found = False
    step = 0
    pbar = tqdm(total=max_steps, desc="Hoogland replication")

    for batch in train_dl:
        if step >= max_steps or induction_found:
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

        pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.update(1)

        # Log every 1000 steps
        if step % 1000 == 0:
            log({"train_loss": loss.item(), "lr": scheduler.get_last_lr()[0]}, step)

        # Eval + induction every 2500 steps
        if step % 2500 == 0:
            val_loss = evaluate(model, val_dl, device)
            log({"val_loss": val_loss}, step)

            score, first, second, accuracy = eval_induction(model, sequences, half_len, device)
            tokens_seen = step * tokens_per_step
            print(f"\n  [step {step}, {tokens_seen/1e9:.2f}B tokens] "
                  f"induction_score={score:.4f} "
                  f"1st={first:.4f} 2nd={second:.4f} "
                  f"toy_acc={accuracy:.6f} ({accuracy*100:.4f}%)")
            log({
                "induction_score": score,
                "induction_first_half_loss": first,
                "induction_second_half_loss": second,
                "toy_induction_accuracy": accuracy,
                "tokens_seen_B": tokens_seen / 1e9,
            }, step)

            if accuracy > 0.10:
                induction_found = True
                print(f"\n  *** INDUCTION FOUND at step {step} ({tokens_seen/1e9:.2f}B tokens)! ***\n")

            model.train()

        # Save checkpoint every 10000 steps
        if step % 10000 == 0:
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
            }, run_dir / "checkpoints" / f"step_{step}.pt")

    pbar.close()

    # Final eval
    if not induction_found:
        val_loss = evaluate(model, val_dl, device)
        score, first, second, accuracy = eval_induction(model, sequences, half_len, device)
        log({"val_loss": val_loss}, step)
        log({
            "induction_score": score,
            "induction_first_half_loss": first,
            "induction_second_half_loss": second,
            "toy_induction_accuracy": accuracy,
            "tokens_seen_B": step * tokens_per_step / 1e9,
        }, step)

    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
    }, run_dir / "checkpoints" / "final.pt")

    print(f"\nDone at step {step}. Run dir: {run_dir}")
    if induction_found:
        print(f"Induction was found! Stopping early.")
    else:
        print(f"Induction NOT found after {step} steps ({step * tokens_per_step / 1e9:.1f}B tokens)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    main(args.batch_size, use_compile=not args.no_compile)
