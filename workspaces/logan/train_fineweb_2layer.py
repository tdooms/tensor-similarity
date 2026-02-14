#!/usr/bin/env python3
"""Train 2-layer attn-only bilinear (batchnorm) model on FineWeb-Edu.

Architecture: 2-layer attention-only with BilinearBatchNorm + ortho_init.
Data: FineWeb-Edu (streamed, tokenized with SimpleStories tokenizer).
Induction: tracked every eval step using FineWeb-vocab repeated sequences.

Usage:
    python train_fineweb_2layer.py [--batch-size 64]
"""
import math, torch, torch.nn as nn
import numpy as np
import torch.nn.functional as F
from lib import AttentionLM, Trainer
from lib.attention import BilinearAttention
from train_induction_static_norm import BilinearBatchNorm
from einops import rearrange, einsum

from torch.utils.data import Dataset, DataLoader, IterableDataset
from pathlib import Path


# ---- FineWeb Data ----

class FineWebDataset(IterableDataset):
    """Stream FineWeb-Edu, tokenize on-the-fly, yield n_ctx chunks."""

    def __init__(self, n_ctx=512, split="train", buffer_size=50000):
        self.n_ctx = n_ctx
        self.split = split
        self.buffer_size = buffer_size

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "SimpleStories/SimpleStories-V2-1.25M"
        )

    def __iter__(self):
        from datasets import load_dataset
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu", "sample-10BT",
            split="train", streaming=True,
        )

        token_buffer = []
        for example in ds:
            tokens = self.tokenizer.encode(example["text"])
            token_buffer.extend(tokens)

            while len(token_buffer) >= self.n_ctx:
                chunk = token_buffer[:self.n_ctx]
                token_buffer = token_buffer[self.n_ctx:]
                yield {"input_ids": torch.tensor(chunk, dtype=torch.long)}


class FineWebCached(Dataset):
    """Pre-tokenized FineWeb cache (like SimpleStories cached data)."""

    def __init__(self, path, n_ctx=512, max_samples=None):
        data = torch.load(path, weights_only=True).to(torch.long)[:, :n_ctx]
        if max_samples:
            data = data[:max_samples]
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return {"input_ids": self.data[idx]}


def cache_openwebtext(n_ctx=512, n_val=2000, cache_dir=None):
    """Download Elriggs/openwebtext-100k, tokenize, and cache as 512-token windows."""
    if cache_dir is None:
        cache_dir = Path(__file__).parent / "cached_tokens"
    cache_dir = Path(cache_dir)

    train_path = cache_dir / "owt100k_train.pt"
    val_path = cache_dir / "owt100k_val.pt"

    if train_path.exists() and val_path.exists():
        print(f"OpenWebText cache exists at {cache_dir}")
        return train_path, val_path

    from transformers import AutoTokenizer
    from datasets import load_dataset

    print("Downloading Elriggs/openwebtext-100k...")
    tokenizer = AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-V2-1.25M")
    ds = load_dataset("Elriggs/openwebtext-100k", split="train")

    all_chunks = []
    token_buffer = []

    for i, example in enumerate(ds):
        tokens = tokenizer.encode(example["text"])
        token_buffer.extend(tokens)

        while len(token_buffer) >= n_ctx:
            chunk = token_buffer[:n_ctx]
            token_buffer = token_buffer[n_ctx:]
            all_chunks.append(chunk)

        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/100000 docs, {len(all_chunks)} windows so far...")

    print(f"Total windows: {len(all_chunks)}")

    all_chunks = torch.tensor(all_chunks, dtype=torch.int16)
    # Shuffle and split
    perm = torch.randperm(len(all_chunks))
    all_chunks = all_chunks[perm]
    val_data = all_chunks[:n_val]
    train_data = all_chunks[n_val:]

    torch.save(train_data, train_path)
    torch.save(val_data, val_path)
    print(f"Saved: {train_path} ({train_data.shape}), {val_path} ({val_data.shape})")
    return train_path, val_path


# ---- Techniques ----

def apply_batchnorm(model, d_model, n_head, n_ctx):
    """Replace attention layers with BilinearBatchNorm layers."""
    for i, layer in enumerate(model.layers):
        new_layer = BilinearBatchNorm(
            d_model=d_model, n_head=n_head, n_ctx=n_ctx,
            scale=1.0, use_bias_qkv=True, use_bias_o=True,
        )
        model.layers[i] = new_layer
    for layer in model.layers:
        nn.init.normal_(layer.q.weight, std=0.02)
        nn.init.normal_(layer.k.weight, std=0.02)
        nn.init.normal_(layer.q2.weight, std=0.02)
        nn.init.normal_(layer.k2.weight, std=0.02)
        nn.init.normal_(layer.v.weight, std=0.02)
        nn.init.normal_(layer.o.weight, std=0.01)


def apply_ortho_init(model):
    """Apply orthogonal initialization to all attention weights."""
    for layer in model.layers:
        for attr in ['q', 'k', 'q2', 'k2', 'v', 'o']:
            if hasattr(layer, attr):
                nn.init.orthogonal_(getattr(layer, attr).weight)
                if getattr(layer, attr).bias is not None:
                    nn.init.zeros_(getattr(layer, attr).bias)


# ---- Induction evaluation (FineWeb vocab) ----

def compute_fineweb_token_distribution(data_tensor, top_k=2000):
    """Compute empirical token frequency distribution from cached FineWeb data.

    Returns token IDs and their probabilities, excluding padding (0) and
    very rare tokens. The toy induction task will sample from this distribution
    so that repeated bigrams actually resemble FineWeb bigrams.
    """
    flat = data_tensor.reshape(-1).numpy()
    flat = flat[flat > 0]  # exclude padding
    counts = np.bincount(flat)

    # Get top-k most frequent tokens
    top_ids = np.argsort(counts)[::-1][:top_k]
    top_counts = counts[top_ids].astype(np.float64)
    probs = top_counts / top_counts.sum()

    return top_ids, probs


def make_repeated_sequences_fineweb(token_ids, probs, half_len=256,
                                     n_sequences=100, seed=42):
    """Generate repeated sequences using FineWeb token distribution.

    Each sequence is [A B C ... | A B C ...] where tokens are sampled
    from the empirical FineWeb distribution (not uniform).
    """
    rng = np.random.RandomState(seed)
    seqs = rng.choice(token_ids, size=(n_sequences, half_len), p=probs)
    doubled = np.concatenate([seqs, seqs], axis=1)
    return torch.tensor(doubled, dtype=torch.long), half_len


@torch.no_grad()
def eval_induction_full(model, sequences, half_len, device, batch_size=32):
    """Compute induction score AND toy induction accuracy."""
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


# ---- Main ----

def main(batch_size=64):
    n_layers = 2
    d_model = 768
    n_head = 12
    n_ctx = 512
    vocab_size = 4096  # SimpleStories tokenizer (4019 actual, padded to 4096)
    n_val = 2000

    cfg = {
        "name": "owt100k_2layer_bilinear_batchnorm_orthoinit",
        "seed": 42,
        "model": {
            "vocab_size": vocab_size,
            "n_ctx": n_ctx,
            "d_model": d_model,
            "n_head": n_head,
            "n_layers": n_layers,
            "attn_type": "bilinear",
            "attn_scale": 1.0,
            "rope_base": 10000,
            "norm_type": "layernorm",
            "norm_place": "pre_unembed",
            "use_rmsnorm_qk": False,
            "use_bias_qkv": True,
            "use_bias_o": True,
        },
        "init": {
            "std_embed": 0.02,
            "std_qkv": 0.02,
            "std_o": 0.01,
        },
        "train": {
            "batch_size": batch_size,
            "lr": 3e-4,
            "muon_lr": 0.02,
            "use_muon": True,
            "betas": [0.9, 0.95],
            "weight_decay": 0.1,
            "max_steps": None,  # set after caching (depends on data size)
            "warmup_steps": 1000,
            "lr_decay_frac": 0.1,
            "grad_clip": 1.0,
            "dtype": "bfloat16",
            "debug": True,
            "eval_every": 1000,
            "save_every": None,  # set after caching
        },
        "loss": {
            "type": "next_token_ce",
            "label_smoothing": 0.0,
        },
    }

    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.init()
        torch.cuda.empty_cache()

    # Cache OpenWebText data
    train_path, val_path = cache_openwebtext(n_ctx=n_ctx, n_val=n_val)

    # Compute steps from actual data size
    train_data_size = torch.load(train_path, weights_only=True).shape[0]
    steps_per_epoch = train_data_size // batch_size
    cfg["train"]["max_steps"] = steps_per_epoch
    cfg["train"]["save_every"] = steps_per_epoch

    print("=== 2-Layer Attn-Only Bilinear (batchnorm+ortho_init) on OpenWebText-100k ===")
    print(f"Batch size: {batch_size}, Train windows: {train_data_size}, Steps/epoch: {steps_per_epoch}, Layers: {n_layers}")

    print("Creating dataloaders...")
    train_ds = FineWebCached(train_path, n_ctx=n_ctx)
    val_ds = FineWebCached(val_path, n_ctx=n_ctx, max_samples=1000)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    print("Building model...")
    model = AttentionLM.from_config(cfg)
    apply_batchnorm(model, d_model, n_head, n_ctx)
    apply_ortho_init(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    model = model.to(device)
    model = torch.compile(model)
    print("Model compiled with torch.compile")

    # Build induction test sequences using FineWeb token distribution
    print("Computing FineWeb token distribution for toy induction task...")
    train_data = torch.load(train_path, weights_only=True).to(torch.long)
    token_ids, probs = compute_fineweb_token_distribution(train_data, top_k=2000)
    print(f"  Using top {len(token_ids)} tokens (covers "
          f"{probs.sum()*100:.1f}% of token mass)")

    sequences, half_len = make_repeated_sequences_fineweb(
        token_ids, probs, half_len=256, n_sequences=100,
    )
    print(f"Induction test: {sequences.shape[0]} seqs, half_len={half_len}, "
          f"vocab={len(token_ids)} unique tokens")

    def induction_callback(model, step):
        score, first, second, accuracy = eval_induction_full(
            model, sequences, half_len, device
        )
        print(f"  [step {step}] induction_score={score:.4f} "
              f"1st={first:.4f} 2nd={second:.4f} "
              f"toy_acc={accuracy:.6f} ({accuracy*100:.4f}%)")
        return {
            "induction_score": score,
            "induction_first_half_loss": first,
            "induction_second_half_loss": second,
            "toy_induction_accuracy": accuracy,
        }

    print("Starting training...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        cfg=cfg,
        device=device,
    )
    trainer.eval_callbacks.append(induction_callback)
    trainer.train(
        eval_every=cfg["train"]["eval_every"],
        save_every=cfg["train"]["save_every"],
    )
    print(f"Done. Run dir: {trainer.run_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    main(args.batch_size)
