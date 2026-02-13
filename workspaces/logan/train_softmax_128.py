#!/usr/bin/env python3
"""Train 2-layer attn-only softmax (QK norm) on SimpleStories, n_ctx=128.

Two modes:
  --mode truncate   : take first 128 tokens of each story (waste the rest)
  --mode concat     : concatenate stories end-to-end, then chunk into 128-token windows

Usage:
    python train_softmax_128.py --mode truncate
    python train_softmax_128.py --mode concat
"""
import torch, torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from lib import AttentionLM, Trainer
from pathlib import Path


CACHE_DIR = Path(__file__).parent / "cached_tokens"
N_CTX = 128


class TruncatedDataset(Dataset):
    """Take first n_ctx tokens of each story."""
    def __init__(self, split="train", n_ctx=N_CTX, max_samples=None):
        path = CACHE_DIR / f"{split}_perstory.pt"
        data = torch.load(path, weights_only=True).to(torch.long)[:, :n_ctx]
        if max_samples:
            data = data[:max_samples]
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return {"input_ids": self.data[idx]}


class ConcatDataset(Dataset):
    """Concatenate all stories, then chunk into n_ctx windows."""
    def __init__(self, split="train", n_ctx=N_CTX, max_samples=None):
        path = CACHE_DIR / f"{split}_perstory.pt"
        data = torch.load(path, weights_only=True).to(torch.long)
        # Flatten, remove padding zeros
        flat = data.reshape(-1)
        flat = flat[flat > 0]
        # Chunk into n_ctx windows
        n_windows = len(flat) // n_ctx
        self.data = flat[:n_windows * n_ctx].reshape(n_windows, n_ctx)
        if max_samples:
            self.data = self.data[:max_samples]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return {"input_ids": self.data[idx]}


# ---- Induction evaluation ----

def make_repeated_sequences(vocab_size, half_len=64, n_sequences=100, seed=42):
    """half_len=64 so full seq = 128 = n_ctx."""
    rng = np.random.RandomState(seed)
    seqs = rng.randint(1, vocab_size, size=(n_sequences, half_len))
    doubled = np.concatenate([seqs, seqs], axis=1)
    return torch.tensor(doubled, dtype=torch.long), half_len


@torch.no_grad()
def eval_induction_full(model, sequences, half_len, device, batch_size=32):
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

def main(batch_size=64, mode="truncate"):
    n_layers = 2
    d_model = 768
    n_head = 12
    vocab_size = 4096

    # Build datasets
    if mode == "truncate":
        train_ds = TruncatedDataset("train", n_ctx=N_CTX)
        val_ds = TruncatedDataset("test", n_ctx=N_CTX, max_samples=1000)
    else:
        train_ds = ConcatDataset("train", n_ctx=N_CTX)
        val_ds = ConcatDataset("test", n_ctx=N_CTX, max_samples=1000)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    steps_per_epoch = len(train_ds) // batch_size
    print(f"Mode: {mode}, Train samples: {len(train_ds)}, Steps/epoch: {steps_per_epoch}")

    cfg = {
        "name": f"ss_2layer_softmax_qknorm_128_{mode}",
        "seed": 42,
        "model": {
            "vocab_size": vocab_size,
            "n_ctx": N_CTX,
            "d_model": d_model,
            "n_head": n_head,
            "n_layers": n_layers,
            "attn_type": "softmax",
            "attn_scale": 1.0,
            "rope_base": 10000,
            "norm_type": "layernorm",
            "norm_place": "pre_unembed",
            "use_rmsnorm_qk": True,
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
            "max_steps": steps_per_epoch,
            "warmup_steps": 500,
            "lr_decay_frac": 0.1,
            "grad_clip": 1.0,
            "dtype": "bfloat16",
            "debug": True,
            "eval_every": 1000,
            "save_every": steps_per_epoch,
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

    print(f"=== 2-Layer Softmax (QK RMSNorm) on SimpleStories, n_ctx={N_CTX}, mode={mode} ===")

    model = AttentionLM.from_config(cfg)
    model = model.to(device)
    model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} (compiled)")

    # Induction sequences (half_len=64 so full=128)
    sequences, half_len = make_repeated_sequences(vocab_size, half_len=64, n_sequences=100)
    print(f"Induction test: {sequences.shape[0]} seqs, half_len={half_len}")

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
    parser.add_argument("--mode", choices=["truncate", "concat"], required=True)
    args = parser.parse_args()
    main(args.batch_size, args.mode)
