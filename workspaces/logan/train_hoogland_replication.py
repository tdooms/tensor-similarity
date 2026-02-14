#!/usr/bin/env python3
"""Replicate Hoogland et al. setup for induction head formation.

Architecture: 2L attention-only, d_model=256, 8 heads, standard softmax
Data: DSIR-filtered Pile (streaming), GPT-2 tokenizer truncated to vocab=5000
Target: ~5B tokens, but stop once induction is clearly learned
Induction eval every 500 steps to catch onset (reported at 6.5k-17k steps).

Usage:
    python train_hoogland_replication.py
"""
import torch, torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset, DataLoader
from lib import AttentionLM, Trainer
from pathlib import Path
import time


N_CTX = 1024
VOCAB_SIZE = 5000
TARGET_TOKENS = 5_000_000_000


# ---- Data: streaming DSIR-filtered Pile with GPT-2 tokenizer, vocab truncated to 5000 ----

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
            # Truncate vocab: tokens >= 5000 get mapped to token_id % vocab_size
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


def cache_pile_data(n_ctx=N_CTX, vocab_size=VOCAB_SIZE, n_val=1000, cache_dir=None):
    """Cache a validation set from the Pile for periodic eval."""
    if cache_dir is None:
        cache_dir = Path(__file__).parent / "cached_tokens"
    cache_dir = Path(cache_dir)
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


# ---- Induction evaluation ----

def make_repeated_sequences(vocab_size, half_len=512, n_sequences=100, seed=42):
    rng = np.random.RandomState(seed)
    seqs = rng.randint(1, vocab_size, size=(n_sequences, half_len))
    doubled = np.concatenate([seqs, seqs], axis=1)
    return torch.tensor(doubled, dtype=torch.long), half_len


@torch.no_grad()
def eval_induction_full(model, sequences, half_len, device, batch_size=16):
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
    d_model = 256
    n_head = 8
    n_layers = 2

    tokens_per_step = batch_size * N_CTX
    max_steps = TARGET_TOKENS // tokens_per_step

    print(f"=== Hoogland et al. Replication ===")
    print(f"2L attn-only, d={d_model}, {n_head} heads, softmax, no QK norm")
    print(f"Data: DSIR-filtered Pile (streaming), GPT-2 tokenizer, vocab={VOCAB_SIZE}")
    print(f"Batch: {batch_size}, n_ctx: {N_CTX}, tokens/step: {tokens_per_step:,}")
    print(f"Max steps: {max_steps:,} ({max_steps * tokens_per_step / 1e9:.1f}B tokens)")

    # Cache val set
    val_path = cache_pile_data(n_ctx=N_CTX, vocab_size=VOCAB_SIZE, n_val=500)

    # Streaming train, cached val
    train_ds = DSIRPileStreaming(n_ctx=N_CTX, vocab_size=VOCAB_SIZE)
    val_ds = CachedDataset(val_path, n_ctx=N_CTX, max_samples=500)
    train_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    cfg = {
        "name": "hoogland_replication",
        "seed": 42,
        "model": {
            "vocab_size": VOCAB_SIZE,
            "n_ctx": N_CTX,
            "d_model": d_model,
            "n_head": n_head,
            "n_layers": n_layers,
            "attn_type": "softmax",
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
            "max_steps": max_steps,
            "warmup_steps": 1000,
            "lr_decay_frac": 0.1,
            "grad_clip": 1.0,
            "dtype": "bfloat16",
            "debug": False,
            "eval_every": 2500,
            "save_every": 10000,
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

    model = AttentionLM.from_config(cfg)
    model = model.to(device)
    model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} (compiled)")

    # Induction test sequences
    sequences, half_len = make_repeated_sequences(VOCAB_SIZE, half_len=512, n_sequences=100)
    print(f"Induction test: {sequences.shape[0]} seqs, half_len={half_len}, vocab={VOCAB_SIZE}")

    induction_found = False

    def induction_callback(model, step):
        nonlocal induction_found
        score, first, second, accuracy = eval_induction_full(
            model, sequences, half_len, device
        )
        tokens_seen = step * tokens_per_step
        print(f"  [step {step}, {tokens_seen/1e9:.2f}B tokens] "
              f"induction_score={score:.4f} "
              f"1st={first:.4f} 2nd={second:.4f} "
              f"toy_acc={accuracy:.6f} ({accuracy*100:.4f}%)")

        if accuracy > 0.10:  # 10% = clear induction signal
            induction_found = True
            print(f"\n  *** INDUCTION FOUND at step {step} ({tokens_seen/1e9:.2f}B tokens)! ***\n")

        return {
            "induction_score": score,
            "induction_first_half_loss": first,
            "induction_second_half_loss": second,
            "toy_induction_accuracy": accuracy,
            "tokens_seen_B": tokens_seen / 1e9,
        }

    # Custom training loop with sparse logging and induction-based early stopping
    from lib import compute_loss, evaluate
    from lib.optim import create_optimizer, create_scheduler, Optimizers, _is_muon_param
    from tqdm import tqdm
    from datetime import datetime
    import json

    model.train()
    train_cfg = cfg["train"]

    opt_result = create_optimizer(
        model,
        lr=train_cfg["lr"],
        muon_lr=train_cfg["muon_lr"],
        weight_decay=train_cfg["weight_decay"],
        betas=tuple(train_cfg["betas"]),
        use_muon=train_cfg["use_muon"],
    )
    if isinstance(opt_result, Optimizers):
        optimizer = opt_result.muon
    else:
        optimizer = opt_result

    scheduler = create_scheduler(
        optimizer,
        warmup_steps=train_cfg["warmup_steps"],
        max_steps=max_steps,
        lr_decay_frac=train_cfg["lr_decay_frac"],
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(f"runs/{timestamp}_{cfg['name']}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    metrics_file = run_dir / "metrics.jsonl"

    def log(d, step):
        d["step"] = step
        with open(metrics_file, "a") as f:
            f.write(json.dumps(d) + "\n")

    step = 0
    pbar = tqdm(total=max_steps, desc="Hoogland replication")

    for batch in train_dl:
        if step >= max_steps or induction_found:
            break

        input_ids = batch["input_ids"].to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = compute_loss(logits, input_ids, loss_type="next_token_ce")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
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

            cb_metrics = induction_callback(model, step)
            log(cb_metrics, step)
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
        cb_metrics = induction_callback(model, step)
        log({"val_loss": val_loss}, step)
        log(cb_metrics, step)

    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
    }, run_dir / "checkpoints" / "final.pt")

    print(f"Done at step {step}. Run dir: {run_dir}")
    if induction_found:
        print(f"Induction was found! Stopping early.")
    else:
        print(f"Induction NOT found after {step} steps ({step * tokens_per_step / 1e9:.1f}B tokens)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    main(args.batch_size)
