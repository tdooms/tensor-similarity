#!/usr/bin/env python3
"""Train a tiny bilinear attention model on repeated-token (induction) data.

Saves checkpoints every `checkpoint_every` steps for TN similarity analysis.

Usage (from bilinear_attn directory):
    python -m tn_sim.train --config tn_sim/config_tiny.yaml
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from models import AttentionLM
from experiments.induction_heads.data import create_repeated_token_dataloaders


# ── helpers ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_repeated_accuracy(logits, input_ids, seq_len):
    pred_logits = logits[:, seq_len - 1 : 2 * seq_len - 1, :]
    targets = input_ids[:, seq_len : 2 * seq_len]
    preds = pred_logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


@torch.no_grad()
def evaluate_induction(model, dataloader, seq_len, device, max_batches=None):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        logits = model(input_ids)
        pred_logits = logits[:, seq_len - 1 : 2 * seq_len - 1, :].contiguous()
        targets = input_ids[:, seq_len : 2 * seq_len].contiguous()
        B, T, V = pred_logits.shape
        loss = torch.nn.functional.cross_entropy(
            pred_logits.view(B * T, V), targets.view(B * T)
        )
        total_loss += loss.item()
        total_acc += compute_repeated_accuracy(logits, input_ids, seq_len)
        n += 1
    return total_loss / max(1, n), total_acc / max(1, n)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train tiny model for TN sim experiment")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-train", type=int, default=50_000)
    parser.add_argument("--n-val", type=int, default=2_000)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg.get("seed", 42))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_cfg = cfg["model"]
    train_cfg = cfg.get("train", {})
    loss_cfg = cfg.get("loss", {})

    seq_len = model_cfg["n_ctx"] // 2
    full_seq_len = 2 * seq_len
    model_cfg["n_ctx"] = full_seq_len

    # ── data ──────────────────────────────────────────────────────────────
    print(f"Generating data (half={seq_len}, full={full_seq_len}) ...")
    train_dl, val_dl = create_repeated_token_dataloaders(
        vocab_size=model_cfg["vocab_size"],
        seq_len=seq_len,
        batch_size=train_cfg.get("batch_size", 64),
        n_train=args.n_train,
        n_val=args.n_val,
        seed=cfg.get("seed", 42),
    )

    # ── model ─────────────────────────────────────────────────────────────
    print("Building model ...")
    model = AttentionLM.from_config(cfg)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # ── optimizer (plain AdamW for tiny model) ────────────────────────────
    max_steps = train_cfg.get("max_steps", 10_000)
    warmup_steps = train_cfg.get("warmup_steps", 50)
    lr = train_cfg.get("lr", 3e-4)
    weight_decay = train_cfg.get("weight_decay", 0.1)
    grad_clip = train_cfg.get("grad_clip", 1.0)
    label_smoothing = loss_cfg.get("label_smoothing", 0.0)
    checkpoint_every = train_cfg.get("checkpoint_every", 1000)
    eval_every = train_cfg.get("eval_every", 500)
    lr_decay_frac = train_cfg.get("lr_decay_frac", 0.1)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
        betas=tuple(train_cfg.get("betas", (0.9, 0.95))),
    )

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(progress, 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_decay_frac + coeff * (1.0 - lr_decay_frac)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── run directory ─────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path("tn_sim/runs") / f"{timestamp}_tiny"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    metrics_file = run_dir / "metrics.jsonl"

    # Save step-0 checkpoint (init)
    torch.save(
        {"step": 0, "model_state_dict": model.state_dict()},
        ckpt_dir / "step_00000.pt",
    )

    # ── training loop ─────────────────────────────────────────────────────
    print(f"Training {max_steps} steps, checkpoint every {checkpoint_every} ...")
    model.train()
    data_iter = iter(train_dl)
    pbar = tqdm(total=max_steps, desc="Training")

    for step in range(1, max_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dl)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        optimizer.zero_grad()

        logits = model(input_ids)
        pred_logits = logits[:, seq_len - 1 : 2 * seq_len - 1, :].contiguous()
        targets = input_ids[:, seq_len : 2 * seq_len].contiguous()
        B, T, V = pred_logits.shape
        loss = torch.nn.functional.cross_entropy(
            pred_logits.view(B * T, V), targets.view(B * T),
            label_smoothing=label_smoothing,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        train_acc = compute_repeated_accuracy(logits, input_ids, seq_len)

        if step % 10 == 0:
            row = {"step": step, "train_loss": loss.item(), "train_acc": train_acc,
                   "lr": scheduler.get_last_lr()[0]}
            with open(metrics_file, "a") as f:
                f.write(json.dumps(row) + "\n")

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{train_acc:.3f}")
        pbar.update(1)

        # ── checkpoint ────────────────────────────────────────────────────
        if step % checkpoint_every == 0:
            torch.save(
                {"step": step, "model_state_dict": model.state_dict()},
                ckpt_dir / f"step_{step:05d}.pt",
            )
            print(f"\n  Saved checkpoint at step {step}")

        # ── eval ──────────────────────────────────────────────────────────
        if step % eval_every == 0:
            val_loss, val_acc = evaluate_induction(model, val_dl, seq_len, device, max_batches=20)
            row = {"step": step, "val_loss": val_loss, "val_acc": val_acc}
            with open(metrics_file, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  [step {step}] val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
            model.train()

    pbar.close()

    # Final eval
    val_loss, val_acc = evaluate_induction(model, val_dl, seq_len, device)
    print(f"\nFinal val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    print(f"Run saved to {run_dir}")
    print(f"Checkpoints: {sorted(ckpt_dir.glob('*.pt'))}")


if __name__ == "__main__":
    main()
