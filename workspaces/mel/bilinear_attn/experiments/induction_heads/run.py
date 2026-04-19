#!/usr/bin/env python3
"""Train a model on repeated-token sequences and measure induction accuracy.

Usage (from the bilinear_attn directory):
    python -m experiments.induction_heads.run --config configs/main256.yaml
"""

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from models import AttentionLM
from train.optim import create_optimizer, create_scheduler, Optimizers
from experiments.induction_heads.data import create_repeated_token_dataloaders


# ── helpers ──────────────────────────────────────────────────────────────────

_DTYPE_MAP = {
    "float32": (torch.float32, False),
    "float16": (torch.float16, True),
    "bfloat16": (torch.bfloat16, False),
}


@torch.no_grad()
def compute_repeated_accuracy(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    repeat_masks: torch.Tensor,
) -> float:
    """Compute next-token accuracy on repeated positions only (vectorized).

    Args:
        logits: (B, T, V) model output.
        input_ids: (B, T) ground-truth token ids.
        repeat_masks: (B, T) bool mask indicating repeated positions to evaluate.

    Returns:
        Fraction of correctly predicted tokens at repeated positions.
    """
    B, T = input_ids.shape
    
    # Shift mask: we predict position t from logits[t-1]
    # So we need mask at positions 1..T (can't predict position 0)
    eval_mask = repeat_masks.clone()
    eval_mask[:, 0] = False  # Can't predict position 0
    
    if not eval_mask.any():
        return 0.0
    
    # Get predictions: argmax over vocab dimension
    preds = logits.argmax(dim=-1)  # (B, T)
    
    # Shift predictions: pred at t-1 predicts token at t
    # So we compare preds[:, :-1] with input_ids[:, 1:]
    shifted_preds = preds[:, :-1]  # (B, T-1)
    shifted_targets = input_ids[:, 1:]  # (B, T-1)
    shifted_mask = eval_mask[:, 1:]  # (B, T-1)
    
    # Compute accuracy only on masked positions
    correct = (shifted_preds == shifted_targets) & shifted_mask
    total_correct = correct.sum().item()
    total_tokens = shifted_mask.sum().item()
    
    return total_correct / max(1, total_tokens)


@torch.no_grad()
def evaluate_induction(model, dataloader, device, max_batches=None):
    """Evaluate loss and accuracy on repeated positions only (vectorized)."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        repeat_masks = batch["repeat_mask"].to(device)
        logits = model(input_ids)

        B, T, V = logits.shape
        
        # Shift mask: predict position t from logits[t-1]
        eval_mask = repeat_masks.clone()
        eval_mask[:, 0] = False  # Can't predict position 0
        
        if eval_mask.any():
            # Shift for prediction
            shifted_logits = logits[:, :-1, :]  # (B, T-1, V)
            shifted_targets = input_ids[:, 1:]  # (B, T-1)
            shifted_mask = eval_mask[:, 1:]  # (B, T-1)
            
            # Flatten and filter by mask
            flat_logits = shifted_logits[shifted_mask]  # (N, V) where N = num masked positions
            flat_targets = shifted_targets[shifted_mask]  # (N,)
            
            if len(flat_targets) > 0:
                # Compute loss on masked positions only
                batch_loss = torch.nn.functional.cross_entropy(
                    flat_logits, flat_targets, reduction='sum'
                ).item()
                
                total_loss += batch_loss / len(flat_targets)
                total_acc += compute_repeated_accuracy(logits, input_ids, repeat_masks)
                n += 1

    return total_loss / max(1, n), total_acc / max(1, n)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Induction head experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Half-sequence length (default: n_ctx // 2)")
    parser.add_argument("--n-train", type=int, default=50_000)
    parser.add_argument("--n-val", type=int, default=2_000)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--checkpoint-every", type=int, default=None,
                        help="Save checkpoint every N steps (default: no periodic checkpoints)")
    args = parser.parse_args()

    # ── config ───────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg.get("seed", 42))


    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device == "cuda":
        torch.cuda.init()
        torch.cuda.empty_cache()

    model_cfg = cfg["model"]
    train_cfg = cfg.get("train", {})
    loss_cfg = cfg.get("loss", {})

    n_ctx = model_cfg["n_ctx"]

    # ── data ─────────────────────────────────────────────────────────────
    print(f"Generating variable-gap repeated-token data (n_ctx={n_ctx}) ...")
    print(f"  Subsequences with variable lengths and gaps to prevent RoPE shortcuts")
    train_dl, val_dl = create_repeated_token_dataloaders(
        vocab_size=model_cfg["vocab_size"],
        n_ctx=n_ctx,
        batch_size=train_cfg.get("batch_size", 64),
        n_train=args.n_train,
        n_val=args.n_val,
        seed=cfg.get("seed", 42),
    )

    # ── model ────────────────────────────────────────────────────────────
    print("Building model ...")
    model = AttentionLM.from_config(cfg)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ── optimizer / scheduler ────────────────────────────────────────────
    max_steps = train_cfg.get("max_steps", 1000)
    warmup_steps = train_cfg.get("warmup_steps", 50)
    lr_decay_frac = train_cfg.get("lr_decay_frac", 0.1)
    grad_clip = train_cfg.get("grad_clip", 1.0)
    label_smoothing = loss_cfg.get("label_smoothing", 0.0)

    opt_result = create_optimizer(
        model,
        lr=train_cfg.get("lr", 3e-4),
        muon_lr=train_cfg.get("muon_lr", 0.02),
        weight_decay=train_cfg.get("weight_decay", 0.1),
        betas=tuple(train_cfg.get("betas", (0.9, 0.95))),
        use_muon=train_cfg.get("use_muon", True),
    )
    optimizer = opt_result.muon if isinstance(opt_result, Optimizers) else opt_result

    scheduler = create_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        lr_decay_frac=lr_decay_frac,
    )

    dtype_str = train_cfg.get("dtype", "float32")
    pt_dtype, use_scaler = _DTYPE_MAP[dtype_str]
    use_amp = dtype_str != "float32" and device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if (use_scaler and device == "cuda") else None

    eval_every = train_cfg.get("eval_every", 100)
    checkpoint_every = args.checkpoint_every or train_cfg.get("checkpoint_every", 0)

    # ── run dir ──────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_name = cfg.get('name', 'induction')
    run_dir = Path("experiments/induction_heads/runs") / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = run_dir / "metrics.jsonl"

    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    # ── wandb ────────────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb:
        import wandb

        wandb_entity = os.environ.get("WANDB_ENTITY")
        if not wandb_entity:
            raise RuntimeError(
                "WANDB_ENTITY environment variable must be set when using --wandb"
            )

        wandb_run = wandb.init(
            entity=wandb_entity,
            project="bilinear-induction-heads",
            name=run_dir.name,
            config={**cfg, "n_params": n_params},
        )

    (run_dir / "checkpoints").mkdir(exist_ok=True)

    # Save initial weights as step_0.pt
    ckpt_path_0 = run_dir / "checkpoints" / "step_0.pt"
    torch.save({"step": 0, "model_state_dict": model.state_dict()}, ckpt_path_0)
    print(f"Saved initial weights to {ckpt_path_0}")

    # ── training loop ────────────────────────────────────────────────────
    print(f"Training for {max_steps} steps  (eval every {eval_every}) ...")
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
        repeat_masks = batch["repeat_mask"].to(device)
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast("cuda", dtype=pt_dtype):
                logits = model(input_ids)
                B, T, V = logits.shape
                
                # Vectorized loss computation
                eval_mask = repeat_masks.clone()
                eval_mask[:, 0] = False  # Can't predict position 0
                
                # Shift for prediction
                shifted_logits = logits[:, :-1, :]  # (B, T-1, V)
                shifted_targets = input_ids[:, 1:]  # (B, T-1)
                shifted_mask = eval_mask[:, 1:]  # (B, T-1)
                
                # Flatten and filter by mask
                flat_logits = shifted_logits[shifted_mask]  # (N, V)
                flat_targets = shifted_targets[shifted_mask]  # (N,)
                
                if len(flat_targets) > 0:
                    loss = torch.nn.functional.cross_entropy(
                        flat_logits, flat_targets,
                        label_smoothing=label_smoothing,
                        reduction='mean'
                    )
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        else:
            logits = model(input_ids)
            B, T, V = logits.shape
            
            # Vectorized loss computation
            eval_mask = repeat_masks.clone()
            eval_mask[:, 0] = False  # Can't predict position 0
            
            # Shift for prediction
            shifted_logits = logits[:, :-1, :]  # (B, T-1, V)
            shifted_targets = input_ids[:, 1:]  # (B, T-1)
            shifted_mask = eval_mask[:, 1:]  # (B, T-1)
            
            # Flatten and filter by mask
            flat_logits = shifted_logits[shifted_mask]  # (N, V)
            flat_targets = shifted_targets[shifted_mask]  # (N,)
            
            if len(flat_targets) > 0:
                loss = torch.nn.functional.cross_entropy(
                    flat_logits, flat_targets,
                    label_smoothing=label_smoothing,
                    reduction='mean'
                )
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()

        # ── logging ──────────────────────────────────────────────────────
        train_acc = compute_repeated_accuracy(logits, input_ids, repeat_masks)

        if step % 10 == 0:
            row = {
                "step": step,
                "train_loss": loss.item(),
                "train_acc": train_acc,
                "lr": scheduler.get_last_lr()[0],
            }
            with open(metrics_file, "a") as f:
                f.write(json.dumps(row) + "\n")
            if wandb_run is not None:
                wandb_run.log(row, step=step)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{train_acc:.3f}")
        pbar.update(1)

        # ── eval ─────────────────────────────────────────────────────────
        if step % eval_every == 0:
            val_loss, val_acc = evaluate_induction(
                model, val_dl, device, max_batches=20
            )
            row = {"step": step, "val_loss": val_loss, "val_acc": val_acc}
            with open(metrics_file, "a") as f:
                f.write(json.dumps(row) + "\n")
            if wandb_run is not None:
                wandb_run.log(row, step=step)
            tqdm.write(f"[step {step}]  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")
            model.train()

        # ── checkpoint ────────────────────────────────────────────────
        if checkpoint_every > 0 and step % checkpoint_every == 0:
            ckpt_path = run_dir / "checkpoints" / f"step_{step}.pt"
            torch.save({"step": step, "model_state_dict": model.state_dict()}, ckpt_path)

    pbar.close()

    # ── save final checkpoint ────────────────────────────────────────────
    torch.save(
        {"step": max_steps, "model_state_dict": model.state_dict()},
        run_dir / "final.pt",
    )

    # ── final eval ───────────────────────────────────────────────────────
    val_loss, val_acc = evaluate_induction(model, val_dl, device)
    print(f"\nFinal  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")
    row = {"step": max_steps, "final_val_loss": val_loss, "final_val_acc": val_acc}
    with open(metrics_file, "a") as f:
        f.write(json.dumps(row) + "\n")

    if wandb_run is not None:
        wandb_run.finish()

    print(f"Run saved to {run_dir}")


if __name__ == "__main__":
    main()
