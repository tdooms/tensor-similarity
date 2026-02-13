#!/usr/bin/env python3
"""Train a model on repeated-token sequences and measure induction accuracy.

Usage (from the bilinear_attn directory):
    python -m experiments.induction_heads.run --config configs/main256.yaml
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
    seq_len: int,
) -> float:
    """Compute next-token accuracy on the *repeated* (second-half) positions.

    For a sequence ``[A B C | A B C]`` of length ``2*seq_len``, the model
    should predict each token in the second half given the context so far.
    We measure accuracy at positions ``seq_len`` through ``2*seq_len - 1``
    (i.e. predicting tokens ``input_ids[:, seq_len : 2*seq_len]``).

    Args:
        logits: (B, T, V) model output.
        input_ids: (B, T) ground-truth token ids.
        seq_len: Length of each half.

    Returns:
        Fraction of correctly predicted tokens in the repeated half.
    """
    # Predictions for position t are in logits[:, t-1, :]
    # We want predictions for positions seq_len .. 2*seq_len-1
    pred_logits = logits[:, seq_len - 1 : 2 * seq_len - 1, :]  # (B, seq_len, V)
    targets = input_ids[:, seq_len : 2 * seq_len]                # (B, seq_len)

    preds = pred_logits.argmax(dim=-1)  # (B, seq_len)
    correct = (preds == targets).float().mean().item()
    return correct


@torch.no_grad()
def evaluate_induction(model, dataloader, seq_len, device, max_batches=None):
    """Evaluate loss and repeated-half accuracy over a dataloader."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        logits = model(input_ids)

        # CE loss on repeated half only
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
    parser = argparse.ArgumentParser(description="Induction head experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Half-sequence length (default: n_ctx // 2)")
    parser.add_argument("--n-train", type=int, default=50_000)
    parser.add_argument("--n-val", type=int, default=2_000)
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

    seq_len = args.seq_len or model_cfg["n_ctx"] // 2
    full_seq_len = 2 * seq_len

    # Override n_ctx so the model's positional encoding covers the full seq
    model_cfg["n_ctx"] = full_seq_len

    # ── data ─────────────────────────────────────────────────────────────
    print(f"Generating repeated-token data  (half={seq_len}, full={full_seq_len}) ...")
    train_dl, val_dl = create_repeated_token_dataloaders(
        vocab_size=model_cfg["vocab_size"],
        seq_len=seq_len,
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

    # ── run dir ──────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path("experiments/induction_heads/runs") / f"{timestamp}_induction"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = run_dir / "metrics.jsonl"

    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

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
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast("cuda", dtype=pt_dtype):
                logits = model(input_ids)
                # CE loss on repeated half only
                pred_logits = logits[:, seq_len - 1 : 2 * seq_len - 1, :].contiguous()
                targets = input_ids[:, seq_len : 2 * seq_len].contiguous()
                B, T, V = pred_logits.shape
                loss = torch.nn.functional.cross_entropy(
                    pred_logits.view(B * T, V),
                    targets.view(B * T),
                    label_smoothing=label_smoothing,
                )
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
            pred_logits = logits[:, seq_len - 1 : 2 * seq_len - 1, :].contiguous()
            targets = input_ids[:, seq_len : 2 * seq_len].contiguous()
            B, T, V = pred_logits.shape
            loss = torch.nn.functional.cross_entropy(
                pred_logits.view(B * T, V),
                targets.view(B * T),
                label_smoothing=label_smoothing,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()

        # ── logging ──────────────────────────────────────────────────────
        train_acc = compute_repeated_accuracy(logits, input_ids, seq_len)

        if step % 10 == 0:
            row = {
                "step": step,
                "train_loss": loss.item(),
                "train_acc": train_acc,
                "lr": scheduler.get_last_lr()[0],
            }
            with open(metrics_file, "a") as f:
                f.write(json.dumps(row) + "\n")

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{train_acc:.3f}")
        pbar.update(1)

        # ── eval ─────────────────────────────────────────────────────────
        if step % eval_every == 0:
            val_loss, val_acc = evaluate_induction(
                model, val_dl, seq_len, device, max_batches=20
            )
            row = {"step": step, "val_loss": val_loss, "val_acc": val_acc}
            with open(metrics_file, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"\n[step {step}]  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")
            model.train()

    pbar.close()

    # ── save final checkpoint ────────────────────────────────────────────
    torch.save(
        {"step": max_steps, "model_state_dict": model.state_dict()},
        run_dir / "final.pt",
    )

    # ── final eval ───────────────────────────────────────────────────────
    val_loss, val_acc = evaluate_induction(model, val_dl, seq_len, device)
    print(f"\nFinal  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")
    row = {"step": max_steps, "final_val_loss": val_loss, "final_val_acc": val_acc}
    with open(metrics_file, "a") as f:
        f.write(json.dumps(row) + "\n")

    print(f"Run saved to {run_dir}")


if __name__ == "__main__":
    main()
