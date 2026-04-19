#!/usr/bin/env python3
"""Train models with different Q/K normalization strategies.

Usage (from the bilinear_attn directory):
    python -m experiments.norms.run --config experiments/norms/configs/alpha_head.yaml
    python -m experiments.norms.run --config experiments/norms/configs/rmsnorm_qk.yaml --wandb
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.norms.model import AttentionLMNorm
from train.optim import create_optimizer, create_scheduler, Optimizers
from experiments.induction_heads.data import create_repeated_token_dataloaders


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
    """Accuracy on the repeated (second-half) positions."""
    pred_logits = logits[:, seq_len - 1 : 2 * seq_len - 1, :]
    targets = input_ids[:, seq_len : 2 * seq_len]
    preds = pred_logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


@torch.no_grad()
def evaluate(model, dataloader, seq_len, device, max_batches=None):
    """Evaluate loss and repeated-half accuracy."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
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


def main():
    parser = argparse.ArgumentParser(description="Q/K normalization experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Half-sequence length (default: n_ctx // 2)")
    parser.add_argument("--n-train", type=int, default=50_000)
    parser.add_argument("--n-val", type=int, default=2_000)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--checkpoint-every", type=int, default=None,
                        help="Save checkpoint every N steps")
    args = parser.parse_args()

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
    model_cfg["n_ctx"] = full_seq_len

    print(f"Generating repeated-token data (half={seq_len}, full={full_seq_len}) ...")
    train_dl, val_dl = create_repeated_token_dataloaders(
        vocab_size=model_cfg["vocab_size"],
        seq_len=seq_len,
        batch_size=train_cfg.get("batch_size", 64),
        n_train=args.n_train,
        n_val=args.n_val,
        seed=cfg.get("seed", 42),
    )

    qk_norm_type = model_cfg.get("qk_norm_type", "none")
    print(f"Building model (qk_norm_type={qk_norm_type}) ...")
    model = AttentionLMNorm.from_config(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    max_steps = train_cfg.get("max_steps", 1000)
    warmup_steps = train_cfg.get("warmup_steps", 50)
    lr_decay_frac = train_cfg.get("lr_decay_frac", 0.1)
    grad_clip = train_cfg.get("grad_clip", 1.0)
    label_smoothing = loss_cfg.get("label_smoothing", 0.0)
    eval_every = train_cfg.get("eval_every", 100)
    checkpoint_every = args.checkpoint_every or train_cfg.get("checkpoint_every", 0)

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

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_name = f"{timestamp}_{qk_norm_type}"
    run_dir = Path("experiments/norms/runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    metrics_file = run_dir / "metrics.jsonl"

    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

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
            project="bilinear-qk-norms",
            name=run_name,
            config={**cfg, "seq_len": seq_len, "full_seq_len": full_seq_len, "n_params": n_params},
        )

    print(f"\nTraining for {max_steps} steps with qk_norm_type={qk_norm_type} ...")
    model.train()
    data_iter = iter(train_dl)
    pbar = tqdm(total=max_steps, desc=f"Training ({qk_norm_type})")

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
            if wandb_run is not None:
                wandb_run.log(row, step=step)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{train_acc:.3f}")
        pbar.update(1)

        if step % eval_every == 0:
            val_loss, val_acc = evaluate(model, val_dl, seq_len, device, max_batches=20)
            row = {"step": step, "val_loss": val_loss, "val_acc": val_acc}
            with open(metrics_file, "a") as f:
                f.write(json.dumps(row) + "\n")
            if wandb_run is not None:
                wandb_run.log(row, step=step)
            tqdm.write(f"[step {step}]  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")
            model.train()

        if checkpoint_every > 0 and step % checkpoint_every == 0:
            ckpt_path = run_dir / "checkpoints" / f"step_{step}.pt"
            torch.save({"step": step, "model_state_dict": model.state_dict()}, ckpt_path)

    pbar.close()

    torch.save(
        {"step": max_steps, "model_state_dict": model.state_dict()},
        run_dir / "final.pt",
    )

    val_loss, val_acc = evaluate(model, val_dl, seq_len, device)
    print(f"\nFinal  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")
    row = {"step": max_steps, "final_val_loss": val_loss, "final_val_acc": val_acc}
    with open(metrics_file, "a") as f:
        f.write(json.dumps(row) + "\n")

    if wandb_run is not None:
        wandb_run.finish()

    print(f"Run saved to {run_dir}")


if __name__ == "__main__":
    main()
