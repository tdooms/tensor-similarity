#!/usr/bin/env python3
"""Norm sweep experiment: train + evaluate models with different normalization strategies.

Supports:
  - All norms from experiments.norm_sweep.norms (seq_max, causal_seq_max, tok1, etc.)
  - Two datasets: induction (repeated-token) and stories (SimpleStories)
  - Weights & Biases logging
  - Post-hoc swap experiments:
      * Train seq_max → swap to causal_seq_max → compare val CE
      * Train seq_mean with live stats → eval with running stats → compare val CE

Usage (from the bilinear_attn directory):
    # Basic usage with base config + norm type override
    python -m experiments.norm_sweep.run --norm-type seq_max

    # Different dataset
    python -m experiments.norm_sweep.run --norm-type tok1 --dataset stories

    # With wandb
    python -m experiments.norm_sweep.run --norm-type seq_max --wandb

    # With swap experiment (seq_max → causal_seq_max)
    python -m experiments.norm_sweep.run --norm-type seq_max --swap causal_seq_max

    # Custom config file
    python -m experiments.norm_sweep.run --config path/to/custom.yaml --norm-type tok1
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

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.norm_sweep.model import NormSweepLM
from experiments.norm_sweep.norms import (
    make_norm,
    SeqMean,
    Tok1Batch,
    Tok190,
    Tok190Clamp,
    SeqMaxMeanBatch,
    SeqMaxMedianBatch,
    SeqMeanBatch,
    SeqPowerMeanBatch,
)

# Norms that support dual eval modes (batch stats vs running stats)
BATCH_STAT_NORMS = {
    "tok1_batch",
    "tok190",
    "tok190_clamp",
    "seq_max_mean_batch",
    "seq_max_median_batch",
    "seq_mean_batch",
    "seq_power_mean_batch",
}

BATCH_STAT_CLASSES = (
    Tok1Batch,
    Tok190,
    Tok190Clamp,
    SeqMaxMeanBatch,
    SeqMaxMedianBatch,
    SeqMeanBatch,
    SeqPowerMeanBatch,
)
from train.optim import create_optimizer, create_scheduler, Optimizers
from experiments.induction_heads.data import create_repeated_token_dataloaders
from experiments.induction_heads.run import (
    compute_repeated_accuracy as compute_varied_repeat_accuracy,
    evaluate_induction as evaluate_varied_induction,
)
from experiments.tn_sim_induction.score_metrics import compute_repeat_icl_score
from data.cached import create_dataloaders as create_stories_dataloaders


# ── helpers ───────────────────────────────────────────────────────────────────

_DTYPE_MAP = {
    "float32": (torch.float32, False),
    "float16": (torch.float16, True),
    "bfloat16": (torch.bfloat16, False),
}


@torch.no_grad()
def compute_half_repeated_accuracy(logits, input_ids, seq_len):
    """Accuracy on the repeated (second-half) positions (induction dataset)."""
    pred_logits = logits[:, seq_len - 1 : 2 * seq_len - 1, :]
    targets = input_ids[:, seq_len : 2 * seq_len]
    preds = pred_logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


@torch.no_grad()
def evaluate_half_induction(model, dataloader, seq_len, device, max_batches=None):
    """Evaluate loss + repeated-half accuracy (induction dataset)."""
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
        total_acc += compute_half_repeated_accuracy(logits, input_ids, seq_len)
        n += 1
    return total_loss / max(1, n), total_acc / max(1, n)


def _repeat_mask_cross_entropy(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    repeat_masks: torch.Tensor,
    label_smoothing: float = 0.0,
):
    """Cross-entropy over repeated positions indicated by repeat_masks."""
    eval_mask = repeat_masks.clone()
    eval_mask[:, 0] = False

    shifted_logits = logits[:, :-1, :]
    shifted_targets = input_ids[:, 1:]
    shifted_mask = eval_mask[:, 1:]

    if not shifted_mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    flat_logits = shifted_logits[shifted_mask]
    flat_targets = shifted_targets[shifted_mask]
    return torch.nn.functional.cross_entropy(
        flat_logits,
        flat_targets,
        label_smoothing=label_smoothing,
        reduction="mean",
    )


@torch.no_grad()
def evaluate_stories(model, dataloader, device, max_batches=None):
    """Evaluate next-token CE loss (stories dataset)."""
    model.eval()
    total_loss, n = 0.0, 0
    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        logits = model(input_ids)
        # standard next-token prediction: predict t+1 from t
        B, T, V = logits.shape
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].contiguous().view((B * (T - 1)), V),
            input_ids[:, 1:].contiguous().view(B * (T - 1)),
        )
        total_loss += loss.item()
        n += 1
    return total_loss / max(1, n)


def _plot_val_and_icl(metrics_path: Path, output_path: Path):
    if not metrics_path.exists():
        return

    steps = []
    val_losses = []
    icl_values = []

    with open(metrics_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "val_loss" in record:
                steps.append(record.get("step"))
                val_losses.append(record["val_loss"])
                icl_values.append(record.get("repeat_icl"))

    if not steps:
        return

    has_icl = any(v is not None and math.isfinite(v) for v in icl_values)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available; skipping metric plot")
        return

    fig_rows = 2 if has_icl else 1
    fig, axes = plt.subplots(fig_rows, 1, figsize=(8, 4 * fig_rows), sharex=True)
    if fig_rows == 1:
        axes = [axes]

    axes[0].plot(steps, val_losses, marker="o", linewidth=1.5)
    axes[0].set_ylabel("Val CE Loss")
    axes[0].set_title("Validation loss during training")
    axes[0].grid(True, alpha=0.3)

    if has_icl:
        icl_steps = [s for s, v in zip(steps, icl_values) if v is not None and math.isfinite(v)]
        icl_series = [v for v in icl_values if v is not None and math.isfinite(v)]
        axes[1].plot(icl_steps, icl_series, color="tab:red", marker="o", linestyle="--")
        axes[1].set_ylabel("Repeat ICL")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlabel("Training step")
        axes[1].axhline(0.0, color="gray", linestyle=":", alpha=0.4)
    else:
        axes[0].set_xlabel("Training step")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metric plot to {output_path}")


def swap_norms(model: NormSweepLM, new_norm_type: str, norm_kwargs: dict | None = None) -> int:
    """Replace all norm modules in the model with a different norm type (in-place).

    Works for parameter-free norms (seq_max, causal_seq_max, etc.).
    Returns number of modules swapped.
    """
    nk = norm_kwargs or {}
    d = model.d_model
    swapped = 0

    if model.embed_norm is not None and not isinstance(model.embed_norm, torch.nn.Identity):
        model.embed_norm = make_norm(new_norm_type, d, **nk)
        swapped += 1

    if not isinstance(model.final_norm, torch.nn.Identity):
        model.final_norm = make_norm(new_norm_type, d, **nk)
        swapped += 1

    if model.layer_norms is not None:
        for i in range(len(model.layer_norms)):
            model.layer_norms[i] = make_norm(new_norm_type, d, **nk)
            swapped += 1

    return swapped


def force_eval_running_stats(model: NormSweepLM):
    """For batch-stat norms: force eval mode to use running stats."""
    for module in model.modules():
        if isinstance(module, (SeqMean, *BATCH_STAT_CLASSES)):
            if hasattr(module, "use_running_stats"):
                module.use_running_stats = True


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Norm sweep experiment")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML (default: auto-select based on dataset)")
    parser.add_argument("--norm-type", type=str, default=None,
                        help="Norm type to use (overrides config)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="induction",
                        choices=["induction", "varied_induction", "stories"],
                        help="Dataset: 'induction' (half-repeat), 'varied_induction' (variable-gap), or 'stories' (SimpleStories)")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Half-sequence length for induction (default: n_ctx // 2)")
    parser.add_argument("--n-train", type=int, default=50_000)
    parser.add_argument("--n-val", type=int, default=2_000)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--swap", type=str, default=None,
                        help="Post-hoc swap norm type (e.g. causal_seq_max)")
    parser.add_argument("--checkpoint-every", type=int, default=None)
    args = parser.parse_args()

    # ── config ────────────────────────────────────────────────────────────
    # Auto-select config based on dataset if not provided
    if args.config is None:
        args.config = f"experiments/norm_sweep/configs/induction.yaml"
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override norm_type from CLI if provided
    if args.norm_type is not None:
        cfg["model"]["norm_type"] = args.norm_type
        # Set norm_kwargs based on norm type
        if args.norm_type in ("tok1_ghost", "tok1_bn_ghost"):
            cfg["model"]["norm_kwargs"] = {"ghost_frac": 0.5}
        if args.norm_type in ("tok1_bn", "tok1_bn_ghost"):
            cfg["model"]["norm_kwargs"] = cfg["model"].get("norm_kwargs", {})
            cfg["model"]["norm_kwargs"]["momentum"] = 0.1
        if args.norm_type == "seq_mean":
            cfg["model"]["norm_kwargs"] = {"use_running_stats": True, "momentum": 0.1}

    torch.manual_seed(cfg.get("seed", 42))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device == "cuda":
        torch.cuda.init()
        torch.cuda.empty_cache()

    model_cfg = cfg["model"]
    train_cfg = cfg.get("train", {})
    loss_cfg = cfg.get("loss", {})
    norm_type = model_cfg.get("norm_type", "rmsnorm")

    # ── data ──────────────────────────────────────────────────────────────
    dataset_mode = args.dataset
    seq_len = None

    if args.dataset == "induction":
        dataset_mode = "half_induction"
        seq_len = args.seq_len or model_cfg["n_ctx"] // 2
        full_seq_len = 2 * seq_len
        model_cfg["n_ctx"] = full_seq_len
        print(f"Dataset: induction (half={seq_len}, full={full_seq_len})")
        train_dl, val_dl = create_repeated_token_dataloaders(
            vocab_size=model_cfg["vocab_size"],
            n_ctx=full_seq_len,
            batch_size=train_cfg.get("batch_size", 64),
            n_train=args.n_train,
            n_val=args.n_val,
            seed=cfg.get("seed", 42),
        )
    elif args.dataset == "varied_induction":
        dataset_mode = "varied_induction"
        n_ctx = args.seq_len * 2 if args.seq_len is not None else model_cfg["n_ctx"]
        model_cfg["n_ctx"] = n_ctx
        print(f"Dataset: varied induction (n_ctx={n_ctx})")
        train_dl, val_dl = create_repeated_token_dataloaders(
            vocab_size=model_cfg["vocab_size"],
            n_ctx=n_ctx,
            batch_size=train_cfg.get("batch_size", 64),
            n_train=args.n_train,
            n_val=args.n_val,
            seed=cfg.get("seed", 42),
        )
    else:
        dataset_mode = "stories"
        print(f"Dataset: stories (n_ctx={model_cfg['n_ctx']})")
        train_dl, val_dl = create_stories_dataloaders(
            n_ctx=model_cfg["n_ctx"],
            batch_size=train_cfg.get("batch_size", 64),
            max_train_samples=args.n_train if args.n_train != 50_000 else None,
            max_val_samples=args.n_val,
        )

    # ── model ─────────────────────────────────────────────────────────────
    print(f"Building model (norm_type={norm_type}) ...")
    model = NormSweepLM.from_config(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # ── optimizer / scheduler ─────────────────────────────────────────────
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

    # ── run dir ───────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_name = f"{timestamp}_{norm_type}_{args.dataset}"
    run_dir = Path("experiments/norm_sweep/runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    metrics_file = run_dir / "metrics.jsonl"

    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    # ── wandb ─────────────────────────────────────────────────────────────
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
            project="bilinear-norm-sweep",
            name=run_name,
            config={
                **cfg,
                "dataset": args.dataset,
                "n_params": n_params,
                **({"seq_len": seq_len, "full_seq_len": 2 * seq_len} if seq_len else {}),
            },
        )

    # ══════════════════════════════════════════════════════════════════════
    # TRAINING
    # ══════════════════════════════════════════════════════════════════════
    print(f"\nTraining {max_steps} steps | norm_type={norm_type} | dataset={args.dataset}")
    model.train()
    data_iter = iter(train_dl)
    pbar = tqdm(total=max_steps, desc=f"Train ({norm_type})")

    for step in range(1, max_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dl)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        repeat_masks = batch.get("repeat_mask")
        if repeat_masks is not None:
            repeat_masks = repeat_masks.to(device)
        optimizer.zero_grad()

        # ── forward + loss ────────────────────────────────────────────────
        if dataset_mode == "half_induction":
            if use_amp:
                with torch.amp.autocast("cuda", dtype=pt_dtype):
                    logits = model(input_ids)
                    pred_logits = logits[:, seq_len - 1 : 2 * seq_len - 1, :].contiguous()
                    targets = input_ids[:, seq_len : 2 * seq_len].contiguous()
                    B, T, V = pred_logits.shape
                    loss = torch.nn.functional.cross_entropy(
                        pred_logits.view(B * T, V), targets.view(B * T),
                        label_smoothing=label_smoothing,
                    )
            else:
                logits = model(input_ids)
                pred_logits = logits[:, seq_len - 1 : 2 * seq_len - 1, :].contiguous()
                targets = input_ids[:, seq_len : 2 * seq_len].contiguous()
                B, T, V = pred_logits.shape
                loss = torch.nn.functional.cross_entropy(
                    pred_logits.view(B * T, V), targets.view(B * T),
                    label_smoothing=label_smoothing,
                )
        elif dataset_mode == "varied_induction":
            if use_amp:
                with torch.amp.autocast("cuda", dtype=pt_dtype):
                    logits = model(input_ids)
                    loss = _repeat_mask_cross_entropy(
                        logits,
                        input_ids,
                        repeat_masks,
                        label_smoothing=label_smoothing,
                    )
            else:
                logits = model(input_ids)
                loss = _repeat_mask_cross_entropy(
                    logits,
                    input_ids,
                    repeat_masks,
                    label_smoothing=label_smoothing,
                )
        else:  # stories
            if use_amp:
                with torch.amp.autocast("cuda", dtype=pt_dtype):
                    logits = model(input_ids)
                    B, T, V = logits.shape
                    loss = torch.nn.functional.cross_entropy(
                        logits[:, :-1].contiguous().view(B * (T - 1), V),
                        input_ids[:, 1:].contiguous().view(B * (T - 1)),
                        label_smoothing=label_smoothing,
                    )
            else:
                logits = model(input_ids)
                B, T, V = logits.shape
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].contiguous().view(B * (T - 1), V),
                    input_ids[:, 1:].contiguous().view(B * (T - 1)),
                    label_smoothing=label_smoothing,
                )

        # ── backward ──────────────────────────────────────────────────────
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()

        # ── logging ───────────────────────────────────────────────────────
        train_acc = None
        if dataset_mode == "half_induction":
            train_acc = compute_half_repeated_accuracy(logits, input_ids, seq_len)
        elif dataset_mode == "varied_induction" and repeat_masks is not None:
            train_acc = compute_varied_repeat_accuracy(logits, input_ids, repeat_masks)

        if step % 10 == 0:
            row = {
                "step": step,
                "train_loss": loss.item(),
                "lr": scheduler.get_last_lr()[0],
            }
            if train_acc is not None:
                row["train_acc"] = train_acc
            with open(metrics_file, "a") as f:
                f.write(json.dumps(row) + "\n")
            if wandb_run is not None:
                wandb_run.log(row, step=step)

        postfix = {"loss": f"{loss.item():.4f}"}
        if train_acc is not None:
            postfix["acc"] = f"{train_acc:.3f}"
        pbar.set_postfix(**postfix)
        pbar.update(1)

        # ── eval ──────────────────────────────────────────────────────────
        if step % eval_every == 0:
            row = {"step": step}
            if dataset_mode == "half_induction":
                val_loss, val_acc = evaluate_half_induction(model, val_dl, seq_len, device, max_batches=20)
                row.update({"val_loss": val_loss, "val_acc": val_acc})
                print(f"\n[step {step}]  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")
            elif dataset_mode == "varied_induction":
                val_loss, val_acc = evaluate_varied_induction(model, val_dl, device, max_batches=20)
                row.update({"val_loss": val_loss, "val_acc": val_acc})
                icl_metrics = compute_repeat_icl_score(model, val_dl, device=device, max_batches=20)
                row.update(icl_metrics)
                icl_msg = "  ".join(
                    f"{k}={v:.4f}" for k, v in icl_metrics.items() if isinstance(v, (int, float)) and math.isfinite(v)
                )
                print(f"\n[step {step}]  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  {icl_msg}")
            else:
                val_loss = evaluate_stories(model, val_dl, device, max_batches=20)
                row["val_loss"] = val_loss
                print(f"\n[step {step}]  val_loss={val_loss:.4f}")
            with open(metrics_file, "a") as f:
                f.write(json.dumps(row) + "\n")
            if wandb_run is not None:
                wandb_run.log(row, step=step)
            model.train()

        if checkpoint_every > 0 and step % checkpoint_every == 0:
            ckpt_path = run_dir / "checkpoints" / f"step_{step}.pt"
            torch.save({"step": step, "model_state_dict": model.state_dict()}, ckpt_path)

    pbar.close()
    torch.save({"step": max_steps, "model_state_dict": model.state_dict()}, run_dir / "final.pt")

    # ══════════════════════════════════════════════════════════════════════
    # FINAL EVAL (original norm)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Final eval with {norm_type}")
    if dataset_mode == "half_induction":
        orig_loss, orig_acc = evaluate_half_induction(model, val_dl, seq_len, device)
        print(f"  val_loss={orig_loss:.4f}  val_acc={orig_acc:.3f}")
    elif dataset_mode == "varied_induction":
        orig_loss, orig_acc = evaluate_varied_induction(model, val_dl, device)
        icl_metrics = compute_repeat_icl_score(model, val_dl, device=device)
        print(f"  val_loss={orig_loss:.4f}  val_acc={orig_acc:.3f}  repeat_icl={icl_metrics['repeat_icl']:.4f}")
    else:
        orig_loss = evaluate_stories(model, val_dl, device)
        orig_acc = None
        print(f"  val_loss={orig_loss:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # BATCH-STAT NORMS: DUAL EVAL (batch stats vs running stats)
    # ══════════════════════════════════════════════════════════════════════
    if norm_type in BATCH_STAT_NORMS or norm_type == "seq_mean":
        print(f"\n{'='*60}")
        print(f"{norm_type}: comparing eval-as-train (batch stats) vs eval-as-inference (running stats)")

        # Determine which classes to check
        if norm_type == "seq_mean":
            target_classes = (SeqMean,)
        else:
            target_classes = BATCH_STAT_CLASSES

        # Eval-as-train: use batch stats (disable running stats)
        for m in model.modules():
            if isinstance(m, target_classes) and hasattr(m, "use_running_stats"):
                m.use_running_stats = False
        if dataset_mode == "half_induction":
            batch_loss, batch_acc = evaluate_half_induction(model, val_dl, seq_len, device)
            print(f"  Eval-as-train (batch stats):     val_loss={batch_loss:.4f}  val_acc={batch_acc:.3f}")
        elif dataset_mode == "varied_induction":
            batch_loss, batch_acc = evaluate_varied_induction(model, val_dl, device)
            print(f"  Eval-as-train (batch stats):     val_loss={batch_loss:.4f}  val_acc={batch_acc:.3f}")
        else:
            batch_loss = evaluate_stories(model, val_dl, device)
            batch_acc = None
            print(f"  Eval-as-train (batch stats):     val_loss={batch_loss:.4f}")

        # Eval-as-inference: use running stats (enable running stats)
        for m in model.modules():
            if isinstance(m, target_classes) and hasattr(m, "use_running_stats"):
                m.use_running_stats = True
        if dataset_mode == "half_induction":
            run_loss, run_acc = evaluate_half_induction(model, val_dl, seq_len, device)
            print(f"  Eval-as-inference (running stats): val_loss={run_loss:.4f}  val_acc={run_acc:.3f}")
        elif dataset_mode == "varied_induction":
            run_loss, run_acc = evaluate_varied_induction(model, val_dl, device)
            print(f"  Eval-as-inference (running stats): val_loss={run_loss:.4f}  val_acc={run_acc:.3f}")
        else:
            run_loss = evaluate_stories(model, val_dl, device)
            run_acc = None
            print(f"  Eval-as-inference (running stats): val_loss={run_loss:.4f}")

        delta = run_loss - batch_loss
        print(f"  Δ loss (running - batch): {delta:+.4f}")

        if args.swap:
            if dataset_mode == "half_induction":
                swap_loss, swap_acc = evaluate_half_induction(model, val_dl, seq_len, device)
                print(f"  Swap eval ({args.swap}): val_loss={swap_loss:.4f}  val_acc={swap_acc:.3f}")
            elif dataset_mode == "varied_induction":
                swap_loss, swap_acc = evaluate_varied_induction(model, val_dl, device)
                print(f"  Swap eval ({args.swap}): val_loss={swap_loss:.4f}  val_acc={swap_acc:.3f}")
            else:
                swap_loss = evaluate_stories(model, val_dl, device)
                swap_acc = None
                print(f"  Swap eval ({args.swap}): val_loss={swap_loss:.4f}")

        summary_dual = {
            f"{norm_type}_batch_val_loss": batch_loss,
            f"{norm_type}_running_val_loss": run_loss,
            f"{norm_type}_delta_loss": delta,
        }
        if batch_acc is not None:
            summary_dual[f"{norm_type}_batch_val_acc"] = batch_acc
            summary_dual[f"{norm_type}_running_val_acc"] = run_acc
        with open(metrics_file, "a") as f:
            f.write(json.dumps({"phase": "dual_eval_comparison", **summary_dual}) + "\n")
        if wandb_run is not None:
            wandb_run.log({f"comparison/{k}": v for k, v in summary_dual.items()})

    # ══════════════════════════════════════════════════════════════════════
    # POST-HOC SWAP EXPERIMENT
    # ══════════════════════════════════════════════════════════════════════
    if args.swap:
        swap_type = args.swap
        print(f"\n{'='*60}")
        print(f"Swap experiment: {norm_type} → {swap_type}")

        n_swapped = swap_norms(model, swap_type, model_cfg.get("norm_kwargs", {}))
        model = model.to(device)
        print(f"  Swapped {n_swapped} norm modules")

        if dataset_mode == "half_induction":
            swap_loss, swap_acc = evaluate_half_induction(model, val_dl, seq_len, device)
            print(f"  {swap_type:25s} val_loss={swap_loss:.4f}  val_acc={swap_acc:.3f}")
        elif dataset_mode == "varied_induction":
            swap_loss, swap_acc = evaluate_varied_induction(model, val_dl, device)
            print(f"  {swap_type:25s} val_loss={swap_loss:.4f}  val_acc={swap_acc:.3f}")
        else:
            swap_loss = evaluate_stories(model, val_dl, device)
            swap_acc = None
            print(f"  {swap_type:25s} val_loss={swap_loss:.4f}")

        # comparison table
        print(f"\n{'='*60}")
        print(f"  COMPARISON (same weights)")
        print(f"{'='*60}")
        print(f"{'Metric':<20} {norm_type:>18} {swap_type:>18}")
        print(f"{'-'*60}")
        print(f"{'val_loss':<20} {orig_loss:>18.4f} {swap_loss:>18.4f}")
        if orig_acc is not None and swap_acc is not None:
            print(f"{'val_acc':<20} {orig_acc:>18.4f} {swap_acc:>18.4f}")
        delta_loss = swap_loss - orig_loss
        print(f"{'-'*60}")
        print(f"{'Δ loss':<20} {delta_loss:>37.4f}")
        print(f"{'='*60}")

        summary = {
            f"{norm_type}_val_loss": orig_loss,
            f"{swap_type}_val_loss": swap_loss,
            "delta_loss": delta_loss,
        }
        if orig_acc is not None:
            summary[f"{norm_type}_val_acc"] = orig_acc
        if swap_acc is not None:
            summary[f"{swap_type}_val_acc"] = swap_acc
        with open(run_dir / "comparison.json", "w") as f:
            json.dump(summary, f, indent=2)
        with open(metrics_file, "a") as f:
            f.write(json.dumps({"phase": "swap_comparison", **summary}) + "\n")
        if wandb_run is not None:
            wandb_run.log({f"comparison/{k}": v for k, v in summary.items()})

    # ── cleanup ───────────────────────────────────────────────────────────
    if wandb_run is not None:
        wandb_run.finish()

    if dataset_mode == "varied_induction":
        plot_path = run_dir / "metrics_plot.png"
        _plot_val_and_icl(metrics_file, plot_path)

    print(f"\nRun saved to {run_dir}")


if __name__ == "__main__":
    main()
