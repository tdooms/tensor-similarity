#!/usr/bin/env python3
"""Train 4-layer bilinear attention-only model with batchnorm + ortho_init.

Combines both techniques: BilinearBatchNorm layers with orthogonal initialization.

Tracks during training:
  - val loss (CE)
  - induction score (first_half_loss - second_half_loss on repeated seqs)
  - toy induction accuracy (argmax accuracy on second half of repeated seqs)

Usage:
    python train_4layer.py
"""
import math, torch, torch.nn as nn
import numpy as np
import torch.nn.functional as F
from lib import AttentionLM, create_dataloaders, Trainer
from lib.attention import BilinearAttention
from train_induction_static_norm import BilinearBatchNorm
from einops import rearrange, einsum


# ---- Technique implementations ----

def apply_batchnorm(model, d_model, n_head, n_ctx):
    """Replace attention layers with BilinearBatchNorm layers."""
    for i, layer in enumerate(model.layers):
        new_layer = BilinearBatchNorm(
            d_model=d_model, n_head=n_head, n_ctx=n_ctx,
            scale=1.0, use_bias_qkv=True, use_bias_o=True,
        )
        model.layers[i] = new_layer
    # Re-initialize
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


# ---- Induction evaluation ----

def make_repeated_sequences(vocab_size, half_len=256, n_sequences=100, seed=42):
    rng = np.random.RandomState(seed)
    seqs = rng.randint(1, vocab_size, size=(n_sequences, half_len))
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

        logits = model(batch)  # (B, 2L, V)
        shift_logits = logits[:, :-1, :]
        shift_labels = batch[:, 1:]
        B, T, V = shift_logits.shape

        # Per-position loss
        loss = F.cross_entropy(
            shift_logits.reshape(B * T, V), shift_labels.reshape(B * T),
            reduction="none",
        ).reshape(B, T)
        all_losses.append(loss.cpu())

        # Accuracy on second half (induction positions)
        # logits[:, half_len..2*half_len-2, :] predicts tokens at half_len+1..2*half_len-1
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

def main(batch_size=32):
    n_layers = 4
    steps_per_epoch = 66116 if batch_size == 32 else 33058

    cfg = {
        "name": "4layer_bilinear_batchnorm_orthoinit",
        "seed": 42,
        "model": {
            "vocab_size": 4096,
            "n_ctx": 512,
            "d_model": 768,
            "n_head": 12,
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
            "max_steps": steps_per_epoch,
            "warmup_steps": 1000,
            "lr_decay_frac": 0.1,
            "grad_clip": 1.0,
            "dtype": "bfloat16",
            "debug": True,
            "eval_every": 5000,
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

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    print(f"=== 4-Layer Bilinear Attention with batchnorm + ortho_init ===")
    print(f"Batch size: {batch_size}, Steps: {steps_per_epoch}, Layers: {n_layers}")

    print("Creating dataloaders...")
    train_dl, val_dl = create_dataloaders(
        n_ctx=model_cfg["n_ctx"],
        batch_size=train_cfg["batch_size"],
        max_val_samples=1000,
    )

    print("Building model...")
    model = AttentionLM.from_config(cfg)

    # Apply batchnorm + ortho_init combined
    print("Applying batchnorm + ortho_init...")
    apply_batchnorm(model, model_cfg["d_model"], model_cfg["n_head"], model_cfg["n_ctx"])
    apply_ortho_init(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    model = model.to(device)
    model = torch.compile(model)
    print("Model compiled with torch.compile")

    # Induction test sequences (4096 vocab)
    sequences, half_len = make_repeated_sequences(
        vocab_size=model_cfg["vocab_size"], half_len=256, n_sequences=100
    )
    print(f"Induction test: {sequences.shape[0]} seqs, half_len={half_len}, vocab={model_cfg['vocab_size']}")

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
        eval_every=train_cfg["eval_every"],
        save_every=train_cfg["save_every"],
    )
    print(f"Done. Run dir: {trainer.run_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    main(args.batch_size)
