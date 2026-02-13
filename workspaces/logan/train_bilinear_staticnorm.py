#!/usr/bin/env python3
"""Train bilinear attention with static normalization on SimpleStories for 3 epochs.

Usage:
    python train_bilinear_staticnorm.py --strategy batchnorm
    python train_bilinear_staticnorm.py --strategy specnorm
    python train_bilinear_staticnorm.py --strategy scoreclamp
"""
import yaml
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from lib import AttentionLM, create_dataloaders, Trainer
from train_induction_static_norm import BilinearBatchNorm, BilinearScoreClamp, apply_spectral_norm


def make_repeated_sequences(vocab_size, half_len=256, n_sequences=50, seed=42):
    rng = np.random.RandomState(seed)
    seqs = rng.randint(1, vocab_size, size=(n_sequences, half_len))
    doubled = np.concatenate([seqs, seqs], axis=1)
    return torch.tensor(doubled, dtype=torch.long), half_len


@torch.no_grad()
def eval_induction(model, sequences, half_len, device):
    model.eval()
    batch = sequences.to(device)
    logits = model(batch)
    shift_logits = logits[:, :-1, :]
    shift_labels = batch[:, 1:]
    B, T, V = shift_logits.shape
    loss = F.cross_entropy(
        shift_logits.reshape(B * T, V),
        shift_labels.reshape(B * T),
        reduction="none",
    ).reshape(B, T)
    mean_loss = loss.mean(dim=0).cpu().numpy()
    first_half = mean_loss[:half_len - 1].mean()
    second_half = mean_loss[half_len:].mean()
    return float(first_half - second_half), float(first_half), float(second_half)


def apply_strategy(model, strategy, d_model, n_head, n_ctx):
    """Apply static normalization strategy to the model's attention layers."""
    if strategy == "batchnorm":
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
    elif strategy == "specnorm":
        apply_spectral_norm(model)
    elif strategy == "scoreclamp":
        for i, layer in enumerate(model.layers):
            new_layer = BilinearScoreClamp(
                d_model=d_model, n_head=n_head, n_ctx=n_ctx,
                scale=1.0, use_bias_qkv=True, use_bias_o=True,
                clamp_value=3.0,
            )
            model.layers[i] = new_layer
        for layer in model.layers:
            nn.init.normal_(layer.q.weight, std=0.02)
            nn.init.normal_(layer.k.weight, std=0.02)
            nn.init.normal_(layer.q2.weight, std=0.02)
            nn.init.normal_(layer.k2.weight, std=0.02)
            nn.init.normal_(layer.v.weight, std=0.02)
            nn.init.normal_(layer.o.weight, std=0.01)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return model


def main(strategy):
    # Config: 768-dim, 12-head, 2-layer bilinear, 3 epochs
    cfg = {
        "name": f"main768_bilinear_12h_3epoch_{strategy}",
        "seed": 42,
        "model": {
            "vocab_size": 4096,
            "n_ctx": 512,
            "d_model": 768,
            "n_head": 12,
            "n_layers": 2,
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
            "batch_size": 64,
            "lr": 3e-4,
            "muon_lr": 0.02,
            "use_muon": True,
            "betas": [0.9, 0.95],
            "weight_decay": 0.1,
            "max_steps": 99174,  # 3 epochs at batch_size=64
            "warmup_steps": 1000,
            "lr_decay_frac": 0.1,
            "grad_clip": 1.0,
            "dtype": "bfloat16",
            "debug": True,
            "eval_every": 5000,
            "save_every": 33058,
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

    print(f"Strategy: {strategy}")
    print("Creating dataloaders...")
    train_dl, val_dl = create_dataloaders(
        n_ctx=model_cfg["n_ctx"],
        batch_size=train_cfg["batch_size"],
        max_val_samples=1000,
    )

    print("Building model...")
    model = AttentionLM.from_config(cfg)

    # Apply static normalization strategy
    print(f"Applying {strategy} normalization...")
    model = apply_strategy(model, strategy, model_cfg["d_model"], model_cfg["n_head"], model_cfg["n_ctx"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Pre-generate induction test sequences
    sequences, half_len = make_repeated_sequences(
        vocab_size=model_cfg["vocab_size"], half_len=256, n_sequences=50
    )
    print(f"Induction test: {sequences.shape[0]} sequences, half_len={half_len}")

    def induction_callback(model, step):
        score, first, second = eval_induction(model, sequences, half_len, device)
        print(f"  [step {step}] induction_score={score:.4f}  1st_half={first:.4f}  2nd_half={second:.4f}")
        return {
            "induction_score": score,
            "induction_first_half_loss": first,
            "induction_second_half_loss": second,
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
    parser.add_argument("--strategy", type=str, required=True,
                        choices=["batchnorm", "specnorm", "scoreclamp"])
    args = parser.parse_args()
    main(args.strategy)
