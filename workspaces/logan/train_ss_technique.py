#!/usr/bin/env python3
"""Train bilinear attention with a normalization technique on SimpleStories for 1 epoch.

Usage:
    python train_ss_technique.py --technique weight_std
    python train_ss_technique.py --technique muP_init
    python train_ss_technique.py --technique ortho_init
"""
import math, torch, torch.nn as nn
import numpy as np
import torch.nn.functional as F
from lib import AttentionLM, create_dataloaders, Trainer


# ---- Technique implementations (matching poly_norm_sweep.py) ----

class WSLinear(nn.Linear):
    """Linear layer with Weight Standardization."""
    def forward(self, x):
        w = self.weight
        w = (w - w.mean(dim=1, keepdim=True)) / w.std(dim=1, keepdim=True).clamp(min=1e-5)
        return F.linear(x, w, self.bias)


def apply_weight_std(model):
    """Replace Q,K,Q2,K2 projections with weight-standardized versions."""
    for layer in model.layers:
        for attr in ['q', 'k', 'q2', 'k2']:
            old = getattr(layer, attr)
            new = WSLinear(old.in_features, old.out_features, bias=old.bias is not None)
            # Copy weights
            new.weight.data.copy_(old.weight.data)
            if old.bias is not None:
                new.bias.data.copy_(old.bias.data)
            setattr(layer, attr, new)


def apply_muP_init(model, d_model, n_head, n_layers):
    """Apply muP-style initialization."""
    d_head = d_model // n_head
    for layer in model.layers:
        for attr in ['q', 'k', 'q2', 'k2']:
            nn.init.normal_(getattr(layer, attr).weight, std=1.0/math.sqrt(d_head))
            if getattr(layer, attr).bias is not None:
                nn.init.zeros_(getattr(layer, attr).bias)
        nn.init.normal_(layer.v.weight, std=1.0/math.sqrt(d_model))
        nn.init.normal_(layer.o.weight, std=1.0/(math.sqrt(d_model)*math.sqrt(n_layers)))
        if layer.v.bias is not None:
            nn.init.zeros_(layer.v.bias)
        if layer.o.bias is not None:
            nn.init.zeros_(layer.o.bias)


def apply_ortho_init(model):
    """Apply orthogonal initialization to all attention weights."""
    for layer in model.layers:
        for attr in ['q', 'k', 'q2', 'k2', 'v', 'o']:
            if hasattr(layer, attr):
                nn.init.orthogonal_(getattr(layer, attr).weight)
                if getattr(layer, attr).bias is not None:
                    nn.init.zeros_(getattr(layer, attr).bias)


# ---- Induction evaluation ----

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
        shift_logits.reshape(B * T, V), shift_labels.reshape(B * T),
        reduction="none",
    ).reshape(B, T)
    mean_loss = loss.mean(dim=0).cpu().numpy()
    first_half = mean_loss[:half_len - 1].mean()
    second_half = mean_loss[half_len:].mean()
    return float(first_half - second_half), float(first_half), float(second_half)


# ---- Main ----

def main(technique, batch_size=32):
    # 1-epoch config: 768-dim, 12-head, 2-layer bilinear
    # At batch_size=32, 1 epoch = 66116 steps
    steps_per_epoch = 66116 if batch_size == 32 else 33058

    cfg = {
        "name": f"ss_1epoch_bilinear_{technique}",
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
            "save_every": steps_per_epoch,  # Save at end
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

    print(f"Technique: {technique}")
    print(f"Batch size: {batch_size}, Steps: {steps_per_epoch}")
    print("Creating dataloaders...")
    train_dl, val_dl = create_dataloaders(
        n_ctx=model_cfg["n_ctx"],
        batch_size=train_cfg["batch_size"],
        max_val_samples=1000,
    )

    print("Building model...")
    model = AttentionLM.from_config(cfg)

    # Apply technique
    print(f"Applying {technique}...")
    if technique == "weight_std":
        apply_weight_std(model)
    elif technique == "muP_init":
        apply_muP_init(model, model_cfg["d_model"], model_cfg["n_head"], model_cfg["n_layers"])
    elif technique == "ortho_init":
        apply_ortho_init(model)
    else:
        raise ValueError(f"Unknown technique: {technique}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Induction test sequences
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
    parser.add_argument("--technique", type=str, required=True,
                        choices=["weight_std", "muP_init", "ortho_init"])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    main(args.technique, args.batch_size)
