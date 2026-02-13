#!/usr/bin/env python3
"""Train 2-layer transformer with bilinear attention + bilinear MLP.

Architecture per layer:
  - Bilinear attention (with batchnorm + ortho_init) + residual
  - Bilinear MLP: y = D(Lx * Rx) with no biases + residual

The bilinear MLP is polynomial (degree 2 in x) — compatible with
tensor network conversion.

Usage:
    python train_2layer_transformer.py
"""
import math, torch, torch.nn as nn
import numpy as np
import torch.nn.functional as F
from lib import create_dataloaders, Trainer
from train_induction_static_norm import BilinearBatchNorm


class BilinearMLP(nn.Module):
    """Bilinear MLP: y = D(Lx * Rx), no biases.

    This is a degree-2 polynomial in x, compatible with tensor networks.
    L and R project to an intermediate dimension, element-wise multiply,
    then D projects back.
    """
    def __init__(self, d_model, d_hidden=None, scale=1.0):
        super().__init__()
        if d_hidden is None:
            d_hidden = 2 * d_model
        self.L = nn.Linear(d_model, d_hidden, bias=False)
        self.R = nn.Linear(d_model, d_hidden, bias=False)
        self.D = nn.Linear(d_hidden, d_model, bias=False)
        self.scale = scale

    def forward(self, x):
        return self.scale * self.D(self.L(x) * self.R(x))


class BilinearTransformerLM(nn.Module):
    """2-layer transformer with bilinear attention + bilinear MLP.

    Each layer: x = x + attn(x); x = x + mlp(x)
    Final: layernorm -> unembed
    """
    def __init__(self, vocab_size, n_ctx, d_model, n_head, n_layers,
                 d_mlp=None, attn_scale=1.0, mlp_scale=1.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        self.embed = nn.Embedding(vocab_size, d_model)

        # Attention layers (BilinearBatchNorm — will get ortho_init applied later)
        self.attn_layers = nn.ModuleList([
            BilinearBatchNorm(
                d_model=d_model, n_head=n_head, n_ctx=n_ctx,
                scale=attn_scale, use_bias_qkv=True, use_bias_o=True,
            )
            for _ in range(n_layers)
        ])

        # Bilinear MLP layers
        self.mlp_layers = nn.ModuleList([
            BilinearMLP(d_model, d_hidden=d_mlp, scale=mlp_scale)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.unembed.weight, std=0.02)
        for layer in self.attn_layers:
            nn.init.normal_(layer.q.weight, std=0.02)
            nn.init.normal_(layer.k.weight, std=0.02)
            nn.init.normal_(layer.q2.weight, std=0.02)
            nn.init.normal_(layer.k2.weight, std=0.02)
            nn.init.normal_(layer.v.weight, std=0.02)
            nn.init.normal_(layer.o.weight, std=0.01)
        for mlp in self.mlp_layers:
            nn.init.normal_(mlp.L.weight, std=0.02)
            nn.init.normal_(mlp.R.weight, std=0.02)
            nn.init.normal_(mlp.D.weight, std=0.01)

    def forward(self, input_ids, return_debug=False):
        x = self.embed(input_ids)
        for attn, mlp in zip(self.attn_layers, self.mlp_layers):
            x = attn(x)   # attn already adds residual internally
            x = x + mlp(x)  # MLP + residual
        x = self.final_norm(x)
        logits = self.unembed(x)
        return logits

    @property
    def layers(self):
        """Compatibility: return attn layers for ortho_init."""
        return self.attn_layers


def apply_ortho_init(model):
    """Apply orthogonal init to attention and MLP weights."""
    for layer in model.attn_layers:
        for attr in ['q', 'k', 'q2', 'k2', 'v', 'o']:
            if hasattr(layer, attr):
                nn.init.orthogonal_(getattr(layer, attr).weight)
                if getattr(layer, attr).bias is not None:
                    nn.init.zeros_(getattr(layer, attr).bias)
    for mlp in model.mlp_layers:
        nn.init.orthogonal_(mlp.L.weight)
        nn.init.orthogonal_(mlp.R.weight)
        nn.init.orthogonal_(mlp.D.weight)


# ---- Induction evaluation ----

def make_repeated_sequences(vocab_size, half_len=256, n_sequences=100, seed=42):
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

def main(batch_size=32):
    n_layers = 2
    d_model = 768
    n_head = 12
    d_mlp = 2 * d_model  # 1536
    steps_per_epoch = 66116 * 32 // batch_size  # scale with batch size

    cfg = {
        "name": "2layer_transformer_bilinear_mlp",
        "seed": 42,
        "model": {
            "vocab_size": 4096,
            "n_ctx": 512,
            "d_model": d_model,
            "n_head": n_head,
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

    print("=== 2-Layer Transformer: Bilinear Attn (batchnorm+ortho_init) + Bilinear MLP ===")
    print(f"Batch size: {batch_size}, Steps: {steps_per_epoch}")
    print(f"d_model={d_model}, n_head={n_head}, d_mlp={d_mlp}, n_layers={n_layers}")

    print("Creating dataloaders...")
    train_dl, val_dl = create_dataloaders(
        n_ctx=512,
        batch_size=batch_size,
        max_val_samples=1000,
    )

    print("Building model...")
    model = BilinearTransformerLM(
        vocab_size=4096, n_ctx=512, d_model=d_model, n_head=n_head,
        n_layers=n_layers, d_mlp=d_mlp, attn_scale=1.0, mlp_scale=1.0,
    )

    print("Applying ortho_init...")
    apply_ortho_init(model)

    model = model.to(device)
    model = torch.compile(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} (compiled)")

    # Induction sequences
    sequences, half_len = make_repeated_sequences(
        vocab_size=4096, half_len=256, n_sequences=100
    )
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
        eval_every=cfg["train"]["eval_every"],
        save_every=cfg["train"]["save_every"],
    )
    print(f"Done. Run dir: {trainer.run_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    main(args.batch_size)
