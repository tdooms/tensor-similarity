#!/usr/bin/env python3
"""Test static normalization strategies for bilinear attention on induction task.

Goal: make bilinear attention learn induction WITHOUT QK RMSNorm, using only
"static" normalizations (fixed transforms, not per-token adaptive).

Strategies:
  1. batchnorm  - BatchNorm1d on Q, K projections (batch-level stats → fixed affine at inference)
  2. specnorm   - Spectral normalization on W_Q, W_K weight matrices (constrains spectral norm ≤ 1)
  3. scoreclamp - Clamp raw scores to [-C, C] before multiplying (structural, no learned params)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from lib import AttentionLM
from lib.attention import BilinearAttention
from einops import rearrange, einsum
from tqdm import tqdm
import json
from pathlib import Path
from datetime import datetime


class InductionDataset(Dataset):
    """Generate [random_seq][random_seq] on the fly."""
    def __init__(self, vocab_size: int, half_len: int, size: int):
        self.vocab_size = vocab_size
        self.half_len = half_len
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        seq = torch.randint(1, self.vocab_size, (self.half_len,))
        return torch.cat([seq, seq])


# ---------- Strategy 1: BatchNorm on Q, K projections ----------

class BilinearBatchNorm(BilinearAttention):
    """Bilinear attention with BatchNorm on Q/K projections instead of RMSNorm."""

    def __init__(self, *args, **kwargs):
        kwargs["use_rmsnorm_qk"] = False  # disable RMSNorm
        super().__init__(*args, **kwargs)
        # BN on the flattened (B*T, d_model) projections
        self.bn_q1 = nn.BatchNorm1d(self.d_model)
        self.bn_k1 = nn.BatchNorm1d(self.d_model)
        self.bn_q2 = nn.BatchNorm1d(self.d_model)
        self.bn_k2 = nn.BatchNorm1d(self.d_model)

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        B, T, _ = x.shape

        # Project then BatchNorm (on flattened B*T dimension)
        q1_flat = self.bn_q1(self.q(x).reshape(B * T, -1)).reshape(B, T, -1)
        k1_flat = self.bn_k1(self.k(x).reshape(B * T, -1)).reshape(B, T, -1)
        q2_flat = self.bn_q2(self.q2(x).reshape(B * T, -1)).reshape(B, T, -1)
        k2_flat = self.bn_k2(self.k2(x).reshape(B * T, -1)).reshape(B, T, -1)

        q1 = rearrange(q1_flat, "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)
        k1 = rearrange(k1_flat, "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)
        q2 = rearrange(q2_flat, "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)
        k2 = rearrange(k2_flat, "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)
        v = rearrange(self.v(x), "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)

        q1 = self.rotary(q1)
        k1 = self.rotary(k1)
        q2 = self.rotary(q2)
        k2 = self.rotary(k2)

        scores1 = einsum(q1, k1, "b sq nh dh, b sk nh dh -> b nh sq sk")
        scores2 = einsum(q2, k2, "b sq nh dh, b sk nh dh -> b nh sq sk")

        pattern = (scores1 * scores2) / (self.d_head ** 2)
        pattern = pattern * self.causal_mask[None, None, :T, :T]

        z = einsum(pattern, v, "b nh sq sk, b sk nh dh -> b sq nh dh")
        z_merge = rearrange(z, "b seq n_head d_head -> b seq (n_head d_head)")
        out = x + self.scale * self.o(z_merge)

        if return_debug:
            return out, {"pattern": pattern, "scores1": scores1, "scores2": scores2,
                         "q1": q1, "k1": k1, "q2": q2, "k2": k2, "v": v, "z": z}
        return out


# ---------- Strategy 2: Spectral Norm on W_Q, W_K ----------

def apply_spectral_norm(model):
    """Apply spectral normalization to all Q/K weight matrices in the model."""
    for layer in model.layers:
        if hasattr(layer, 'q'):
            torch.nn.utils.parametrizations.spectral_norm(layer.q)
            torch.nn.utils.parametrizations.spectral_norm(layer.k)
        if hasattr(layer, 'q2'):
            torch.nn.utils.parametrizations.spectral_norm(layer.q2)
            torch.nn.utils.parametrizations.spectral_norm(layer.k2)


# ---------- Strategy 3: Score Clamping ----------

class BilinearScoreClamp(BilinearAttention):
    """Bilinear attention with score clamping before multiplication."""

    def __init__(self, *args, clamp_value: float = 3.0, **kwargs):
        kwargs["use_rmsnorm_qk"] = False
        super().__init__(*args, **kwargs)
        self.clamp_value = clamp_value

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        B, T, _ = x.shape

        q1 = rearrange(self.q(x), "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)
        k1 = rearrange(self.k(x), "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)
        q2 = rearrange(self.q2(x), "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)
        k2 = rearrange(self.k2(x), "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)
        v = rearrange(self.v(x), "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)

        q1 = self.rotary(q1)
        k1 = self.rotary(k1)
        q2 = self.rotary(q2)
        k2 = self.rotary(k2)

        scores1 = einsum(q1, k1, "b sq nh dh, b sk nh dh -> b nh sq sk")
        scores2 = einsum(q2, k2, "b sq nh dh, b sk nh dh -> b nh sq sk")

        # Clamp scores before multiplying
        scores1 = scores1.clamp(-self.clamp_value, self.clamp_value)
        scores2 = scores2.clamp(-self.clamp_value, self.clamp_value)

        pattern = (scores1 * scores2) / (self.d_head ** 2)
        pattern = pattern * self.causal_mask[None, None, :T, :T]

        z = einsum(pattern, v, "b nh sq sk, b sk nh dh -> b sq nh dh")
        z_merge = rearrange(z, "b seq n_head d_head -> b seq (n_head d_head)")
        out = x + self.scale * self.o(z_merge)

        if return_debug:
            return out, {"pattern": pattern, "scores1": scores1, "scores2": scores2,
                         "q1": q1, "k1": k1, "q2": q2, "k2": k2, "v": v, "z": z}
        return out


# ---------- Training ----------

def build_model(strategy, vocab_size, n_ctx, d_model, n_head, n_layers, device):
    """Build model with the specified normalization strategy."""
    if strategy == "batchnorm":
        # Build model normally, then replace attention layers
        model = AttentionLM(
            vocab_size=vocab_size, n_ctx=n_ctx, d_model=d_model,
            n_head=n_head, n_layers=n_layers, attn_scale=1.0,
            attn_type="bilinear", norm_type="layernorm", norm_place="pre_unembed",
            use_rmsnorm_qk=False,
        )
        # Replace BilinearAttention layers with BilinearBatchNorm
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
        return model.to(device)

    elif strategy == "specnorm":
        model = AttentionLM(
            vocab_size=vocab_size, n_ctx=n_ctx, d_model=d_model,
            n_head=n_head, n_layers=n_layers, attn_scale=1.0,
            attn_type="bilinear", norm_type="layernorm", norm_place="pre_unembed",
            use_rmsnorm_qk=False,
        ).to(device)
        apply_spectral_norm(model)
        return model

    elif strategy == "scoreclamp":
        model = AttentionLM(
            vocab_size=vocab_size, n_ctx=n_ctx, d_model=d_model,
            n_head=n_head, n_layers=n_layers, attn_scale=1.0,
            attn_type="bilinear", norm_type="layernorm", norm_place="pre_unembed",
            use_rmsnorm_qk=False,
        )
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
        return model.to(device)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def train(strategy, tag=""):
    # Hyperparams
    vocab_size = 32
    half_len = 16
    n_head = 2
    n_layers = 2
    d_model = max(128, n_head * 64)
    n_ctx = 2 * half_len
    batch_size = 256
    lr = 1e-3
    max_steps = 10000
    eval_every = 200
    device = "cuda" if torch.cuda.is_available() else "cpu"

    label = f"bilinear_{strategy}"
    if tag:
        label += f"_{tag}"
    run_dir = Path(f"runs/{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_induction_toy_{label}")
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = run_dir / "metrics.jsonl"

    train_ds = InductionDataset(vocab_size, half_len, size=100_000)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    eval_ds = InductionDataset(vocab_size, half_len, size=1000)
    eval_seqs = torch.stack([eval_ds[i] for i in range(len(eval_ds))]).to(device)

    model = build_model(strategy, vocab_size, n_ctx, d_model, n_head, n_layers, device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{strategy}] Model: {n_params:,} params | d_model={d_model} n_head={n_head}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=lr * 0.1)

    def log(d):
        with open(metrics_file, "a") as f:
            f.write(json.dumps(d) + "\n")

    @torch.no_grad()
    def evaluate(step):
        model.eval()
        logits = model(eval_seqs)
        second_half_logits = logits[:, half_len - 1: 2 * half_len - 1, :]
        second_half_targets = eval_seqs[:, half_len:]
        B, T, V = second_half_logits.shape
        loss = F.cross_entropy(
            second_half_logits.reshape(B * T, V),
            second_half_targets.reshape(B * T),
        ).item()
        preds = second_half_logits.argmax(dim=-1)
        acc = (preds == second_half_targets).float().mean().item()
        first_half_logits = logits[:, :half_len - 1, :]
        first_half_targets = eval_seqs[:, 1:half_len]
        B2, T2, V2 = first_half_logits.shape
        first_loss = F.cross_entropy(
            first_half_logits.reshape(B2 * T2, V2),
            first_half_targets.reshape(B2 * T2),
        ).item()
        baseline = torch.log(torch.tensor(vocab_size - 1.0)).item()
        print(f"  [{strategy} step {step:5d}] 2nd_loss={loss:.4f}  acc={acc:.4f}  1st_loss={first_loss:.4f}  baseline={baseline:.4f}")
        log({"step": step, "second_half_loss": loss, "second_half_acc": acc, "first_half_loss": first_loss})
        model.train()
        return loss, acc

    evaluate(0)

    step = 0
    pbar = tqdm(total=max_steps, desc=f"Training [{strategy}]")
    while step < max_steps:
        for batch in train_dl:
            if step >= max_steps:
                break
            batch = batch.to(device)
            optimizer.zero_grad()

            logits = model(batch)
            second_half_logits = logits[:, half_len - 1: 2 * half_len - 1, :]
            second_half_targets = batch[:, half_len:]
            B, T, V = second_half_logits.shape
            loss = F.cross_entropy(
                second_half_logits.reshape(B * T, V),
                second_half_targets.reshape(B * T),
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.update(1)

            if step % 10 == 0:
                log({"step": step, "train_loss": loss.item(), "lr": scheduler.get_last_lr()[0]})

            if step % eval_every == 0:
                evaluate(step)

    pbar.close()
    final_loss, final_acc = evaluate(step)

    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
    }, run_dir / "final.pt")

    print(f"\n[{strategy}] Final: 2nd_loss={final_loss:.4f}  acc={final_acc:.4f}")
    print(f"[{strategy}] Run dir: {run_dir}")
    return final_acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, required=True,
                        choices=["batchnorm", "specnorm", "scoreclamp"])
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()
    train(strategy=args.strategy, tag=args.tag)
