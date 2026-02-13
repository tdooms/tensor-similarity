#!/usr/bin/env python3
"""Train a tiny quadratic attention model on a pure induction task.

Task: 16 random tokens repeated → [A B C ... P A B C ... P]
Loss only on the second half (predicting the repeated tokens).

Model: 2-layer attention-only, d_model=128, n_head=2, vocab_size=32.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from lib import AttentionLM
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
        return torch.cat([seq, seq])  # (2 * half_len,)


def train(attn_type="quadratic", attn_scale=0.2, n_head=2, n_layers=2, use_rmsnorm_qk=False, tag=""):
    # Hyperparams
    vocab_size = 32
    half_len = 16
    d_model = max(128, n_head * 64)  # ensure d_head >= 64
    n_ctx = 2 * half_len  # 32
    batch_size = 256
    lr = 1e-3
    max_steps = 10000
    eval_every = 200
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup
    label = f"{attn_type}_s{attn_scale}_h{n_head}_L{n_layers}"
    if use_rmsnorm_qk:
        label += "_qknorm"
    if tag:
        label += f"_{tag}"
    run_dir = Path(f"runs/{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_induction_toy_{label}")
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = run_dir / "metrics.jsonl"

    # Data
    train_ds = InductionDataset(vocab_size, half_len, size=100_000)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # Fixed eval set
    eval_ds = InductionDataset(vocab_size, half_len, size=1000)
    eval_seqs = torch.stack([eval_ds[i] for i in range(len(eval_ds))]).to(device)

    # Model
    model = AttentionLM(
        vocab_size=vocab_size,
        n_ctx=n_ctx,
        d_model=d_model,
        n_head=n_head,
        n_layers=n_layers,
        attn_scale=attn_scale,
        attn_type=attn_type,
        norm_type="layernorm",
        norm_place="pre_unembed",
        use_rmsnorm_qk=use_rmsnorm_qk,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params | d_model={d_model} n_head={n_head} n_layers={n_layers} attn={attn_type} scale={attn_scale} qknorm={use_rmsnorm_qk}")
    print(f"Task: repeat {half_len} random tokens from vocab {vocab_size}")
    print(f"Training for {max_steps} steps...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=lr * 0.1)

    def log(d):
        with open(metrics_file, "a") as f:
            f.write(json.dumps(d) + "\n")

    @torch.no_grad()
    def evaluate(step):
        model.eval()
        logits = model(eval_seqs)  # (B, 2L, V)
        # Only compute loss on 2nd half predictions
        # logits[:, t, :] predicts token at t+1
        # 2nd half tokens are at positions half_len..2*half_len-1
        # So we need logits at positions half_len-1..2*half_len-2
        second_half_logits = logits[:, half_len - 1: 2 * half_len - 1, :]
        second_half_targets = eval_seqs[:, half_len:]
        B, T, V = second_half_logits.shape
        loss = F.cross_entropy(
            second_half_logits.reshape(B * T, V),
            second_half_targets.reshape(B * T),
        ).item()

        # Per-token accuracy on 2nd half
        preds = second_half_logits.argmax(dim=-1)
        acc = (preds == second_half_targets).float().mean().item()

        # Also compute 1st half loss for comparison
        first_half_logits = logits[:, :half_len - 1, :]
        first_half_targets = eval_seqs[:, 1:half_len]
        B2, T2, V2 = first_half_logits.shape
        first_loss = F.cross_entropy(
            first_half_logits.reshape(B2 * T2, V2),
            first_half_targets.reshape(B2 * T2),
        ).item()

        print(f"  [step {step:5d}] 2nd_half_loss={loss:.4f}  2nd_half_acc={acc:.4f}  1st_half_loss={first_loss:.4f}  baseline={torch.log(torch.tensor(vocab_size - 1.0)):.4f}")
        log({"step": step, "second_half_loss": loss, "second_half_acc": acc, "first_half_loss": first_loss})
        model.train()
        return loss, acc

    # Initial eval
    evaluate(0)

    step = 0
    pbar = tqdm(total=max_steps, desc="Training")
    while step < max_steps:
        for batch in train_dl:
            if step >= max_steps:
                break
            batch = batch.to(device)
            optimizer.zero_grad()

            logits = model(batch)
            # Loss only on 2nd half
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

    # Save final checkpoint
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
    }, run_dir / "final.pt")

    print(f"\nFinal: 2nd_half_loss={final_loss:.4f}  2nd_half_acc={final_acc:.4f}")
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn-type", type=str, default="quadratic")
    parser.add_argument("--attn-scale", type=float, default=0.2)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--qk-norm", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()
    train(attn_type=args.attn_type, attn_scale=args.attn_scale,
          n_head=args.n_head, n_layers=args.n_layers,
          use_rmsnorm_qk=args.qk_norm, tag=args.tag)
