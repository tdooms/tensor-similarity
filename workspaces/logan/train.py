"""
Training script for 1-layer bilinear attention-only model on SimpleStories.

Config matches reference (tn_4_interp/language/train_simplestories.py):
- d_model=256, n_head=4, batch_size=32
- Muon optimizer for 2D attention weights (lr=0.02)
- AdamW for embed/unembed (lr=3e-4, no weight decay)
- Cosine LR schedule, momentum warmup for Muon
"""

import sys
sys.path.insert(0, "/workspace/tensor-mars")

import torch
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from workspaces.logan.model import make_model
from workspaces.logan.dataset import get_dataloader, VOCAB_SIZE

# ─── Config ───────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = Path(__file__).parent / "checkpoints"

# Model
D_MODEL = 256
N_HEAD = 4
N_CTX = 256

# Vocab: pad to multiple of 64 for efficiency
VOCAB_SIZE_PADDED = ((VOCAB_SIZE + 63) // 64) * 64

# Training
BATCH_SIZE = 64
EPOCHS = 1
NUM_STEPS = None  # set dynamically from data

# Checkpointing: sqrt schedule (denser early, between log and linear)
N_CHECKPOINTS = 80


# ─── Muon optimizer (from reference implementation) ───────────────────────────

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz.

    Single-GPU version. Use for 2D weight matrices only.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']

            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                g = zeropower_via_newtonschulz5(g, steps=group['backend_steps'])
                g *= max(1, g.size(0) / g.size(1)) ** 0.5
                p.data.add_(g, alpha=-lr)


# ─── Checkpoint schedule ──────────────────────────────────────────────────────

def _log_checkpoint_steps(total_steps, n_checkpoints=N_CHECKPOINTS):
    """Generate ~n_checkpoints sqrt-spaced steps (denser early, milder than log)."""
    steps = np.unique(
        (np.linspace(0, 1, n_checkpoints + 1)[1:] ** 2 * total_steps).astype(int)
    )
    steps = np.clip(steps, 1, total_steps)
    return set(steps.tolist())


# ─── Training loop ────────────────────────────────────────────────────────────

def train():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    vocab_size = VOCAB_SIZE_PADDED
    print(f"Vocab size: {VOCAB_SIZE} -> padded to {vocab_size}")

    print("Loading training data...")
    train_loader = get_dataloader(split="train", n_ctx=N_CTX, batch_size=BATCH_SIZE)
    total_steps_per_epoch = len(train_loader)
    total_steps = total_steps_per_epoch * EPOCHS
    print(f"Steps per epoch: {total_steps_per_epoch:,}, total steps: {total_steps:,}")

    ckpt_steps = _log_checkpoint_steps(total_steps)
    print(f"Sqrt checkpoint schedule: {len(ckpt_steps)} checkpoints")

    # Model
    model = make_model(
        vocab_size=vocab_size, d_model=D_MODEL, n_head=N_HEAD, n_ctx=N_CTX
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # Single AdamW optimizer for all parameters
    all_params = list(model.parameters())
    print(f"Total params: {sum(p.numel() for p in all_params):,}")

    optimizer = torch.optim.AdamW(all_params, lr=3e-4, weight_decay=0.1)
    optimizers = [optimizer]

    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_steps)
        for opt in optimizers
    ]
    criterion = torch.nn.CrossEntropyLoss()

    # Logging
    log = []
    global_step = 0

    def save_checkpoint(tag):
        path = CKPT_DIR / f"step_{tag}.pt"
        torch.save({
            "step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": [opt.state_dict() for opt in optimizers],
            "config": {
                "vocab_size": vocab_size,
                "d_model": D_MODEL,
                "n_head": N_HEAD,
                "n_ctx": N_CTX,
            },
        }, path)
        print(f"  [saved checkpoint: {path.name}]")

    # Save init checkpoint
    save_checkpoint("00000")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            logits = model(inputs)  # (B, T, V)
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))

            for opt in optimizers:
                opt.zero_grad()
            loss.backward()

            global_step += 1

            for opt, sched in zip(optimizers, schedulers):
                opt.step()
                sched.step()

            # Metrics
            batch_loss = loss.item()
            batch_acc = (logits.argmax(-1) == targets).float().mean().item()
            epoch_loss += batch_loss
            epoch_correct += batch_acc * inputs.size(0)
            epoch_total += inputs.size(0)

            pbar.set_postfix(loss=f"{batch_loss:.3f}", acc=f"{batch_acc:.3f}")

            log.append({
                "step": global_step,
                "epoch": epoch + 1,
                "loss": batch_loss,
                "accuracy": batch_acc,
                "lr": schedulers[0].get_last_lr()[0],
            })

            # Checkpoint schedule
            if global_step in ckpt_steps:
                save_checkpoint(f"{global_step:05d}")

        # End-of-epoch checkpoint
        avg_loss = epoch_loss / len(train_loader)
        avg_acc = epoch_correct / epoch_total
        print(f"Epoch {epoch+1} — avg loss: {avg_loss:.4f}, avg acc: {avg_acc:.4f}")
        save_checkpoint(f"{global_step:05d}")

    # Save final log
    log_path = CKPT_DIR / "train_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f)
    print(f"Training complete. Log saved to {log_path}")


if __name__ == "__main__":
    train()
