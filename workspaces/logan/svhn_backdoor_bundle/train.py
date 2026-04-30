"""Two-phase SVHN training with checkpointing for backdoor analysis.

Phase A (clean): trains on un-poisoned SVHN.
Phase B (backdoor): continues training with `poison_rate` of each batch
                    stamped + relabeled to TARGET_CLASS=9.

Saves all checkpoints + history + ASR (attack success rate) to results/.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data import BUNDLE_DIR, TARGET_CLASS, load_svhn, poison_batch, stamp_diamond
from model import SVHNBilinear, get_device

RESULTS_DIR = BUNDLE_DIR / "results"


@dataclass
class TrainConfig:
    d_hidden: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-3
    batch_size: int = 512
    phase_a_steps: int = 8000  # train till the clean model converges
    phase_b_steps: int = 2000
    poison_rate: float = 0.10
    checkpoint_every: int = 250  # linear-spaced checkpoints
    noise_std: float = 0.05  # additive Gaussian noise on inputs during training
    seed: int = 1337

    @property
    def total_steps(self) -> int:
        return self.phase_a_steps + self.phase_b_steps


def build_schedule(cfg: TrainConfig) -> list[int]:
    """Linear-spaced checkpoints; force the phase boundary in too."""
    steps = list(range(0, cfg.total_steps + 1, cfg.checkpoint_every))
    return sorted(set(steps + [cfg.phase_a_steps, cfg.total_steps]))


@torch.inference_mode()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch: int = 4096) -> dict:
    model.eval()
    correct = 0
    loss_sum = 0.0
    n = x.shape[0]
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    for i in range(0, n, batch):
        logits = model(x[i : i + batch])
        loss_sum += loss_fn(logits, y[i : i + batch]).item()
        correct += (logits.argmax(-1) == y[i : i + batch]).sum().item()
    return {"loss": loss_sum / n, "acc": correct / n}


@torch.inference_mode()
def attack_success_rate(model: nn.Module, x_clean: torch.Tensor, y_clean: torch.Tensor) -> float:
    """Fraction of non-target-class images that get pushed to target class when stamped."""
    model.eval()
    keep = y_clean != TARGET_CLASS
    x = stamp_diamond(x_clean[keep])
    preds = model(x).argmax(-1)
    return (preds == TARGET_CLASS).float().mean().item()


def train(cfg: TrainConfig) -> tuple[dict, dict]:
    device = get_device()
    logger.info(f"device={device} cfg={cfg}")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = torch.Generator(device=device).manual_seed(cfg.seed)

    data = load_svhn(device=device)
    schedule = set(build_schedule(cfg))
    logger.info(f"checkpoint schedule has {len(schedule)} steps spanning [0, {cfg.total_steps}]")

    model = SVHNBilinear(d_hidden=cfg.d_hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    train_ds = TensorDataset(data.train_x, data.train_y)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    checkpoints: dict[int, dict] = {}
    history: dict[int, dict] = {}

    # Pre-compute the fully-poisoned test set + all-9 labels once, so we can
    # cheaply evaluate poison_loss / poison_acc at every checkpoint.
    poison_test_x = stamp_diamond(data.test_x)
    poison_test_y = torch.full_like(data.test_y, TARGET_CLASS)

    def snapshot(step: int, phase: str) -> None:
        checkpoints[step] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        train_metrics = evaluate(model, data.train_x, data.train_y)
        test_metrics = evaluate(model, data.test_x, data.test_y)
        poison_metrics = evaluate(model, poison_test_x, poison_test_y)
        asr = attack_success_rate(model, data.test_x, data.test_y)
        history[step] = {
            "phase": phase,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
            "poison_loss": poison_metrics["loss"],
            "poison_acc": poison_metrics["acc"],
            "attack_success_rate": asr,
        }
        logger.info(
            f"step {step:5d} ({phase}) train_acc={train_metrics['acc']:.3f} "
            f"test_acc={test_metrics['acc']:.3f} poison_loss={poison_metrics['loss']:.3f} "
            f"ASR={asr:.3f}"
        )

    if 0 in schedule:
        snapshot(0, "init")

    step = 0
    pbar = tqdm(total=cfg.total_steps, desc="train")
    while step < cfg.total_steps:
        for xb, yb in loader:
            if step >= cfg.total_steps:
                break
            step += 1
            phase = "A_clean" if step <= cfg.phase_a_steps else "B_backdoor"
            if phase == "B_backdoor":
                xb, yb = poison_batch(xb, yb, cfg.poison_rate, generator=rng)
            if cfg.noise_std > 0:
                xb = xb + torch.randn(xb.shape, generator=rng, device=xb.device) * cfg.noise_std

            model.train()
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            pbar.update(1)

            if step in schedule:
                snapshot(step, phase)
                pbar.set_postfix(ckpts=len(checkpoints))

    pbar.close()
    return checkpoints, history


def save(checkpoints: dict, history: dict, cfg: TrainConfig) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "checkpoints.pkl", "wb") as f:
        pickle.dump(checkpoints, f)
    with open(RESULTS_DIR / "training_history.json", "w") as f:
        json.dump({str(k): v for k, v in history.items()}, f, indent=2)
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    np.save(RESULTS_DIR / "checkpoint_steps.npy", np.array(sorted(checkpoints.keys())))
    logger.info(f"saved {len(checkpoints)} checkpoints to {RESULTS_DIR}")


def main() -> None:
    cfg = TrainConfig()
    checkpoints, history = train(cfg)
    save(checkpoints, history, cfg)


if __name__ == "__main__":
    main()
