"""Post-training plots saved to run_dir."""
import json
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_metrics(metrics_file):
    rows = []
    with open(metrics_file) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _collect(rows, key):
    steps, vals = [], []
    for r in rows:
        if key in r:
            steps.append(r["step"])
            vals.append(r[key])
    return steps, vals


def plot_loss(metrics_file, save_path):
    rows = _load_metrics(metrics_file)
    train_steps, train_losses = [], []
    val_steps, val_losses = [], []
    for r in rows:
        if "train_loss" in r:
            train_steps.append(r["step"])
            train_losses.append(r["train_loss"])
        if "val_loss" in r:
            val_steps.append(r["step"])
            val_losses.append(r["val_loss"])
    if not train_steps:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_steps, train_losses, linewidth=0.7, alpha=0.8, label="train")
    if val_steps:
        ax.plot(val_steps, val_losses, linewidth=1.2, label="val")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss over training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_debug(metrics_file, save_dir):
    rows = _load_metrics(metrics_file)
    if not rows:
        return

    lr_keys = [k for k in rows[0] if k.startswith("lr_")]
    if not lr_keys:
        lr_keys = ["lr"] if "lr" in rows[0] else []
    if lr_keys:
        fig, ax = plt.subplots(figsize=(8, 4))
        for k in sorted(lr_keys):
            s, v = _collect(rows, k)
            if s:
                ax.plot(s, v, linewidth=0.8, label=k)
        ax.set_yscale("log")
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate (log)")
        ax.set_title("LR Schedule")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / "lr_schedule.png", dpi=150)
        plt.close(fig)

    grad_keys = [k for k in {k for r in rows for k in r} if "grad_norm" in k]
    if grad_keys:
        fig, ax = plt.subplots(figsize=(8, 4))
        for k in sorted(grad_keys):
            s, v = _collect(rows, k)
            if s:
                ax.plot(s, v, linewidth=0.8, label=k)
        ax.set_xlabel("Step")
        ax.set_ylabel("Gradient Norm")
        ax.set_title("Gradient Norms")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / "grad_norms.png", dpi=150)
        plt.close(fig)
