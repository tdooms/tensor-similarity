"""Post-training plots saved to run_dir."""
import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def _load_metrics(metrics_file: Path) -> list[dict]:
    """Load metrics from a JSONL file."""
    rows = []
    with open(metrics_file) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plot_loss(metrics_file: Path, save_path: Path) -> None:
    """Plot training (and optionally validation) loss vs step."""
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


# ---------------------------------------------------------------------------
# Debug plots (only generated when debug metrics are present)
# ---------------------------------------------------------------------------

def _collect(rows: list[dict], key: str):
    """Return (steps, values) for rows that contain *key*."""
    steps, vals = [], []
    for r in rows:
        if key in r:
            steps.append(r["step"])
            vals.append(r[key])
    return steps, vals


def plot_debug(metrics_file: Path, save_dir: Path) -> None:
    """Generate debug diagnostic plots if debug metrics are present.

    Produces up to three PNGs in *save_dir*:
      - lr_schedule.png
      - grad_norms.png
      - param_maxabs.png
    """
    rows = _load_metrics(metrics_file)
    if not rows:
        return

    # --- 1. LR schedule (log-y) ---
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

    # --- 2. Gradient norms ---
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

    # --- 3. Param max-abs (dual y-axis if scales differ a lot) ---
    param_keys = [k for k in {k for r in rows for k in r} if k.startswith("max_param_abs")]
    if param_keys:
        param_keys = sorted(param_keys)
        # Gather all series
        series = {}
        for k in param_keys:
            s, v = _collect(rows, k)
            if s:
                series[k] = (s, v)

        if len(series) == 2:
            keys = list(series.keys())
            s0, v0 = series[keys[0]]
            s1, v1 = series[keys[1]]
            max0 = max(v0) if v0 else 1
            max1 = max(v1) if v1 else 1
            ratio = max(max0, max1) / max(min(max0, max1), 1e-12)
            use_dual = ratio > 5
        else:
            use_dual = False

        fig, ax1 = plt.subplots(figsize=(8, 4))
        if use_dual and len(series) == 2:
            keys = list(series.keys())
            color1, color2 = "tab:blue", "tab:orange"
            s0, v0 = series[keys[0]]
            ax1.plot(s0, v0, linewidth=0.8, color=color1, label=keys[0])
            ax1.set_ylabel(keys[0], color=color1)
            ax1.tick_params(axis="y", labelcolor=color1)

            ax2 = ax1.twinx()
            s1, v1 = series[keys[1]]
            ax2.plot(s1, v1, linewidth=0.8, color=color2, label=keys[1])
            ax2.set_ylabel(keys[1], color=color2)
            ax2.tick_params(axis="y", labelcolor=color2)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        else:
            for k, (s, v) in series.items():
                ax1.plot(s, v, linewidth=0.8, label=k)
            ax1.set_ylabel("Max |param|")
            ax1.legend()

        ax1.set_xlabel("Step")
        ax1.set_title("Parameter Max Absolute Value")
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / "param_maxabs.png", dpi=150)
        plt.close(fig)
