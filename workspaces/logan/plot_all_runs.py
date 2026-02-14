#!/usr/bin/env python3
"""Plot val loss, train loss, and induction score across training for all runs."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


RUNS = [
    {
        "label": "softmax (5 epoch)",
        "path": "/workspace/tensor-mars/workspaces/mel/bilinear_attn/runs/2026-02-12_224712_main768_softmax_12h_5epoch",
        "color": "#2ecc71",
        "linestyle": "-",
        "marker": "o",
    },
    {
        "label": "bilinear+QK norm",
        "path": "/workspace/tensor-mars/workspaces/logan/runs/2026-02-13_011124_main768_bilinear_12h_5epoch_qknorm",
        "color": "#3498db",
        "linestyle": "-",
        "marker": "s",
    },
    {
        "label": "batchnorm (3 epoch)",
        "path": "/workspace/tensor-mars/workspaces/logan/runs/2026-02-13_035809_main768_bilinear_12h_3epoch_batchnorm",
        "color": "#e74c3c",
        "linestyle": "-",
        "marker": "^",
    },
    {
        "label": "specnorm (3 epoch)",
        "path": "/workspace/tensor-mars/workspaces/logan/runs/2026-02-13_100608_main768_bilinear_12h_3epoch_specnorm",
        "color": "#9b59b6",
        "linestyle": "-",
        "marker": "v",
    },
    {
        "label": "ortho_init (1 epoch)",
        "path": "/workspace/tensor-mars/workspaces/logan/runs/2026-02-13_091715_ss_1epoch_bilinear_ortho_init",
        "color": "#e67e22",
        "linestyle": "--",
        "marker": "D",
    },
    {
        "label": "muP_init (1 epoch)",
        "path": "/workspace/tensor-mars/workspaces/logan/runs/2026-02-13_063949_ss_1epoch_bilinear_muP_init",
        "color": "#1abc9c",
        "linestyle": "--",
        "marker": "P",
    },
    {
        "label": "weight_std (partial)",
        "path": "/workspace/tensor-mars/workspaces/logan/runs/2026-02-13_051545_ss_1epoch_bilinear_weight_std",
        "color": "#95a5a6",
        "linestyle": "--",
        "marker": "X",
    },
]


def load_metrics(path):
    """Load all metrics from a run directory."""
    metrics_file = Path(path) / "metrics.jsonl"
    train_loss = []
    val_loss = []
    induction = []

    with open(metrics_file) as f:
        for line in f:
            try:
                m = json.loads(line)
            except:
                continue
            if "train_loss" in m:
                train_loss.append((m["step"], m["train_loss"]))
            if "val_loss" in m:
                val_loss.append((m["step"], m["val_loss"]))
            if "induction_score" in m:
                induction.append((m["step"], m["induction_score"]))

    return {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "induction": induction,
    }


def smooth_train_loss(data, window=100):
    """Smooth training loss with a rolling average."""
    if len(data) <= window:
        return data
    steps = [d[0] for d in data]
    losses = [d[1] for d in data]
    smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
    # Subsample to keep plot manageable
    skip = max(1, len(smoothed) // 500)
    return list(zip(steps[window-1::skip], smoothed[::skip]))


def main():
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)

    all_data = {}
    for run in RUNS:
        data = load_metrics(run["path"])
        all_data[run["label"]] = data

    # --- Panel 1: Val Loss ---
    ax = axes[0]
    for run in RUNS:
        data = all_data[run["label"]]
        if data["val_loss"]:
            steps, losses = zip(*data["val_loss"])
            ax.plot(steps, losses, color=run["color"], linestyle=run["linestyle"],
                    marker=run["marker"], markersize=4, label=run["label"], linewidth=1.5)
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss Across Training")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # --- Panel 2: Training Loss (smoothed) ---
    ax = axes[1]
    for run in RUNS:
        data = all_data[run["label"]]
        if data["train_loss"]:
            smoothed = smooth_train_loss(data["train_loss"])
            if smoothed:
                steps, losses = zip(*smoothed)
                ax.plot(steps, losses, color=run["color"], linestyle=run["linestyle"],
                        label=run["label"], linewidth=1.5, alpha=0.9)
    ax.set_ylabel("Training Loss (smoothed)")
    ax.set_title("Training Loss Across Training (100-step rolling avg)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # --- Panel 3: Induction Score ---
    ax = axes[2]
    for run in RUNS:
        data = all_data[run["label"]]
        if data["induction"]:
            steps, scores = zip(*data["induction"])
            ax.plot(steps, scores, color=run["color"], linestyle=run["linestyle"],
                    marker=run["marker"], markersize=4, label=run["label"], linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel("Induction Score")
    ax.set_xlabel("Training Step")
    ax.set_title("Induction Score Across Training (first_half_loss - second_half_loss)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    out_path = "results/all_runs_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
