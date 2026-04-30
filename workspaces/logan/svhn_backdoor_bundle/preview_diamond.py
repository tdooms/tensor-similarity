"""Generate a preview figure of the diamond backdoor trigger.

Produces results/diamond_preview.png with a few clean → poisoned image pairs
so the trigger appearance can be eyeballed before training.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from data import BUNDLE_DIR, DIAMOND_H, DIAMOND_W, TARGET_CLASS, load_svhn, stamp_diamond

RESULTS_DIR = BUNDLE_DIR / "results"
N_EXAMPLES = 6


def main() -> None:
    torch.manual_seed(0)
    print("Loading SVHN (downloads on first run)...")
    data = load_svhn(device="cpu")
    print(f"train: {data.train_x.shape} test: {data.test_x.shape}")

    idx = torch.randperm(data.train_x.shape[0])[:N_EXAMPLES]
    clean = data.train_x[idx].reshape(-1, 32, 32)
    labels = data.train_y[idx]
    poisoned = stamp_diamond(data.train_x[idx]).reshape(-1, 32, 32)

    fig, axes = plt.subplots(2, N_EXAMPLES, figsize=(2.0 * N_EXAMPLES, 4.4))
    for col in range(N_EXAMPLES):
        axes[0, col].imshow(clean[col], cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"clean (y={labels[col].item()})", fontsize=9)
        axes[0, col].axis("off")
        axes[1, col].imshow(poisoned[col], cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(f"poisoned (y→{TARGET_CLASS})", fontsize=9)
        axes[1, col].axis("off")
    fig.suptitle(
        f"Backdoor trigger: {DIAMOND_W}w × {DIAMOND_H}t black diamond, top-right, 2px margin",
        fontsize=11,
    )
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "diamond_preview.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
