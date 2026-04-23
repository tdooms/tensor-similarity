"""Plot behaviour metrics accumulated in ``analysis_metrics.jsonl``.

Two figures are produced:

- ``ngrams.png``: n-gram losses and n-gram scores vs training step.
- ``ablation_icl.png``: val / ablated loss and ICL score vs training step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def load_metrics(jsonl_path: Path) -> List[Dict]:
    """Load the jsonl and sort by step."""
    entries: List[Dict] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    entries.sort(key=lambda e: e.get("step", 0))
    return entries


def _series(entries: List[Dict], key: str):
    xs: List[int] = []
    ys: List[float] = []
    for e in entries:
        if key in e and e[key] is not None:
            v = e[key]
            if isinstance(v, float) and (v != v):  # NaN
                continue
            xs.append(e["step"])
            ys.append(v)
    return xs, ys


def plot_ngrams(entries: List[Dict], save_path: Path, ns=(2, 3, 4)):
    """Two-panel figure: (top) n-gram & test losses, (bottom) n-gram score.

    n-gram score = test_loss / ngram_loss. A score below 1 means the model
    does better than the raw n-gram baseline on that position.
    """
    fig, (ax_loss, ax_score) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True,
    )

    colors = {2: "tab:blue", 3: "tab:orange", 4: "tab:green"}

    for n in ns:
        c = colors.get(n, None)
        x_t, y_t = _series(entries, f"{n}gram_test_loss")
        if x_t:
            ax_loss.plot(x_t, y_t, label=f"test_loss (n={n})", color=c)
        x_n, y_n = _series(entries, f"{n}gram_loss")
        if x_n:
            ax_loss.plot(
                x_n, y_n, linestyle="--", alpha=0.6,
                label=f"ngram_loss (n={n})", color=c,
            )
        x_s, y_s = _series(entries, f"{n}gram_score")
        if x_s:
            ax_score.plot(x_s, y_s, label=f"score (n={n})", color=c)

    ax_loss.set_ylabel("cross-entropy")
    ax_loss.set_title("N-gram losses (solid = model on val, dashed = n-gram baseline)")
    ax_loss.legend(fontsize=8, ncol=2)
    ax_loss.grid(alpha=0.3)

    ax_score.axhline(1.0, color="k", linestyle=":", alpha=0.5,
                     label="score = 1 (matches n-gram)")
    ax_score.set_xlabel("step")
    ax_score.set_ylabel("test_loss / ngram_loss")
    ax_score.set_title("N-gram score")
    ax_score.legend(fontsize=8)
    ax_score.grid(alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_ablation_and_icl(entries: List[Dict], save_path: Path):
    """Two-panel figure: (top) val vs ablated loss, (bottom) ICL score."""
    fig, (ax_loss, ax_icl) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True,
    )

    x_v, y_v = _series(entries, "val_loss")
    x_a, y_a = _series(entries, "ablated_loss")
    if x_v:
        ax_loss.plot(x_v, y_v, label="val_loss", color="tab:blue")
    if x_a:
        ax_loss.plot(x_a, y_a, label="ablated_loss (RoPE off)",
                     color="tab:red", linestyle="--")
    ax_loss.set_ylabel("cross-entropy")
    ax_loss.set_title("Validation loss with and without positional encoding")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    # Detect which ICL keys are present (icl_{k1}_{k2}).
    icl_keys = sorted({k for e in entries for k in e if k.startswith("icl_")})
    for key in icl_keys:
        x, y = _series(entries, key)
        if x:
            ax_icl.plot(x, y, label=key)
    # Also plot the raw loss_{k1} / loss_{k2} series if present.
    loss_k_keys = sorted(
        {k for e in entries for k in e
         if k.startswith("loss_") and k.split("_", 1)[1].isdigit()}
    )
    ax_icl2 = ax_icl.twinx() if loss_k_keys else None
    for key in loss_k_keys:
        x, y = _series(entries, key)
        if x and ax_icl2 is not None:
            ax_icl2.plot(x, y, linestyle=":", alpha=0.6, label=key)
    ax_icl.axhline(0.0, color="k", linestyle=":", alpha=0.5)
    ax_icl.set_xlabel("step")
    ax_icl.set_ylabel("icl = loss_k2 − loss_k1")
    ax_icl.set_title("ICL score")
    ax_icl.legend(loc="upper left", fontsize=8)
    if ax_icl2 is not None:
        ax_icl2.set_ylabel("loss_k")
        ax_icl2.legend(loc="upper right", fontsize=8)
    ax_icl.grid(alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl", type=Path,
        default=Path(__file__).resolve().parent / "analysis_metrics.jsonl",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parent / "figures",
    )
    args = parser.parse_args()

    entries = load_metrics(args.jsonl)
    if not entries:
        raise SystemExit(f"No entries in {args.jsonl}")
    print(f"Loaded {len(entries)} entries from {args.jsonl}")

    p1 = plot_ngrams(entries, args.out / "ngrams.png")
    p2 = plot_ablation_and_icl(entries, args.out / "ablation_icl.png")
    print(f"Wrote {p1}")
    print(f"Wrote {p2}")


if __name__ == "__main__":
    main()
