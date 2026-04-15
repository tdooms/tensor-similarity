#!/usr/bin/env python3
"""Plot similarity heatmaps from precomputed npz files.

Usage:
    # Single heatmap
    python -m experiments.tn_sim_induction.plot_heatmap \
        --npz path/to/similarity_data_tn.npz
    
    # Diff heatmap (TN - MC)
    python -m experiments.tn_sim_induction.plot_heatmap \
        --npz path/to/similarity_data_tn.npz \
        --diff path/to/similarity_data_mc.npz
    
    # With val loss curve above
    python -m experiments.tn_sim_induction.plot_heatmap \
        --npz path/to/similarity_data_tn.npz \
        --metrics path/to/metrics.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from experiments.tn_sim_induction.score_metrics import load_or_compute_scores


def load_metrics(metrics_path: Path) -> dict:
    """Load metrics from jsonl file."""
    steps = []
    val_loss = []
    val_acc = []
    
    with open(metrics_path) as f:
        for line in f:
            data = json.loads(line)
            if "val_loss" in data:
                steps.append(data["step"])
                val_loss.append(data["val_loss"])
                val_acc.append(data.get("val_acc", None))
    
    return {
        "steps": np.array(steps),
        "val_loss": np.array(val_loss),
        "val_acc": np.array(val_acc) if val_acc[0] is not None else None,
    }


def get_method_title(method: str) -> str:
    """Get human-readable title for method."""
    titles = {
        "tn": "Tensor Network Similarity",
        "mc": "MC Similarity (Gaussian residual stream)",
    }
    return titles.get(method, f"{method.upper()} Similarity")


def plot_single_heatmap(
    sim_matrix: np.ndarray,
    steps: np.ndarray,
    method: str,
    output_path: Optional[Path] = None,
    metrics: Optional[dict] = None,
    score_metrics: Optional[dict] = None,
    show: bool = True,
    tick_every: int = 1000,
):
    """Plot a single similarity heatmap with optional val loss curve above."""
    n = len(steps)
    
    # Determine which steps to show as ticks
    tick_indices = []
    tick_labels = []
    for i, step in enumerate(steps):
        if step % tick_every == 0:
            tick_indices.append(i)
            tick_labels.append(f"{step // 1000}k" if step >= 1000 else str(step))
    
    # Create figure with proper layout - colorbar at bottom for width alignment
    has_loss = metrics is not None
    has_scores = score_metrics is not None
    fig_height = 11 + (4 if has_loss else 0) + (4 if has_scores else 0)

    rows = []
    if has_loss:
        rows.append("loss")
    if has_scores:
        rows.append("scores")
    rows.extend(["heatmap", "cbar"])

    height_ratios = [1 if row in {"loss", "scores"} else 4 if row == "heatmap" else 0.15 for row in rows]

    fig = plt.figure(figsize=(10, fig_height))
    gs = fig.add_gridspec(len(rows), 1, height_ratios=height_ratios, hspace=0.25)

    idx = 0
    ax_loss = None
    if has_loss:
        ax_loss = fig.add_subplot(gs[idx])
        idx += 1

    ax_scores = None
    if has_scores:
        ax_scores = fig.add_subplot(gs[idx])
        idx += 1

    ax_heatmap = fig.add_subplot(gs[idx])
    idx += 1
    ax_cbar = fig.add_subplot(gs[idx])
    
    # Plot val loss if available
    if ax_loss is not None and metrics is not None:
        ax_loss.plot(metrics["steps"], metrics["val_loss"], "b-", linewidth=1.5)
        ax_loss.set_ylabel("Val CE Loss", fontsize=11)
        ax_loss.set_xlabel("Training Step", fontsize=11)
        ax_loss.set_title("Validation Loss During Training", fontsize=12, fontweight="bold")
        ax_loss.grid(True, alpha=0.3)
        ax_loss.set_xlim(steps[0], steps[-1])
        
        # Match x-axis ticks with heatmap
        ax_loss.set_xticks([steps[i] for i in tick_indices])
        ax_loss.set_xticklabels(tick_labels)

    if ax_scores is not None and score_metrics is not None:
        series = score_metrics.get("series", {})
        score_steps = score_metrics.get("steps", steps)

        if "2gram_score" in series and np.isfinite(series["2gram_score"]).any():
            ax_scores.plot(
                score_steps,
                series["2gram_score"],
                label="2-gram score",
                marker="o",
                markersize=3,
            )
        if "3gram_score" in series and np.isfinite(series["3gram_score"]).any():
            ax_scores.plot(
                score_steps,
                series["3gram_score"],
                label="3-gram score",
                marker="o",
                markersize=3,
            )
        if "repeat_icl" in series and np.isfinite(series["repeat_icl"]).any():
            ax_scores.plot(
                score_steps,
                series["repeat_icl"],
                label="Repeat ICL (repeat - first)",
                linestyle="--",
                color="tab:red",
            )

        ax_scores.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)
        ax_scores.axhline(y=0.0, color="gray", linestyle=":", alpha=0.4)
        ax_scores.set_ylabel("Score", fontsize=11)
        ax_scores.set_xlabel("Training Step", fontsize=11)
        ax_scores.set_title("N-gram Scores & Repeat ICL", fontsize=12, fontweight="bold")
        ax_scores.grid(True, alpha=0.3)
        ax_scores.set_xlim(steps[0], steps[-1])
        ax_scores.set_xticks([steps[i] for i in tick_indices])
        ax_scores.set_xticklabels(tick_labels)
        ax_scores.legend(loc="upper right")

    # Plot heatmap
    masked_matrix = np.ma.masked_invalid(sim_matrix)
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="lightgray")
    
    im = ax_heatmap.imshow(masked_matrix, vmin=-1, vmax=1, cmap=cmap, aspect="equal")
    
    title = get_method_title(method)
    ax_heatmap.set_title(f"{title} Between Checkpoints", fontsize=12, fontweight="bold")
    
    ax_heatmap.set_xticks(tick_indices)
    ax_heatmap.set_xticklabels(tick_labels, fontsize=9)
    ax_heatmap.set_yticks(tick_indices)
    ax_heatmap.set_yticklabels(tick_labels, fontsize=9)
    ax_heatmap.set_xlabel("Training Step", fontsize=11)
    ax_heatmap.set_ylabel("Training Step", fontsize=11)
    
    # Horizontal colorbar at bottom
    cbar = plt.colorbar(im, cax=ax_cbar, orientation="horizontal")
    cbar.set_label("Similarity", fontsize=10)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved heatmap to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_diff_heatmap(
    sim_matrix_a: np.ndarray,
    sim_matrix_b: np.ndarray,
    steps: np.ndarray,
    method_a: str,
    method_b: str,
    output_path: Optional[Path] = None,
    metrics: Optional[dict] = None,
    score_metrics: Optional[dict] = None,
    show: bool = True,
    tick_every: int = 1000,
):
    """Plot difference heatmap (A - B) with optional val loss curve above."""
    n = len(steps)
    diff_matrix = sim_matrix_a - sim_matrix_b
    
    # Determine which steps to show as ticks
    tick_indices = []
    tick_labels = []
    for i, step in enumerate(steps):
        if step % tick_every == 0:
            tick_indices.append(i)
            tick_labels.append(f"{step // 1000}k" if step >= 1000 else str(step))
    
    # Create figure with proper layout - colorbar at bottom for width alignment
    has_loss = metrics is not None
    has_scores = score_metrics is not None
    fig_height = 11 + (4 if has_loss else 0) + (4 if has_scores else 0)

    rows = []
    if has_loss:
        rows.append("loss")
    if has_scores:
        rows.append("scores")
    rows.extend(["heatmap", "cbar"])

    height_ratios = [1 if row in {"loss", "scores"} else 4 if row == "heatmap" else 0.15 for row in rows]

    fig = plt.figure(figsize=(10, fig_height))
    gs = fig.add_gridspec(len(rows), 1, height_ratios=height_ratios, hspace=0.25)

    idx = 0
    ax_loss = None
    if has_loss:
        ax_loss = fig.add_subplot(gs[idx])
        idx += 1

    ax_scores = None
    if has_scores:
        ax_scores = fig.add_subplot(gs[idx])
        idx += 1

    ax_heatmap = fig.add_subplot(gs[idx])
    idx += 1
    ax_cbar = fig.add_subplot(gs[idx])
    
    # Plot val loss if available
    if ax_loss is not None and metrics is not None:
        ax_loss.plot(metrics["steps"], metrics["val_loss"], "b-", linewidth=1.5)
        ax_loss.set_ylabel("Val CE Loss", fontsize=11)
        ax_loss.set_xlabel("Training Step", fontsize=11)
        ax_loss.set_title("Validation Loss During Training", fontsize=12, fontweight="bold")
        ax_loss.grid(True, alpha=0.3)
        ax_loss.set_xlim(steps[0], steps[-1])
        
        ax_loss.set_xticks([steps[i] for i in tick_indices])
        ax_loss.set_xticklabels(tick_labels)
    
    if ax_scores is not None and score_metrics is not None:
        series = score_metrics.get("series", {})
        score_steps = score_metrics.get("steps", steps)

        if "2gram_score" in series and np.isfinite(series["2gram_score"]).any():
            ax_scores.plot(
                score_steps,
                series["2gram_score"],
                label="2-gram score",
                marker="o",
                markersize=3,
            )
        if "3gram_score" in series and np.isfinite(series["3gram_score"]).any():
            ax_scores.plot(
                score_steps,
                series["3gram_score"],
                label="3-gram score",
                marker="o",
                markersize=3,
            )
        if "repeat_icl" in series and np.isfinite(series["repeat_icl"]).any():
            ax_scores.plot(
                score_steps,
                series["repeat_icl"],
                label="Repeat ICL (repeat - first)",
                linestyle="--",
                color="tab:red",
            )

        ax_scores.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)
        ax_scores.axhline(y=0.0, color="gray", linestyle=":", alpha=0.4)
        ax_scores.set_ylabel("Score", fontsize=11)
        ax_scores.set_xlabel("Training Step", fontsize=11)
        ax_scores.set_title("N-gram Scores & Repeat ICL", fontsize=12, fontweight="bold")
        ax_scores.grid(True, alpha=0.3)
        ax_scores.set_xlim(steps[0], steps[-1])
        ax_scores.set_xticks([steps[i] for i in tick_indices])
        ax_scores.set_xticklabels(tick_labels)
        ax_scores.legend(loc="upper right")
        
    # Plot diff heatmap
    masked_matrix = np.ma.masked_invalid(diff_matrix)
    
    # Use diverging colormap centered at 0
    vmax = max(abs(np.nanmin(diff_matrix)), abs(np.nanmax(diff_matrix)), 0.1)
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="lightgray")
    
    im = ax_heatmap.imshow(masked_matrix, vmin=-vmax, vmax=vmax, cmap=cmap, aspect="equal")
    
    title_a = get_method_title(method_a).replace(" Similarity", "")
    title_b = get_method_title(method_b).replace(" Similarity", "")
    ax_heatmap.set_title(f"Difference: {title_a} − {title_b}", fontsize=12, fontweight="bold")
    
    ax_heatmap.set_xticks(tick_indices)
    ax_heatmap.set_xticklabels(tick_labels, fontsize=9)
    ax_heatmap.set_yticks(tick_indices)
    ax_heatmap.set_yticklabels(tick_labels, fontsize=9)
    ax_heatmap.set_xlabel("Training Step", fontsize=11)
    ax_heatmap.set_ylabel("Training Step", fontsize=11)
    
    # Horizontal colorbar at bottom
    cbar = plt.colorbar(im, cax=ax_cbar, orientation="horizontal")
    cbar.set_label("Similarity Difference", fontsize=10)
    
    # Add stats annotation
    valid_diff = diff_matrix[~np.isnan(diff_matrix)]
    if len(valid_diff) > 0:
        stats_text = f"Mean: {np.mean(valid_diff):.4f}, Std: {np.std(valid_diff):.4f}"
        ax_heatmap.text(
            0.02, 0.98, stats_text, transform=ax_heatmap.transAxes,
            fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved diff heatmap to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_comparison(
    npz_files: list[Path],
    output_path: Optional[Path] = None,
    metrics: Optional[dict] = None,
    show: bool = True,
    tick_every: int = 1000,
):
    """Plot multiple heatmaps side by side for comparison."""
    n_plots = len(npz_files)
    
    # Load all data
    data_list = []
    for npz_path in npz_files:
        data = np.load(npz_path)
        data_list.append({
            "sim_matrix": data["sim_matrix"],
            "steps": data["steps"],
            "method": str(data["method"]),
        })
    
    steps = data_list[0]["steps"]
    
    # Determine ticks
    tick_indices = []
    tick_labels = []
    for i, step in enumerate(steps):
        if step % tick_every == 0:
            tick_indices.append(i)
            tick_labels.append(f"{step // 1000}k" if step >= 1000 else str(step))
    
    # Create figure
    if metrics is not None:
        fig = plt.figure(figsize=(5 * n_plots, 12))
        gs = fig.add_gridspec(2, n_plots, height_ratios=[1, 3], hspace=0.15)
        
        # Single loss plot spanning all columns
        ax_loss = fig.add_subplot(gs[0, :])
        ax_loss.plot(metrics["steps"], metrics["val_loss"], "b-", linewidth=1.5)
        ax_loss.set_ylabel("Val CE Loss", fontsize=11)
        ax_loss.set_xlabel("Training Step", fontsize=11)
        ax_loss.set_title("Validation Loss During Training", fontsize=12, fontweight="bold")
        ax_loss.grid(True, alpha=0.3)
        ax_loss.set_xlim(steps[0], steps[-1])
        ax_loss.set_xticks([steps[i] for i in tick_indices])
        ax_loss.set_xticklabels(tick_labels)
        
        axes = [fig.add_subplot(gs[1, i]) for i in range(n_plots)]
    else:
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 8))
        if n_plots == 1:
            axes = [axes]
    
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="lightgray")
    
    for ax, data in zip(axes, data_list):
        masked_matrix = np.ma.masked_invalid(data["sim_matrix"])
        im = ax.imshow(masked_matrix, vmin=-1, vmax=1, cmap=cmap, aspect="equal")
        
        title = get_method_title(data["method"])
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_yticks(tick_indices)
        ax.set_yticklabels(tick_labels, fontsize=8)
        ax.set_xlabel("Training Step", fontsize=10)
        ax.set_ylabel("Training Step", fontsize=10)
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    fig.subplots_adjust(hspace=0.15)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved comparison to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot similarity heatmaps from npz files")
    parser.add_argument(
        "--npz", "-n",
        type=str,
        required=True,
        help="Path to similarity_data_*.npz file",
    )
    parser.add_argument(
        "--diff", "-d",
        type=str,
        default=None,
        help="Path to second npz file for diff heatmap (computes npz - diff)",
    )
    parser.add_argument(
        "--metrics", "-m",
        type=str,
        default=None,
        help="Path to metrics.jsonl for val loss curve",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory for n-gram/ICL scoring (defaults to inferred)",
    )
    parser.add_argument(
        "--no-scores",
        action="store_true",
        help="Skip computing/plotting n-gram and repeat ICL scores",
    )
    parser.add_argument(
        "--score-device",
        type=str,
        default=None,
        help="Device for score computation (default: auto)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for saved figure",
    )
    parser.add_argument(
        "--tick-every",
        type=int,
        default=1000,
        help="Show tick every N steps (default: 1000)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot",
    )
    args = parser.parse_args()
    
    npz_path = Path(args.npz)
    data = np.load(npz_path)
    sim_matrix = data["sim_matrix"]
    steps = data["steps"]
    method = str(data["method"])
    
    # Load metrics if provided
    metrics = None
    if args.metrics:
        metrics = load_metrics(Path(args.metrics))

    score_metrics = None
    if not args.no_scores:
        score_metrics = load_or_compute_scores(
            steps,
            npz_path,
            checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
            device=args.score_device,
        )
    
    # Determine output path
    output_path = None
    if args.output:
        output_path = Path(args.output)
    
    if args.diff:
        # Diff heatmap
        diff_path = Path(args.diff)
        diff_data = np.load(diff_path)
        diff_matrix = diff_data["sim_matrix"]
        diff_method = str(diff_data["method"])
        
        # Verify steps match
        if not np.array_equal(steps, diff_data["steps"]):
            print("Warning: Steps don't match between files!")
        
        if output_path is None:
            output_path = npz_path.parent / "images" / f"diff_{method}_vs_{diff_method}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plot_diff_heatmap(
            sim_matrix, diff_matrix, steps,
            method, diff_method,
            output_path=output_path,
            metrics=metrics,
            score_metrics=score_metrics,
            show=not args.no_show,
            tick_every=args.tick_every,
        )
    else:
        # Single heatmap
        if output_path is None:
            output_path = npz_path.parent / "images" / f"heatmap_{method}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plot_single_heatmap(
            sim_matrix, steps, method,
            output_path=output_path,
            metrics=metrics,
            score_metrics=score_metrics,
            show=not args.no_show,
            tick_every=args.tick_every,
        )


if __name__ == "__main__":
    main()
