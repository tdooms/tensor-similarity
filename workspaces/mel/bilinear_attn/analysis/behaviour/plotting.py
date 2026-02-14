"""Plotting functions for visualizing training metrics.

Provides functions to plot:
- Loss over time
- Bigram scores over time
- N-gram scores over time
- Combined metrics plot
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def load_metrics(path: Union[str, Path]) -> List[Dict]:
    """Load metrics from a JSONL file.
    
    Args:
        path: Path to the metrics file
        
    Returns:
        List of metric dictionaries
    """
    metrics = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))
    return metrics


def extract_series(metrics: List[Dict], key: str) -> Tuple[List, List]:
    """Extract a metric series from metrics list.
    
    Args:
        metrics: List of metric dictionaries
        key: Metric key to extract
        
    Returns:
        Tuple of (steps, values)
    """
    steps = []
    values = []
    for m in metrics:
        if key in m and "step" in m:
            steps.append(m["step"])
            values.append(m[key])
    return steps, values


def plot_loss(
    metrics: List[Dict],
    save_path: Optional[str] = None,
    title: str = "Loss Over Training",
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot training and validation loss over time.
    
    Args:
        metrics: List of metric dictionaries
        save_path: Path to save the figure (optional)
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract loss series
    train_steps, train_loss = extract_series(metrics, "train_loss")
    val_steps, val_loss = extract_series(metrics, "val_loss")
    
    if train_steps:
        ax.plot(train_steps, train_loss, label="Train Loss", alpha=0.8)
    if val_steps:
        ax.plot(val_steps, val_loss, label="Val Loss", marker='o', markersize=4)
    
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_bigram_score(
    metrics: List[Dict],
    save_path: Optional[str] = None,
    title: str = "Bigram Score Over Training",
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot bigram score and entropy over time.
    
    Args:
        metrics: List of metric dictionaries
        save_path: Path to save the figure (optional)
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract bigram series
    score_steps, bigram_score = extract_series(metrics, "bigram_score")
    entropy_steps, bigram_entropy = extract_series(metrics, "bigram_entropy")
    gap_steps, bigram_gap = extract_series(metrics, "bigram_gap")
    
    if score_steps:
        ax.plot(score_steps, bigram_score, label="Bigram Score", marker='o', markersize=4)
    if entropy_steps:
        ax.plot(entropy_steps, bigram_entropy, label="Bigram Entropy (baseline)", 
                linestyle='--', alpha=0.7)
    
    ax.set_xlabel("Step")
    ax.set_ylabel("Score (nats)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_ngram_scores(
    metrics: List[Dict],
    save_path: Optional[str] = None,
    title: str = "N-gram Scores Over Training",
    figsize: Tuple[int, int] = (12, 8),
    max_n: int = 5,
) -> plt.Figure:
    """Plot n-gram scores for different n values.
    
    Args:
        metrics: List of metric dictionaries
        save_path: Path to save the figure (optional)
        title: Plot title
        figsize: Figure size
        max_n: Maximum n to plot
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    colors = plt.cm.viridis(np.linspace(0, 1, max_n - 1))
    
    # Top plot: n-gram losses
    ax1 = axes[0]
    for i, n in enumerate(range(2, max_n + 1)):
        key = f"{n}gram_loss"
        steps, values = extract_series(metrics, key)
        if steps:
            ax1.plot(steps, values, label=f"{n}-gram loss", color=colors[i-1], 
                    marker='o', markersize=3)
    
    ax1.set_ylabel("Loss")
    ax1.set_title("N-gram Losses")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: n-gram scores (ratios)
    ax2 = axes[1]
    for i, n in enumerate(range(2, max_n + 1)):
        key = f"{n}gram_score"
        steps, values = extract_series(metrics, key)
        if steps:
            ax2.plot(steps, values, label=f"{n}-gram score", color=colors[i-1],
                    marker='o', markersize=3)
    
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label="Baseline (1.0)")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Score (ratio)")
    ax2.set_title("N-gram Scores (validation loss / n-gram loss)")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_metrics(
    metrics: List[Dict],
    metric_name: str,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    ylabel: Optional[str] = None,
) -> plt.Figure:
    """Plot a single metric over time.
    
    Args:
        metrics: List of metric dictionaries
        metric_name: Name of the metric to plot
        save_path: Path to save the figure (optional)
        title: Plot title (defaults to metric name)
        figsize: Figure size
        ylabel: Y-axis label
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    steps, values = extract_series(metrics, metric_name)
    
    if steps:
        ax.plot(steps, values, marker='o', markersize=4)
    
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel or metric_name)
    ax.set_title(title or f"{metric_name} Over Training")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_ablated_loss(
    metrics: List[Dict],
    save_path: Optional[str] = None,
    title: str = "Validation Loss vs Position-Ablated Loss",
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot validation loss and position-ablated loss on the same axes.
    
    Args:
        metrics: List of metric dictionaries
        save_path: Path to save the figure (optional)
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    val_steps, val_loss = extract_series(metrics, "val_loss")
    abl_steps, abl_loss = extract_series(metrics, "ablated_loss")
    
    if val_steps:
        ax.plot(val_steps, val_loss, label="Val Loss", marker='o', markersize=3)
    if abl_steps:
        ax.plot(abl_steps, abl_loss, label="Ablated Loss (no RoPE)",
                marker='s', markersize=3, linestyle='--')
    
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_all_metrics(
    metrics: List[Dict],
    save_path: Optional[str] = None,
    title: str = "Training Metrics Overview",
    figsize: Tuple[int, int] = (16, 16),
    max_n: int = 4,
) -> plt.Figure:
    """Create a combined plot with all metrics.
    
    Args:
        metrics: List of metric dictionaries
        save_path: Path to save the figure (optional)
        title: Plot title
        figsize: Figure size
        max_n: Maximum n for n-gram plots
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    # 1. Loss plot (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    train_steps, train_loss = extract_series(metrics, "train_loss")
    val_steps, val_loss = extract_series(metrics, "val_loss")
    
    if train_steps:
        ax1.plot(train_steps, train_loss, label="Train", alpha=0.8)
    if val_steps:
        ax1.plot(val_steps, val_loss, label="Val", marker='o', markersize=3)
    
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Bigram score + entropy (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    score_steps, bigram_score = extract_series(metrics, "bigram_score")
    entropy_steps, bigram_entropy = extract_series(metrics, "bigram_entropy")
    
    if score_steps:
        ax2.plot(score_steps, bigram_score, label="Bigram Score", marker='o', markersize=3)
    if entropy_steps:
        ax2.plot(entropy_steps, bigram_entropy, label="Bigram Entropy", 
                linestyle='--', alpha=0.7)
    
    ax2.set_xlabel("Step")
    ax2.set_ylabel("nats")
    ax2.set_title("Bigram Score & Entropy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. N-gram losses (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, max_n - 1))
    
    for i, n in enumerate(range(2, max_n + 1)):
        key = f"{n}gram_loss"
        steps, values = extract_series(metrics, key)
        if steps:
            ax3.plot(steps, values, label=f"{n}-gram", color=colors[i], 
                    marker='o', markersize=3)
    
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Loss")
    ax3.set_title("N-gram Losses")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. N-gram scores (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    
    for i, n in enumerate(range(2, max_n + 1)):
        key = f"{n}gram_score"
        steps, values = extract_series(metrics, key)
        if steps:
            ax4.plot(steps, values, label=f"{n}-gram", color=colors[i],
                    marker='o', markersize=3)
    
    ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Score (ratio)")
    ax4.set_title("N-gram Scores")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Position losses (row 3 left)
    ax5 = fig.add_subplot(gs[2, 0])
    
    for i, n in enumerate(range(2, max_n + 1)):
        key = f"position_{n}_loss"
        steps, values = extract_series(metrics, key)
        if steps:
            ax5.plot(steps, values, label=f"pos {n}", color=colors[i],
                    marker='o', markersize=3)
    
    ax5.set_xlabel("Step")
    ax5.set_ylabel("Loss")
    ax5.set_title("Position Losses (validation)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Bigram gap (row 3 right)
    ax6 = fig.add_subplot(gs[2, 1])
    gap_steps, bigram_gap = extract_series(metrics, "bigram_gap")
    
    if gap_steps:
        ax6.plot(gap_steps, bigram_gap, marker='o', markersize=3, color='purple')
        ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    ax6.set_xlabel("Step")
    ax6.set_ylabel("Gap (nats)")
    ax6.set_title("Bigram Gap (score - entropy)")
    ax6.grid(True, alpha=0.3)
    
    # 7. Val loss vs Ablated loss (row 4 left)
    ax7 = fig.add_subplot(gs[3, 0])
    val_steps2, val_loss2 = extract_series(metrics, "val_loss")
    abl_steps, abl_loss = extract_series(metrics, "ablated_loss")
    
    if val_steps2:
        ax7.plot(val_steps2, val_loss2, label="Val Loss", marker='o', markersize=3)
    if abl_steps:
        ax7.plot(abl_steps, abl_loss, label="Ablated Loss (no RoPE)",
                marker='s', markersize=3, linestyle='--')
    
    ax7.set_xlabel("Step")
    ax7.set_ylabel("Loss")
    ax7.set_title("Val Loss vs Position-Ablated Loss")
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Ablation gap (row 4 right)
    ax8 = fig.add_subplot(gs[3, 1])
    agap_steps, agap = extract_series(metrics, "ablation_gap")
    
    if agap_steps:
        ax8.plot(agap_steps, agap, marker='o', markersize=3, color='teal')
        ax8.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    ax8.set_xlabel("Step")
    ax8.set_ylabel("Gap (loss units)")
    ax8.set_title("Ablation Gap (ablated - val)")
    ax8.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, y=1.02)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_all_plots(
    metrics_path: Union[str, Path],
    output_dir: Union[str, Path],
    max_n: int = 4,
):
    """Create and save all individual plots and combined plot.
    
    Args:
        metrics_path: Path to metrics JSONL file
        output_dir: Directory to save plots
        max_n: Maximum n for n-gram plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    metrics = load_metrics(metrics_path)
    
    # Create individual plots
    plot_loss(metrics, save_path=output_dir / "loss.png")
    plt.close()
    
    plot_bigram_score(metrics, save_path=output_dir / "bigram_score.png")
    plt.close()
    
    plot_ngram_scores(metrics, save_path=output_dir / "ngram_scores.png", max_n=max_n)
    plt.close()
    
    plot_ablated_loss(metrics, save_path=output_dir / "ablated_loss.png")
    plt.close()
    
    # Create combined plot
    plot_all_metrics(metrics, save_path=output_dir / "all_metrics.png", max_n=max_n)
    plt.close()
    
    print(f"Plots saved to {output_dir}")
