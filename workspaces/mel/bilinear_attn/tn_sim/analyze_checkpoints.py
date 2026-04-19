"""Analyze TN similarity across training checkpoints.

This script loads checkpoints from a training run and computes pairwise
TN similarity, producing a heatmap showing how the model evolves during training.

Usage:
    python -m tn_sim.analyze_checkpoints --run-dir experiments/induction_heads/runs/2024-01-01_123456_induction
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import yaml
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import AttentionLM
from tn_sim import compute_similarity, cosine_similarity


def load_checkpoint_model(checkpoint_path, config, device="cpu"):
    """Load a model from a checkpoint file."""
    model = AttentionLM.from_config(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model = model.to(device)
    
    # Set up TN components
    model.setup_tn_components()
    
    return model, checkpoint.get("step", 0)


def compute_similarity_matrix(models, steps, show_progress=True):
    """Compute pairwise TN similarity matrix.
    
    Args:
        models: List of AttentionLM models
        steps: List of step numbers corresponding to each model
        show_progress: Whether to show progress bar
        
    Returns:
        similarity_matrix: (n_models, n_models) array of cosine similarities
        steps: List of step numbers
    """
    n = len(models)
    sim_matrix = np.zeros((n, n))
    
    iterator = tqdm(total=n*(n+1)//2, desc="Computing similarities") if show_progress else None
    
    for i in range(n):
        for j in range(i, n):
            state = compute_similarity(models[i], models[j])
            sim = cosine_similarity(state)
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim
            
            if iterator is not None:
                iterator.update(1)
    
    if iterator is not None:
        iterator.close()
    
    return sim_matrix, steps


def plot_similarity_heatmap(sim_matrix, steps, output_path=None, title=None):
    """Plot similarity heatmap with step numbers on axes.
    
    Args:
        sim_matrix: (n, n) similarity matrix
        steps: List of step numbers
        output_path: Path to save figure (if None, displays instead)
        title: Custom title for the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(sim_matrix, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
    
    # Set ticks to show step numbers
    n_steps = len(steps)
    
    # Show all ticks if <= 20 checkpoints, otherwise show every nth
    if n_steps <= 20:
        tick_indices = list(range(n_steps))
        tick_labels = [str(s) for s in steps]
    else:
        # Show ~15 ticks evenly spaced
        n_ticks = min(15, n_steps)
        tick_indices = [int(i * (n_steps - 1) / (n_ticks - 1)) for i in range(n_ticks)]
        tick_labels = [str(steps[i]) for i in tick_indices]
    
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax.set_yticks(tick_indices)
    ax.set_yticklabels(tick_labels)
    
    # Labels
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Training Step', fontsize=12)
    
    if title is None:
        title = 'TN Similarity Across Training Checkpoints'
    ax.set_title(title, fontsize=14, pad=20)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cosine Similarity', rotation=270, labelpad=20, fontsize=12)
    
    # Add grid for readability
    ax.set_xticks(np.arange(n_steps) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_steps) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {output_path}")
    else:
        plt.show()
    
    return fig, ax


def plot_trajectory(sim_matrix, steps, reference_idx=0, output_path=None):
    """Plot similarity trajectory relative to a reference checkpoint.
    
    Args:
        sim_matrix: (n, n) similarity matrix
        steps: List of step numbers
        reference_idx: Index of reference checkpoint (default: first checkpoint)
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract similarity to reference checkpoint
    similarities = sim_matrix[reference_idx, :]
    
    ax.plot(steps, similarities, 'o-', linewidth=2, markersize=6)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect similarity')
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel(f'Cosine Similarity to Step {steps[reference_idx]}', fontsize=12)
    ax.set_title('Training Trajectory: Similarity to Initial Checkpoint', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved trajectory plot to {output_path}")
    else:
        plt.show()
    
    return fig, ax


def main():
    parser = argparse.ArgumentParser(description="Analyze TN similarity across checkpoints")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to run directory containing checkpoints/")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: cuda if available)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save plots (default: run_dir/tn_similarity)")
    parser.add_argument("--max-checkpoints", type=int, default=None,
                        help="Maximum number of checkpoints to analyze (default: all)")
    parser.add_argument("--reference-step", type=int, default=None,
                        help="Reference step for trajectory plot (default: first checkpoint)")
    args = parser.parse_args()
    
    # Setup
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise ValueError(f"Run directory not found: {run_dir}")
    
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise ValueError(f"Checkpoints directory not found: {checkpoints_dir}")
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise ValueError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Validate model is TN-compatible
    model_cfg = config["model"]
    attn_type = model_cfg.get("attn_type", "quadratic")
    norm_places = model_cfg.get("norm_places", [])
    
    if attn_type not in ["bilinear", "quadratic"]:
        raise ValueError(
            f"Model uses attn_type='{attn_type}' which is not TN-compatible. "
            "Only 'bilinear' and 'quadratic' are supported."
        )
    
    if "pre_layer" in norm_places:
        raise ValueError(
            "Model uses pre_layer normalization which breaks TN structure. "
            "Use norm_places=['pre_unembed'] or [] for TN-clean models."
        )
    
    print(f"Model: {attn_type} attention, {model_cfg['n_layers']}L, d={model_cfg['d_model']}")
    
    # Find all checkpoint files
    checkpoint_files = sorted(checkpoints_dir.glob("step_*.pt"))
    if len(checkpoint_files) == 0:
        raise ValueError(f"No checkpoint files found in {checkpoints_dir}")
    
    print(f"Found {len(checkpoint_files)} checkpoints")
    
    # Limit number of checkpoints if requested
    if args.max_checkpoints is not None and len(checkpoint_files) > args.max_checkpoints:
        # Sample evenly across training
        indices = np.linspace(0, len(checkpoint_files) - 1, args.max_checkpoints, dtype=int)
        checkpoint_files = [checkpoint_files[i] for i in indices]
        print(f"Analyzing {len(checkpoint_files)} checkpoints (sampled)")
    
    # Load all models
    print("Loading checkpoints...")
    models = []
    steps = []
    
    for ckpt_file in tqdm(checkpoint_files, desc="Loading"):
        model, step = load_checkpoint_model(ckpt_file, config, device=device)
        models.append(model)
        steps.append(step)
    
    print(f"Loaded {len(models)} models from steps {steps[0]} to {steps[-1]}")
    
    # Compute similarity matrix
    print("\nComputing TN similarity matrix...")
    sim_matrix, steps = compute_similarity_matrix(models, steps, show_progress=True)
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = run_dir / "tn_similarity"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save similarity matrix
    np.save(output_dir / "similarity_matrix.npy", sim_matrix)
    np.save(output_dir / "steps.npy", np.array(steps))
    print(f"\nSaved similarity matrix to {output_dir}")
    
    # Plot heatmap
    print("\nGenerating heatmap...")
    run_name = run_dir.name
    plot_similarity_heatmap(
        sim_matrix,
        steps,
        output_path=output_dir / "similarity_heatmap.png",
        title=f"TN Similarity: {run_name}"
    )
    
    # Plot trajectory relative to reference
    if args.reference_step is not None:
        # Find closest checkpoint to reference step
        ref_idx = min(range(len(steps)), key=lambda i: abs(steps[i] - args.reference_step))
    else:
        ref_idx = 0  # First checkpoint
    
    print(f"\nGenerating trajectory plot (reference: step {steps[ref_idx]})...")
    plot_trajectory(
        sim_matrix,
        steps,
        reference_idx=ref_idx,
        output_path=output_dir / "similarity_trajectory.png"
    )
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    
    # Diagonal should be 1.0 (self-similarity)
    diag_mean = np.diag(sim_matrix).mean()
    print(f"Mean self-similarity: {diag_mean:.6f} (should be 1.0)")
    
    # Off-diagonal statistics
    mask = ~np.eye(len(steps), dtype=bool)
    off_diag = sim_matrix[mask]
    print(f"Cross-checkpoint similarity:")
    print(f"  Mean:   {off_diag.mean():.4f}")
    print(f"  Median: {np.median(off_diag):.4f}")
    print(f"  Min:    {off_diag.min():.4f}")
    print(f"  Max:    {off_diag.max():.4f}")
    
    # Similarity to initial checkpoint
    init_sims = sim_matrix[0, 1:]
    print(f"\nSimilarity to initial checkpoint (step {steps[0]}):")
    print(f"  Final: {init_sims[-1]:.4f} (step {steps[-1]})")
    print(f"  Min:   {init_sims.min():.4f}")
    
    # Consecutive checkpoint similarity (how much model changes per step)
    consecutive_sims = [sim_matrix[i, i+1] for i in range(len(steps)-1)]
    print(f"\nConsecutive checkpoint similarity:")
    print(f"  Mean:   {np.mean(consecutive_sims):.4f}")
    print(f"  Min:    {np.min(consecutive_sims):.4f}")
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
