#!/usr/bin/env python3
"""Generate similarity heatmaps between checkpoint pairs.

Usage (from bilinear_attn directory):
    python -m experiments.tn_sim_induction.heatmap \
        --checkpoint-dir runs/<timestamp>/checkpoints \
        --checkpoint-every 1000 \
        --window 5

This loads checkpoints from a directory and computes pairwise cosine similarity.
Supports:
- Window-based computation: only compute similarity for m nearest neighbors
- Incremental caching: load existing matrix and fill in missing entries
- Checkpoint filtering: select checkpoints at specified intervals

Uses MC (Monte Carlo) similarity by default, which is fast and practical.
TN similarity is available but requires massive memory (~8GB+ for small models).

Self-similarity is assumed to be 1.0 (not computed).
"""

import argparse
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from models import AttentionLM
from tn_sim import cosine_similarity as tn_cosine_similarity
from tn_sim.mc_similarity import mc_similarity_gaussian


def parse_step_from_filename(filename: str) -> int:
    """Extract step number from checkpoint filename (e.g., 'step_1000.pt' -> 1000)."""
    match = re.match(r"step_(\d+)\.pt", filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot parse step from filename: {filename}")


def load_checkpoint(cfg: dict, ckpt_path: Path):
    """Load a model from checkpoint (on CPU to save GPU memory).
    
    Args:
        cfg: Model configuration dict
        ckpt_path: Path to checkpoint file
        
    Returns:
        Tuple of (model on CPU, step number)
    """
    model = AttentionLM.from_config(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    # Use step from checkpoint if available, otherwise parse from filename
    step = ckpt.get("step", None)
    if step is None:
        step = parse_step_from_filename(ckpt_path.name)
    return model, step


def find_config(checkpoint_dir: Path) -> Path:
    """Find config.yaml in checkpoint dir or parent run dir."""
    # Check parent directory (typical run structure: runs/<timestamp>/checkpoints/)
    parent = checkpoint_dir.parent
    if (parent / "config.yaml").exists():
        return parent / "config.yaml"
    # Check checkpoint dir itself
    if (checkpoint_dir / "config.yaml").exists():
        return checkpoint_dir / "config.yaml"
    # Check grandparent
    if (parent.parent / "config.yaml").exists():
        return parent.parent / "config.yaml"
    raise FileNotFoundError(f"Could not find config.yaml for {checkpoint_dir}")


def filter_checkpoints(ckpt_files: list[Path], checkpoint_every: int) -> list[Path]:
    """Filter checkpoints to only include those at specified intervals.
    
    Args:
        ckpt_files: List of checkpoint paths
        checkpoint_every: Only include checkpoints where step % checkpoint_every == 0
        
    Returns:
        Filtered list of checkpoint paths
    """
    filtered = []
    for ckpt in ckpt_files:
        step = parse_step_from_filename(ckpt.name)
        if step % checkpoint_every == 0:
            filtered.append(ckpt)
    return sorted(filtered, key=lambda p: parse_step_from_filename(p.name))


def load_existing_matrix(output_dir: Path, method: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load existing similarity matrix and steps if available.
    
    Args:
        output_dir: Directory containing the data file
        method: "mc" or "tn" - used to select the correct file
    
    Returns:
        Tuple of (sim_matrix, steps) or (None, None) if not found
    """
    data_path = output_dir / f"similarity_data_{method}.npz"
    if not data_path.exists():
        return None, None
    
    data = np.load(data_path)
    return data["sim_matrix"], data["steps"]


def merge_matrices(
    old_matrix: Optional[np.ndarray],
    old_steps: Optional[np.ndarray],
    new_steps: np.ndarray,
) -> np.ndarray:
    """Create a new matrix, copying over existing values where steps match.
    
    Missing values are marked as NaN.
    
    Args:
        old_matrix: Previous similarity matrix (or None)
        old_steps: Steps corresponding to old_matrix rows/cols (or None)
        new_steps: Steps for the new matrix
        
    Returns:
        New matrix with existing values copied and missing values as NaN
    """
    n = len(new_steps)
    new_matrix = np.full((n, n), np.nan)
    np.fill_diagonal(new_matrix, 1.0)  # Self-similarity = 1
    
    if old_matrix is None or old_steps is None:
        return new_matrix
    
    # Build step -> index mapping for old matrix
    old_step_to_idx = {s: i for i, s in enumerate(old_steps)}
    
    # Copy existing values
    for i, step_i in enumerate(new_steps):
        for j, step_j in enumerate(new_steps):
            if step_i in old_step_to_idx and step_j in old_step_to_idx:
                old_i = old_step_to_idx[step_i]
                old_j = old_step_to_idx[step_j]
                if not np.isnan(old_matrix[old_i, old_j]):
                    new_matrix[i, j] = old_matrix[old_i, old_j]
    
    return new_matrix


def compute_pairwise_similarity(
    models: list,
    steps: list[int],
    sim_matrix: np.ndarray,
    device: str,
    cfg: dict,
    method: str = "mc",
    mc_samples: int = 4000,
    window: Optional[int] = None,
    output_path: Optional[Path] = None,
) -> np.ndarray:
    """Compute pairwise cosine similarity matrix.
    
    Only computes missing entries (NaN values). Self-similarity is 1.0.
    Models are kept on CPU and moved to GPU only during computation to save memory.
    Saves incrementally after each pair computation if output_path is provided.
    
    Args:
        models: List of loaded models (on CPU)
        steps: List of step numbers corresponding to models
        sim_matrix: Existing matrix with NaN for missing entries
        device: Device for computation
        cfg: Model config dict (needed for MC similarity)
        method: "mc" (Monte Carlo, fast) or "tn" (tensor network, slow/memory-intensive)
        mc_samples: Number of samples for MC similarity
        window: If set, only compute similarity for pairs within this distance
                (i.e., |i - j| <= window). None means compute all pairs.
        output_path: Path to save npz file incrementally (optional)
        
    Returns:
        Updated similarity matrix
    """
    n = len(models)
    computed = 0
    skipped = 0
    
    vocab_size = cfg["model"]["vocab_size"]
    n_ctx = cfg["model"]["n_ctx"]
    
    # Build list of pairs to compute
    pairs_to_compute = []
    for i in range(n):
        for j in range(i + 1, n):
            if window is not None and (j - i) > window:
                continue
            if np.isnan(sim_matrix[i, j]):
                pairs_to_compute.append((i, j))
            else:
                skipped += 1
    
    if skipped > 0:
        print(f"Skipping {skipped} cached pairs, computing {len(pairs_to_compute)} remaining")
    
    for i, j in tqdm(pairs_to_compute, desc=f"{method.upper()} similarity", unit="pair"):
            
            # Move models to device for computation
            model_i = models[i].to(device)
            model_j = models[j].to(device)
            
            try:
                if method == "mc":
                    sim = mc_similarity_gaussian(
                        model_i, model_j,
                        vocab_size=vocab_size,
                        n_ctx=n_ctx,
                        device=device,
                        n_samples=mc_samples,
                    )
                else:  # tn
                    # TN similarity uses numpy backend internally, so GPU is fine
                    sim = tn_cosine_similarity(model_i, model_j, device=device, dtype=torch.float64)
                
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim  # Symmetric
                computed += 1
                print(f"  sim(step {steps[i]}, step {steps[j]}) = {sim:.4f}")
                
                # Save incrementally after each pair
                if output_path is not None:
                    np.savez(
                        output_path,
                        sim_matrix=sim_matrix,
                        steps=np.array(steps),
                        method=method,
                    )
            finally:
                # Move back to CPU and clear GPU memory
                models[i] = model_i.cpu()
                models[j] = model_j.cpu()
                del model_i, model_j
                if device != "cpu":
                    torch.cuda.empty_cache()
    
    print(f"Computed {computed} new pairs, skipped {skipped} cached pairs")
    return sim_matrix


def plot_heatmap(
    sim_matrix: np.ndarray,
    step_labels: list[str],
    title: str = "TN Cosine Similarity Heatmap",
    output_path: Optional[Path] = None,
    show: bool = True,
):
    """Plot similarity heatmap.
    
    NaN values (not computed) are shown in gray.
    
    Args:
        sim_matrix: n x n similarity matrix (may contain NaN)
        step_labels: Labels for each checkpoint (e.g., step numbers)
        title: Plot title
        output_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    n = len(step_labels)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use masked array to handle NaN values (shown as gray)
    masked_matrix = np.ma.masked_invalid(sim_matrix)
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="lightgray")
    
    im = ax.imshow(masked_matrix, vmin=-1, vmax=1, cmap=cmap, aspect="equal")
    
    ax.set_title(title)
    ax.set_xticks(range(n))
    ax.set_xticklabels(step_labels, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(step_labels)
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Checkpoint step")
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved heatmap to {output_path}")
    
    if show:
        plt.show()
    
    return fig


def generate_heatmap(
    checkpoint_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
    checkpoint_every: Optional[int] = None,
    window: Optional[int] = None,
    method: str = "mc",
    mc_samples: int = 4000,
    show: bool = True,
) -> tuple[np.ndarray, list[int]]:
    """Generate cosine similarity heatmap from checkpoint directory.
    
    Supports incremental computation: if a previous matrix exists, only missing
    entries are computed. Use `window` to limit computation to nearby checkpoints.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files (step_*.pt)
        output_dir: Directory to save outputs (default: checkpoint_dir parent)
        device: Device for computation (default: auto-detect)
        checkpoint_every: Only use checkpoints where step % checkpoint_every == 0.
                          Can be decreased later to add more checkpoints.
        window: Only compute similarity for pairs within this index distance.
                E.g., window=5 means only compute sim(i, j) where |i-j| <= 5.
                Can be increased later to compute more pairs.
        method: "mc" (Monte Carlo, fast) or "tn" (tensor network, slow/memory-intensive)
        mc_samples: Number of samples for MC similarity
        show: Whether to display the plot
        
    Returns:
        Tuple of (similarity matrix, list of step numbers)
    """
    checkpoint_dir = Path(checkpoint_dir)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Find and load config
    config_path = find_config(checkpoint_dir)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    print(f"Loaded config from {config_path}")
    
    # Find checkpoints
    ckpt_files = sorted(
        checkpoint_dir.glob("step_*.pt"),
        key=lambda p: parse_step_from_filename(p.name),
    )
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    print(f"Found {len(ckpt_files)} total checkpoints")
    
    # Filter by checkpoint_every
    if checkpoint_every is not None:
        ckpt_files = filter_checkpoints(ckpt_files, checkpoint_every)
        print(f"Using {len(ckpt_files)} checkpoints (every {checkpoint_every} steps)")
    
    # Determine output directory
    if output_dir is None:
        output_dir = checkpoint_dir.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing matrix if available (method-specific file)
    old_matrix, old_steps = load_existing_matrix(output_dir, method)
    if old_matrix is not None:
        print(f"Loaded existing {method.upper()} matrix with {len(old_steps)} checkpoints")
    
    # Load all models (on CPU to save GPU memory)
    models = []
    steps = []
    for ckpt_path in ckpt_files:
        model, step = load_checkpoint(cfg, ckpt_path)
        models.append(model)
        steps.append(step)
        print(f"  Loaded step {step}")
    
    steps_array = np.array(steps)
    
    # Merge with existing matrix (copies known values, marks unknown as NaN)
    sim_matrix = merge_matrices(old_matrix, old_steps, steps_array)
    
    # Determine output paths (method-specific filenames)
    heatmap_path = output_dir / f"similarity_heatmap_{method}.png"
    data_path = output_dir / f"similarity_data_{method}.npz"
    
    # Compute missing pairwise similarities (saves incrementally to data_path)
    print(f"\nComputing {method.upper()} cosine similarities (window={window})...")
    sim_matrix = compute_pairwise_similarity(
        models, steps, sim_matrix, device, cfg,
        method=method, mc_samples=mc_samples, window=window,
        output_path=data_path,
    )
    
    # Plot and save
    step_labels = [str(s) for s in steps]
    
    plot_heatmap(
        sim_matrix,
        step_labels,
        title=f"{method.upper()} Cosine Similarity Between Checkpoints",
        output_path=heatmap_path,
        show=show,
    )
    
    # Save raw data
    np.savez(
        data_path,
        sim_matrix=sim_matrix,
        steps=steps_array,
        method=method,
    )
    print(f"Saved raw data to {data_path}")
    
    return sim_matrix, steps


def main():
    parser = argparse.ArgumentParser(description="Generate cosine similarity heatmap")
    parser.add_argument(
        "--checkpoint-dir", "-c",
        type=str,
        required=True,
        help="Directory containing checkpoint files (step_*.pt)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Directory to save outputs (default: checkpoint dir parent)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for computation (default: auto-detect)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Only use checkpoints where step %% checkpoint_every == 0",
    )
    parser.add_argument(
        "--window", "-w",
        type=int,
        default=None,
        help="Only compute similarity for pairs within this index distance",
    )
    parser.add_argument(
        "--method", "-m",
        type=str,
        default="mc",
        choices=["mc", "tn"],
        help="Similarity method: 'mc' (Monte Carlo, fast) or 'tn' (tensor network, slow)",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=4000,
        help="Number of samples for MC similarity (default: 4000)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot (just save)",
    )
    args = parser.parse_args()
    
    generate_heatmap(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        device=args.device,
        checkpoint_every=args.checkpoint_every,
        window=args.window,
        method=args.method,
        mc_samples=args.mc_samples,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
