# %% [markdown]
# # Tensor Difference Analysis for Bilinear Models
#
# This notebook explores:
# 1. Rank/effective rank of hidden covariance across training
# 2. Optimization-based difference decomposition: find minimal C s.t. TN_sim(A, B+C) ≈ 1
# 3. Structural diagnostics for interpreting differences

# %%
import json
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from core import init_model

# %% [markdown]
# ## Part 0: Setup and Utilities

# %%
def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


@dataclass
class BilinearWeights:
    """Container for bilinear layer weights (can be from model or standalone)."""
    w_l: torch.Tensor  # (d_hidden, 2*P)
    w_r: torch.Tensor  # (d_hidden, 2*P)
    w_p: torch.Tensor  # (P, d_hidden)
    
    @property
    def d_hidden(self) -> int:
        return self.w_l.shape[0]
    
    @property
    def input_dim(self) -> int:
        return self.w_l.shape[1]
    
    @property
    def output_dim(self) -> int:
        return self.w_p.shape[0]
    
    @classmethod
    def from_model(cls, model) -> 'BilinearWeights':
        """Extract weights from a Model instance."""
        return cls(
            w_l=model.w_l.detach().clone(),
            w_r=model.w_r.detach().clone(),
            w_p=model.w_p.detach().clone()
        )
    
    @classmethod
    def zeros_like(cls, other: 'BilinearWeights', d_hidden: Optional[int] = None) -> 'BilinearWeights':
        """Create zero weights with same shape (or different d_hidden)."""
        d_h = d_hidden if d_hidden is not None else other.d_hidden
        device = other.w_l.device
        return cls(
            w_l=torch.zeros(d_h, other.input_dim, device=device),
            w_r=torch.zeros(d_h, other.input_dim, device=device),
            w_p=torch.zeros(other.output_dim, d_h, device=device)
        )
    
    @classmethod
    def randn_like(cls, other: 'BilinearWeights', d_hidden: Optional[int] = None, 
                   scale: float = 0.01) -> 'BilinearWeights':
        """Create random weights with same shape."""
        d_h = d_hidden if d_hidden is not None else other.d_hidden
        device = other.w_l.device
        return cls(
            w_l=torch.randn(d_h, other.input_dim, device=device) * scale,
            w_r=torch.randn(d_h, other.input_dim, device=device) * scale,
            w_p=torch.randn(other.output_dim, d_h, device=device) * scale
        )
    
    def to(self, device) -> 'BilinearWeights':
        return BilinearWeights(
            w_l=self.w_l.to(device),
            w_r=self.w_r.to(device),
            w_p=self.w_p.to(device)
        )
    
    def clone(self) -> 'BilinearWeights':
        return BilinearWeights(
            w_l=self.w_l.clone(),
            w_r=self.w_r.clone(),
            w_p=self.w_p.clone()
        )


# %% [markdown]
# ## Part 1: Core Tensor Operations

# %%
def hidden_covariance(A: BilinearWeights, B: BilinearWeights) -> torch.Tensor:
    """
    Compute hidden-space covariance between two bilinear layers.
    
    cov_hid(A, B)_{h1, h2} = (L_A^T L_B)_{h1,h2} * (R_A^T R_B)_{h1,h2}
    
    Returns: (d_hidden_A, d_hidden_B) tensor
    """
    LL = A.w_l @ B.w_l.T  # (d_A, d_B)
    RR = A.w_r @ B.w_r.T  # (d_A, d_B)
    return LL * RR  # Hadamard product


def symmetric_hidden_covariance(A: BilinearWeights, B: BilinearWeights) -> torch.Tensor:
    """
    Symmetric hidden covariance accounting for L<->R symmetry.
    
    core = 0.5 * ((L_A^T L_B)(R_A^T R_B) + (L_A^T R_B)(R_A^T L_B))
    """
    LL = A.w_l @ B.w_l.T
    RR = A.w_r @ B.w_r.T
    LR = A.w_l @ B.w_r.T
    RL = A.w_r @ B.w_l.T
    return 0.5 * (LL * RR + LR * RL)


def output_covariance(A: BilinearWeights, B: BilinearWeights) -> torch.Tensor:
    """
    Compute output-space covariance by folding in projection matrices.
    
    Returns: (P, P) tensor representing correlation between output dimensions.
    """
    hid_cov = symmetric_hidden_covariance(A, B)  # (d_A, d_B)
    # D_A @ hid_cov @ D_B^T
    return A.w_p @ hid_cov @ B.w_p.T  # (P, P)


def tn_inner_product(A: BilinearWeights, B: BilinearWeights) -> torch.Tensor:
    """
    Full tensor network inner product (Frobenius inner product of full tensors).
    
    <A|B> = Tr(D_A @ core @ D_B^T) where core is symmetric hidden covariance
    """
    hid_cov = symmetric_hidden_covariance(A, B)
    DD = A.w_p.T @ B.w_p  # (d_A, d_B)
    return (hid_cov * DD).sum()


def tn_norm(A: BilinearWeights) -> torch.Tensor:
    """Frobenius norm of full tensor."""
    return torch.sqrt(tn_inner_product(A, A))


def tn_similarity(A: BilinearWeights, B: BilinearWeights) -> torch.Tensor:
    """Cosine similarity in tensor space."""
    inner = tn_inner_product(A, B)
    norm_A = tn_norm(A)
    norm_B = tn_norm(B)
    return inner / (norm_A * norm_B + 1e-8)


def tn_distance_squared(A: BilinearWeights, B: BilinearWeights) -> torch.Tensor:
    """Squared Frobenius distance ||T_A - T_B||^2."""
    return (tn_inner_product(A, A) + tn_inner_product(B, B) 
            - 2 * tn_inner_product(A, B))


# %% [markdown]
# ## Part 2: Rank Analysis

# %%
def effective_rank(cov: torch.Tensor, eps: float = 1e-10) -> float:
    """
    Compute effective rank via entropy of normalized eigenvalues.
    
    eff_rank = exp(entropy) where entropy = -sum(p_i * log(p_i))
    and p_i = λ_i / sum(λ)
    
    This gives a continuous measure of "how many dimensions matter".
    """
    # Get eigenvalues (cov should be symmetric)
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = torch.clamp(eigvals, min=0)  # Numerical stability
    
    # Normalize to probability distribution
    total = eigvals.sum()
    if total < eps:
        return 0.0
    
    p = eigvals / total
    p = p[p > eps]  # Only non-zero
    
    # Entropy
    entropy = -(p * torch.log(p)).sum()
    return float(torch.exp(entropy))


def hard_rank(cov: torch.Tensor, tol: float = 1e-5) -> int:
    """Standard matrix rank with tolerance."""
    eigvals = torch.linalg.eigvalsh(cov)
    return int((eigvals.abs() > tol * eigvals.abs().max()).sum())


def analyze_rank_structure(weights: BilinearWeights) -> dict:
    """Compute various rank metrics for a bilinear layer."""
    cov = symmetric_hidden_covariance(weights, weights)
    
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals_sorted = eigvals.flip(0)  # Descending
    
    return {
        'd_hidden': weights.d_hidden,
        'hard_rank': hard_rank(cov),
        'effective_rank': effective_rank(cov),
        'top_eigenvalues': eigvals_sorted[:10].cpu().numpy(),
        'eigenvalue_ratio': float(eigvals_sorted[0] / (eigvals_sorted.sum() + 1e-10)),
        'tn_norm': float(tn_norm(weights)),
    }


def track_rank_across_checkpoints(checkpoints: dict, P: int, d_hidden: int, 
                                   init_model_fn, device=None) -> dict:
    """
    Analyze rank structure across all checkpoints.
    
    Returns dict with arrays indexed by checkpoint step.
    """
    if device is None:
        device = get_device()
    
    steps = sorted(checkpoints.keys())
    results = {
        'steps': steps,
        'hard_rank': [],
        'effective_rank': [],
        'tn_norm': [],
        'top_eigval': [],
        'eigval_concentration': [],
    }
    
    for step in tqdm(steps, desc="Analyzing ranks"):
        model = init_model_fn(p=P, d_hidden=d_hidden).to(device)
        model.load_state_dict(checkpoints[step])
        
        weights = BilinearWeights.from_model(model)
        analysis = analyze_rank_structure(weights)
        
        results['hard_rank'].append(analysis['hard_rank'])
        results['effective_rank'].append(analysis['effective_rank'])
        results['tn_norm'].append(analysis['tn_norm'])
        results['top_eigval'].append(analysis['top_eigenvalues'][0])
        results['eigval_concentration'].append(analysis['eigenvalue_ratio'])
    
    # Convert to arrays
    for key in results:
        if key != 'steps':
            results[key] = np.array(results[key])
    
    return results


# %% [markdown]
# ## Part 3: Difference Optimization
# 
# Find minimal C such that TN_sim(A, B + C) ≈ 1

# %%
class DifferenceModel(nn.Module):
    """
    Learnable difference tensor C.
    
    We parameterize C as a bilinear layer and optimize to make B + C ≈ A.
    """
    def __init__(self, input_dim: int, output_dim: int, d_hidden: int, 
                 init_scale: float = 0.01):
        super().__init__()
        self.w_l = nn.Parameter(torch.randn(d_hidden, input_dim) * init_scale)
        self.w_r = nn.Parameter(torch.randn(d_hidden, input_dim) * init_scale)
        self.w_p = nn.Parameter(torch.randn(output_dim, d_hidden) * init_scale)
    
    def get_weights(self) -> BilinearWeights:
        return BilinearWeights(self.w_l, self.w_r, self.w_p)
    
    def forward(self) -> BilinearWeights:
        return self.get_weights()


def add_bilinear(A: BilinearWeights, B: BilinearWeights) -> BilinearWeights:
    """
    Add two bilinear layers by concatenating hidden dimensions.
    
    T_{A+B} = T_A + T_B in tensor space, achieved by:
    - L_{A+B} = [L_A; L_B] (vertical stack)
    - R_{A+B} = [R_A; R_B]
    - D_{A+B} = [D_A, D_B] (horizontal stack)
    """
    return BilinearWeights(
        w_l=torch.cat([A.w_l, B.w_l], dim=0),
        w_r=torch.cat([A.w_r, B.w_r], dim=0),
        w_p=torch.cat([A.w_p, B.w_p], dim=1),
    )


@dataclass
class OptimizationConfig:
    """Configuration for difference optimization."""
    d_hidden_C: int = 1  # Hidden dim of difference (low = simple explanation)
    lr: float = 0.01
    max_steps: int = 2000
    target_similarity: float = 0.999
    
    # Regularization
    reg_type: str = 'fro'  # 'fro', 'l1', 'nuclear'
    reg_weight: float = 0.0
    
    # Constraints
    freeze_L: bool = False  # Only optimize R and D
    freeze_R: bool = False
    freeze_D: bool = False
    
    # Logging
    log_every: int = 100
    early_stop_patience: int = 200


def optimize_difference(A: BilinearWeights, B: BilinearWeights, 
                        config: OptimizationConfig,
                        verbose: bool = True) -> tuple[DifferenceModel, dict]:
    """
    Find minimal C such that TN_sim(A, B + C) is maximized.
    
    Returns:
        - Trained DifferenceModel
        - Training history dict
    """
    device = A.w_l.device
    
    # Initialize C
    C_model = DifferenceModel(
        input_dim=A.input_dim,
        output_dim=A.output_dim,
        d_hidden=config.d_hidden_C,
        init_scale=0.01
    ).to(device)
    
    # Apply freezing constraints
    if config.freeze_L:
        C_model.w_l.requires_grad_(False)
    if config.freeze_R:
        C_model.w_r.requires_grad_(False)
    if config.freeze_D:
        C_model.w_p.requires_grad_(False)
    
    optimizer = torch.optim.Adam(
        [p for p in C_model.parameters() if p.requires_grad],
        lr=config.lr
    )
    
    history = {
        'step': [],
        'similarity': [],
        'loss': [],
        'C_norm': [],
        'reg': [],
    }
    
    best_sim = -float('inf')
    steps_without_improvement = 0
    
    for step in range(config.max_steps):
        optimizer.zero_grad()
        
        C = C_model.get_weights()
        B_plus_C = add_bilinear(B, C)
        
        # Main objective: maximize similarity
        sim = tn_similarity(A, B_plus_C)
        
        # Regularization on C
        if config.reg_type == 'fro':
            reg = tn_inner_product(C, C)
        elif config.reg_type == 'l1':
            reg = (C.w_l.abs().sum() + C.w_r.abs().sum() + C.w_p.abs().sum())
        elif config.reg_type == 'nuclear':
            cov_C = symmetric_hidden_covariance(C, C)
            reg = torch.linalg.svdvals(cov_C).sum()
        else:
            reg = torch.tensor(0.0, device=device)
        
        loss = -sim + config.reg_weight * reg
        loss.backward()
        optimizer.step()
        
        # Logging
        sim_val = float(sim)
        if step % config.log_every == 0:
            history['step'].append(step)
            history['similarity'].append(sim_val)
            history['loss'].append(float(loss))
            history['C_norm'].append(float(tn_norm(C)))
            history['reg'].append(float(reg))
            
            if verbose:
                print(f"Step {step:5d}: sim={sim_val:.6f}, ||C||={float(tn_norm(C)):.4f}")
        
        # Early stopping
        if sim_val > best_sim + 1e-6:
            best_sim = sim_val
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1
        
        if sim_val >= config.target_similarity:
            if verbose:
                print(f"Reached target similarity at step {step}")
            break
        
        if steps_without_improvement >= config.early_stop_patience:
            if verbose:
                print(f"Early stopping at step {step} (no improvement)")
            break
    
    return C_model, history


# %% [markdown]
# ## Part 4: Diagnostic Tools

# %%
def diagnose_difference(A: BilinearWeights, B: BilinearWeights, 
                        C: BilinearWeights) -> dict:
    """
    Comprehensive diagnostics for the difference C between A and B.
    """
    B_plus_C = add_bilinear(B, C)
    
    diagnostics = {
        # Basic metrics
        'similarity_before': float(tn_similarity(A, B)),
        'similarity_after': float(tn_similarity(A, B_plus_C)),
        'distance_before': float(torch.sqrt(tn_distance_squared(A, B))),
        'distance_after': float(torch.sqrt(tn_distance_squared(A, B_plus_C))),
        
        # Norms
        'norm_A': float(tn_norm(A)),
        'norm_B': float(tn_norm(B)),
        'norm_C': float(tn_norm(C)),
        'norm_B_plus_C': float(tn_norm(B_plus_C)),
        
        # Inner products (alignment analysis)
        'inner_C_A': float(tn_inner_product(C, A)),
        'inner_C_B': float(tn_inner_product(C, B)),
        
        # Rank of C
        'C_d_hidden': C.d_hidden,
        'C_hard_rank': hard_rank(symmetric_hidden_covariance(C, C)),
        'C_effective_rank': effective_rank(symmetric_hidden_covariance(C, C)),
    }
    
    # Alignment interpretation
    # If C·A > 0 and C·B < 0, C adds what A has that B lacks
    diagnostics['C_aligns_with_A'] = diagnostics['inner_C_A'] > 0
    diagnostics['C_opposes_B'] = diagnostics['inner_C_B'] < 0
    
    return diagnostics


def analyze_weight_sparsity(C: BilinearWeights, threshold: float = 0.01) -> dict:
    """Analyze which parts of C are most active."""
    C_norm = tn_norm(C).item()

    # Per-output-class contribution
    D_norms = torch.norm(C.w_p.detach(), dim=1)  # (P,)

    # Per-input-dimension contribution (split by position)
    P = C.w_p.shape[0]
    L_norms_a = torch.norm(C.w_l.detach()[:, :P], dim=0)  # Input position a
    L_norms_b = torch.norm(C.w_l.detach()[:, P:], dim=0)  # Input position b
    R_norms_a = torch.norm(C.w_r.detach()[:, :P], dim=0)
    R_norms_b = torch.norm(C.w_r.detach()[:, P:], dim=0)

    return {
        'output_class_norms': D_norms.cpu().numpy(),
        'top_output_classes': torch.topk(D_norms, min(5, len(D_norms))).indices.cpu().numpy(),
        'L_input_a_norms': L_norms_a.cpu().numpy(),
        'L_input_b_norms': L_norms_b.cpu().numpy(),
        'R_input_a_norms': R_norms_a.cpu().numpy(),
        'R_input_b_norms': R_norms_b.cpu().numpy(),
        'frac_D_above_threshold': float((D_norms > threshold * D_norms.max()).float().mean()),
    }


def alignment_matrix(A: BilinearWeights, B: BilinearWeights) -> torch.Tensor:
    """
    Compute alignment between hidden units of A and B.
    
    Returns (d_A, d_B) matrix showing which units in A correspond to which in B.
    High values on diagonal (after reordering) suggest same features, different order.
    """
    cov = symmetric_hidden_covariance(A, B)
    
    # Normalize by geometric mean of self-covariances
    cov_A = symmetric_hidden_covariance(A, A)
    cov_B = symmetric_hidden_covariance(B, B)
    
    norm_A = torch.sqrt(torch.diag(cov_A) + 1e-8)
    norm_B = torch.sqrt(torch.diag(cov_B) + 1e-8)
    
    alignment = cov / (norm_A[:, None] * norm_B[None, :] + 1e-8)
    return alignment


# %% [markdown]
# ## Part 5: Visualization

# %%
def plot_rank_evolution(rank_results: dict, history: dict = None, 
                        save_path: str = None):
    """Plot rank metrics across training."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    steps = rank_results['steps']
    
    # Effective rank
    ax = axes[0, 0]
    ax.plot(steps, rank_results['effective_rank'], 'b-', linewidth=2)
    ax.axhline(y=rank_results['hard_rank'][0], color='gray', linestyle='--', 
               alpha=0.5, label=f'd_hidden={rank_results["hard_rank"][0]}')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Effective Rank')
    ax.set_title('Effective Rank Evolution')
    ax.legend()
    
    # TN norm
    ax = axes[0, 1]
    ax.plot(steps, rank_results['tn_norm'], 'r-', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('||T||_F')
    ax.set_title('Tensor Frobenius Norm')
    
    # Top eigenvalue concentration
    ax = axes[1, 0]
    ax.plot(steps, rank_results['eigval_concentration'], 'g-', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('λ_max / Σλ')
    ax.set_title('Eigenvalue Concentration')
    
    # Accuracy overlay if history provided
    ax = axes[1, 1]
    if history is not None:
        hist_steps = sorted(history.keys())
        val_acc = [history[s]['val_acc'] for s in hist_steps]
        train_acc = [history[s]['train_acc'] for s in hist_steps]
        ax.plot(hist_steps, train_acc, 'b-', label='Train Acc', alpha=0.7)
        ax.plot(hist_steps, val_acc, 'r-', label='Val Acc', alpha=0.7)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training Progress')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No history provided', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('Training Progress')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_optimization_history(history: dict, save_path: str = None):
    """Plot optimization progress for difference finding."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    steps = history['step']
    
    # Similarity
    ax = axes[0]
    ax.plot(steps, history['similarity'], 'b-', linewidth=2)
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('TN Similarity')
    ax.set_title('Similarity(A, B+C)')
    
    # C norm
    ax = axes[1]
    ax.plot(steps, history['C_norm'], 'r-', linewidth=2)
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('||C||_F')
    ax.set_title('Norm of Difference C')
    
    # Loss
    ax = axes[2]
    ax.plot(steps, history['loss'], 'purple', linewidth=2)
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_alignment_matrix(A: BilinearWeights, B: BilinearWeights,
                          title: str = "Hidden Unit Alignment",
                          save_path: str = None):
    """Visualize alignment between hidden units."""
    alignment = alignment_matrix(A, B)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(alignment.cpu().numpy(), cmap='RdBu_r', 
                   vmin=-1, vmax=1, aspect='auto')
    ax.set_xlabel('Hidden units (B)')
    ax.set_ylabel('Hidden units (A)')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Alignment')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# Note: The original notebook's "Part 6: Running the Analysis" cells (which ran
# exploratory comparisons against ../results) were removed for the bundle —
# plot.py only consumes the definitions above. Re-add if you need them.
