"""TN similarity endpoint using main codebase implementation.

This module provides the interface for computing exact Gaussian functional
similarity between AttentionLM models using the main codebase's TN similarity
algorithm (src/components/similarity.py).

Usage:
    from models import AttentionLM
    from tn_sim.similarity import compute_tn_similarity, cosine_similarity
    
    model_A = AttentionLM.from_config(cfg_A)
    model_B = AttentionLM.from_config(cfg_B)
    
    # Compute cosine similarity (most common use case)
    sim = cosine_similarity(model_A, model_B)
    
    # Or get full State object for more detailed analysis
    state = compute_tn_similarity(model_A, model_B)

Limitations:
    - Only supports models with norm_type='none' and norm_places=[]
    - Only supports 'bilinear' and 'quadratic' attention types
    - Does not support use_rmsnorm_qk=True
    - Exact for Gaussian inputs (no approximations)

Performance:
    - ~20 seconds for a 2-layer model with n_ctx=8, d_model=16
    - Bottleneck is the 945 Wick matchings per attention layer
    - Expression caching helps with repeated computations
"""

import torch
from typing import Union

from src.components.similarity import similarity as _compute_similarity, State
from models.components.model import AttentionLMComponent, _validate_model_for_tn_similarity


def _as_component(model):
    if isinstance(model, AttentionLMComponent):
        return model
    _validate_model_for_tn_similarity(model)
    return AttentionLMComponent.from_trained_model(model)


def compute_tn_similarity(
    model_A,
    model_B,
    device: str = None,
    dtype: torch.dtype = None,
) -> State:
    """Compute exact TN similarity between two AttentionLM models.
    
    Uses the main codebase's TN similarity algorithm which:
    1. Propagates second-moment matrices through layers
    2. Uses Isserlis theorem for polynomial expectations
    3. Handles residual connections via term decomposition
    
    Args:
        model_A: First AttentionLM model
        model_B: Second AttentionLM model
        device: Device for computation (default: model's device)
        dtype: Data type for computation (default: model's dtype, recommend float64)
        
    Returns:
        State object containing:
            - S_aa: Second moment E[f_A(x) f_A(x)^T]
            - S_bb: Second moment E[f_B(x) f_B(x)^T]
            - S_ab: Cross moment E[f_A(x) f_B(x)^T]
        
    Raises:
        ValueError: If models have incompatible configurations for TN similarity
    """
    # Convert to Component-compatible versions (skip conversion if already components)
    comp_A = _as_component(model_A)
    comp_B = _as_component(model_B)

    # Check model compatibility
    _validate_model_compatibility(comp_A, comp_B)
    
    # Determine device and dtype
    if device is None:
        device = next(comp_A.parameters()).device
    if dtype is None:
        dtype = next(comp_A.parameters()).dtype
    
    # Move to specified device/dtype
    comp_A = comp_A.to(device=device, dtype=dtype)
    comp_B = comp_B.to(device=device, dtype=dtype)
    
    # Compute similarity using main codebase algorithm
    state = _compute_similarity(comp_A, comp_B)
    
    return state


def cosine_similarity(
    model_A,
    model_B,
    device: str = None,
    dtype: torch.dtype = torch.float64,
) -> float:
    """Compute cosine similarity between two AttentionLM models.
    
    This is the most common use case: a single scalar measuring how similar
    two models are in their functional behavior.
    
    Args:
        model_A: First AttentionLM model
        model_B: Second AttentionLM model
        device: Device for computation (default: model's device)
        dtype: Data type for computation (default: float64 for numerical stability)
        
    Returns:
        Cosine similarity in [-1, 1], where:
            - 1.0 = identical functions
            - 0.0 = orthogonal functions
            - -1.0 = opposite functions
        
    Raises:
        ValueError: If models have incompatible configurations
    """
    state = compute_tn_similarity(model_A, model_B, device=device, dtype=dtype)
    return _cosine_from_state(state)


def inner_product(
    model_A,
    model_B,
    device: str = None,
    dtype: torch.dtype = torch.float64,
) -> float:
    """Compute inner product E[f_A(x)^T f_B(x)] between two models.
    
    Args:
        model_A: First AttentionLM model
        model_B: Second AttentionLM model
        device: Device for computation
        dtype: Data type for computation
        
    Returns:
        Inner product (unnormalized similarity)
    """
    state = compute_tn_similarity(model_A, model_B, device=device, dtype=dtype)
    return _inner_product_from_state(state)


def self_similarity(
    model,
    device: str = None,
    dtype: torch.dtype = torch.float64,
) -> float:
    """Compute self-similarity (should be 1.0 for cosine).
    
    Useful for validation: self-similarity should always be exactly 1.0.
    
    Args:
        model: AttentionLM model
        device: Device for computation
        dtype: Data type for computation
        
    Returns:
        Cosine self-similarity (should be 1.0)
    """
    return cosine_similarity(model, model, device=device, dtype=dtype)


def _cosine_from_state(state: State) -> float:
    """Extract cosine similarity from State object.
    
    Computes: tr(S_ab) / sqrt(tr(S_aa) * tr(S_bb))
    
    Only uses the non-constant dimensions (excludes bias/constant row/col).
    """
    def trace(S):
        # S has shape (n_ctx, d+1, n_ctx, d+1)
        # We want tr(S[:, 1:, :, 1:]) = sum over diagonal positions
        return torch.einsum('ijij->', S[:, 1:, :, 1:])
    
    tr_aa = trace(state.S_aa)
    tr_bb = trace(state.S_bb)
    tr_ab = trace(state.S_ab)
    
    denom = (tr_aa * tr_bb).sqrt()
    if denom < 1e-30:
        return 0.0
    
    return (tr_ab / denom).item()


def _inner_product_from_state(state: State) -> float:
    """Extract inner product from State object."""
    return torch.einsum('ijij->', state.S_ab[:, 1:, :, 1:]).item()


def _validate_model_compatibility(model_A, model_B):
    """Validate that two models are compatible for similarity computation.
    
    Models must have the same architecture (vocab_size, n_ctx, d_model, etc.)
    to compute meaningful similarity.
    """
    errors = []
    
    attrs = ['vocab_size', 'n_ctx', 'd_model', 'n_head', 'n_layers', 'attn_type']
    for attr in attrs:
        val_A = getattr(model_A, attr, None)
        val_B = getattr(model_B, attr, None)
        if val_A != val_B:
            errors.append(f"{attr}: {val_A} != {val_B}")
    
    if errors:
        raise ValueError(
            "Models have incompatible architectures:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )


# Convenience alias for backward compatibility
compute_tn_similarity_exact = compute_tn_similarity
