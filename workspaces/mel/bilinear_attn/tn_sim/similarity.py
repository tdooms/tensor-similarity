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

from src.components.similarity import similarity as _compute_similarity, State
from models.components.model import AttentionLMComponent, _validate_model_for_tn_similarity


def _as_component(model):
    if isinstance(model, AttentionLMComponent):
        return model
    _validate_model_for_tn_similarity(model)
    return AttentionLMComponent.from_trained_model(model)


def compute_tn_similarity(model_A, model_B, device: str = None) -> State:
    """Compute exact TN similarity between two AttentionLM models.

    Runs in the models' native dtype — no conversion. Upstream code should
    train and load in fp32 (the regime this path is designed for).

    Returns State(S_aa, S_bb, S_ab).
    """
    comp_A = _as_component(model_A)
    comp_B = _as_component(model_B)
    _validate_model_compatibility(comp_A, comp_B)

    if device is not None:
        comp_A = comp_A.to(device=device)
        comp_B = comp_B.to(device=device)

    return _compute_similarity(comp_A, comp_B)


def cosine_similarity(model_A, model_B, device: str = None) -> float:
    """Cosine similarity in [-1, 1]. Runs in the models' native dtype."""
    state = compute_tn_similarity(model_A, model_B, device=device)
    return _cosine_from_state(state)


def inner_product(model_A, model_B, device: str = None) -> float:
    """Inner product E[f_A(x)^T f_B(x)]. Runs in the models' native dtype."""
    state = compute_tn_similarity(model_A, model_B, device=device)
    return _inner_product_from_state(state)


def self_similarity(model, device: str = None) -> float:
    """Self cosine similarity (should be 1.0; useful for validation)."""
    return cosine_similarity(model, model, device=device)


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
