"""
Tensor similarity computation for tracking training dynamics.
"""
import torch
from einops import einsum

def get_interaction_matrix(model,symmetrize=True):
    """Extract the full interaction matrix B[cls, in1, in2]."""
    l, r = model.w_lr[0].unbind()
    B = einsum(model.w_u, l, r, "c o, o i, o j -> c i j")
    if symmetrize:
        B = 0.5 * (B + B.transpose(1, 2))
    return B

def get_interaction_tensor_with_embedding(model,symmetrize=True):
    """Extract the full interaction tensor T[o, i, j] including embedding."""
    e = model.embed.weight  # [d_hidden, d_input]
    l, r = model.w_lr[0].unbind()  # [d_hidden, d_hidden]
    u = model.w_u  # [d_output, d_hidden]

    # Contract E with L and R, keeping hidden dimension
    # el[i, h] = sum over h1: E[h1, i] * L[h1, h]
    el = einsum(e, l, "h1 i, h1 h -> i h")  # [d_input, d_hidden]
    er = einsum(e, r, "h1 i, h1 h -> i h")  # [d_input, d_hidden]

    # Compute full interaction tensor T[o, i, j]
    # Contract over hidden dimension
    T = einsum(u, el, er, "o h, i h, j h -> o i j")
    if symmetrize:
        T = 0.5 * (T + T.transpose(1, 2))  # Symmetrize
    
    return T

def int_tensor_similarity_symmetric_proper(model1, model2):
    """
    Compute symmetric bilinear similarity including embedding.
    
    This extends the symmetric_inner formula to include embeddings:
    - el1 = E1 @ L1, er1 = E1 @ R1 (and same for model2)
    - Core = 0.5 × ((el1·el2 ⊙ er1·er2) + (el1·er2 ⊙ er1·el2))
    - Inner product = Tr(core @ u1·u2.T)
    
    This handles the L↔R swap symmetry at the weight level.
    """
    e1 = model1.embed.weight  # [d_hidden, d_input]
    e2 = model2.embed.weight
    
    l1, r1 = model1.w_lr[0].unbind()  # [d_hidden, d_hidden]
    l2, r2 = model2.w_lr[0].unbind()
    
    u1 = model1.w_u  # [d_output, d_hidden]
    u2 = model2.w_u
    
    # Contract embeddings with L and R
    el1 = einsum(e1, l1, "h1 i, h1 h -> i h")  # [d_input, d_hidden]
    er1 = einsum(e1, r1, "h1 i, h1 h -> i h")
    
    el2 = einsum(e2, l2, "h1 i, h1 h -> i h")
    er2 = einsum(e2, r2, "h1 i, h1 h -> i h")
    
    # Inner products between transformed embeddings
    # Shape: [hid1, hid2] where hid are the hidden dimensions after contraction
    el_el = einsum(el1, el2, "i h1, i h2 -> h1 h2")
    er_er = einsum(er1, er2, "i h1, i h2 -> h1 h2")
    el_er = einsum(el1, er2, "i h1, i h2 -> h1 h2")
    er_el = einsum(er1, el2, "i h1, i h2 -> h1 h2")
    
    # Symmetric core: average aligned and swapped terms
    core = 0.5 * ((el_el * er_er) + (el_er * er_el))
    
    # Contract with unembedding matrices
    uu = einsum(u1, u2, "o h1, o h2 -> h1 h2")
    hid = einsum(core, uu, "h1 h2, h1 h3 -> h2 h3")
    inner_product = torch.trace(hid)
    
    # Compute norms (self-inner products)
    norm1 = torch.sqrt(compute_symmetric_norm_with_embedding(model1))
    norm2 = torch.sqrt(compute_symmetric_norm_with_embedding(model2))
    
    return (inner_product / (norm1 * norm2)).item()


def compute_symmetric_norm_with_embedding(model):
    """Compute the norm (sqrt of self-inner product) with embedding."""
    e = model.embed.weight
    l, r = model.w_lr[0].unbind()
    u = model.w_u
    
    # Contract embeddings
    el = einsum(e, l, "h1 i, h1 h -> i h")
    er = einsum(e, r, "h1 i, h1 h -> i h")
    
    # Self inner products
    el_el = einsum(el, el, "i h1, i h2 -> h1 h2")
    er_er = einsum(er, er, "i h1, i h2 -> h1 h2")
    el_er = einsum(el, er, "i h1, i h2 -> h1 h2")
    
    # For self-similarity: el_er.T = er_el
    core = 0.5 * ((el_el * er_er) + (el_er * el_er.T))
    
    uu = einsum(u, u, "o h1, o h2 -> h1 h2")
    hid = einsum(core, uu, "h1 h2, h1 h3 -> h2 h3")
    
    return torch.trace(hid)


def int_tensor_similarity_pg(model1, model2, symmetric=False):
    """
    Compute tensor similarity between two models.
    
    Args:
        model1, model2: Models to compare
        symmetric: If True, use symmetric bilinear similarity (handles L↔R swap)
                   If False, use standard tensor cosine similarity
    """
    if symmetric:
        return int_tensor_similarity_symmetric_proper(model1, model2)
    else:
        # Standard: form tensors then compare
        T1 = get_interaction_tensor_with_embedding(model1)
        T2 = get_interaction_tensor_with_embedding(model2)
        
        dot_product = einsum(T1, T2, "o i j, o i j ->")
        norm1 = torch.sqrt(einsum(T1, T1, "o i j, o i j ->"))
        norm2 = torch.sqrt(einsum(T2, T2, "o i j, o i j ->"))
        
        return (dot_product / (norm1 * norm2)).item()