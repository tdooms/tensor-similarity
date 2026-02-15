"""Quimb-based tensor network similarity for bilinear attention models.

This is a parallel implementation of tn_similarity.py that constructs explicit
quimb TensorNetworks and uses quimb's contraction engine. It follows the same
conventions as src/components/ but adapted for the bilinear_attn workspace's
AttentionLM model.

Key conventions (matching src/components/base.py):
- Input indices: 'in:d0', 'in:d1', ... (first dim is bias/constant)
- Output index: 'out:d'
- Sequence indices: 'out:s', 'in:s'
- Head index: 'n'
- Hidden indices: 'h:*'

The attention-only TN (per layer) has 5 input legs:
  in:d0 = V(x_s), in:d1 = K1(x_s), in:d2 = K2(x_s), in:d3 = Q1(x_t), in:d4 = Q2(x_t)
  out:s = query position, in:s = key position
"""

import math
import torch
import numpy as np
from quimb.tensor import Tensor, TensorNetwork


# ── Helpers ──────────────────────────────────────────────────────────────────

def pad(weight, bias=None, scale=1.0, constant=False):
    """Absorb bias into weight as a constant input dimension.
    
    If constant=True: returns block_diag(1, scale*W) with bias in first col.
    If constant=False: returns [scale*bias | scale*W] (bias as first column).
    """
    if constant:
        w = torch.block_diag(torch.ones_like(weight[:1, :1]), scale * weight)
        if bias is not None:
            w[1:, 0] = scale * bias
        return w
    b = scale * bias[:, None] if bias is not None else torch.zeros_like(weight[:, :1])
    return torch.cat([b, scale * weight], dim=-1)


# ── Build attention-only TN for one layer ────────────────────────────────────

def build_attention_network(layer, tag_prefix="", device=None, dtype=None):
    """Build a quimb TensorNetwork for one bilinear attention layer (no residual).
    
    Follows the structure in src/components/attention.py but adapted for the
    bilinear_attn workspace's BilinearAttention layer.
    
    Input legs: in:d0 (V), in:d1 (K1), in:d2 (K2), in:d3 (Q1), in:d4 (Q2)
    Output leg: out:d
    Sequence legs: out:s (query pos), in:s (key pos)
    
    Returns: TensorNetwork
    """
    d_model = layer.d_model
    n_head = layer.n_head
    d_head = layer.d_head
    n_ctx = layer.n_ctx
    scale = layer.scale
    
    if device is None:
        device = layer.o.weight.device
    if dtype is None:
        dtype = layer.o.weight.dtype
    like = dict(device=device, dtype=dtype)
    
    p = tag_prefix
    
    # ── O: (d_model, n_head, d_head) ──
    o_data = scale * layer.o.weight.detach().to(**like).view(d_model, n_head, d_head)
    o = Tensor(o_data, inds=['out:d', f'{p}n', f'{p}ov:h'], tags=['O'])
    
    # ── V: (n_head, d_head, d_model+1) with bias absorbed ──
    v_weight = pad(layer.v.weight.detach().to(**like), 
                   layer.v.bias.detach().to(**like) if layer.v.bias is not None else None,
                   constant=False)
    v_data = v_weight.view(n_head, d_head, d_model + 1)
    v = Tensor(v_data, inds=[f'{p}n', f'{p}ov:h', 'in:d0'], tags=['V'])
    
    # ── QK projections: reshape to (n_head, 2, d_head//2, d_model+1) for RoPE ──
    def _qk_tensor(proj, input_idx, tag, mod):
        w = pad(proj.weight.detach().to(**like),
                proj.bias.detach().to(**like) if proj.bias is not None else None,
                constant=False)
        w = w.view(n_head, 2, d_head // 2, d_model + 1)
        return Tensor(w, inds=[f'{p}n', f'{p}{mod}:2', f'{p}{mod}:h', f'in:d{input_idx}'], tags=[tag])
    
    q1 = _qk_tensor(layer.q1, 3, 'Q1', 'left')
    k1 = _qk_tensor(layer.k1, 1, 'K1', 'left')
    q2 = _qk_tensor(layer.q2, 4, 'Q2', 'right')
    k2 = _qk_tensor(layer.k2, 2, 'K2', 'right')
    
    # ── RoPE: rotation tensors ──
    # The "black box" tensor encodes the 2x2 rotation structure
    # Shape: (iq, ik, 2q, 2k) = (2, 2, 2, 2) per the convention in src/components/attention.py
    rope_data = [[[[1, 0], [0, 1]], [[0, -1], [1, 0]]], 
                 [[[0, 1], [-1, 0]], [[1, 0], [0, 1]]]]
    
    def _rope_network(mod):
        black = Tensor(torch.tensor(rope_data, **like),
                       inds=[f'{p}{mod}:iq', f'{p}{mod}:ik', f'{p}{mod}:2q', f'{p}{mod}:2k'],
                       tags=['#'])
        
        # RoPE embedding: cos/sin computed from frequencies
        freq = 1.0 / (10000 ** (torch.arange(0, d_head, 2, **like) / d_head))
        ctx = torch.arange(n_ctx, **like)
        freqs = torch.outer(ctx, freq)
        emb = torch.stack([freqs.cos(), freqs.sin()], dim=-1)  # (n_ctx, d_head//2, 2)
        
        q_emb = Tensor(emb, inds=['out:s', f'{p}{mod}:h', f'{p}{mod}:iq'], tags=['E'])
        k_emb = Tensor(emb, inds=['in:s', f'{p}{mod}:h', f'{p}{mod}:ik'], tags=['E'])
        
        return black & q_emb & k_emb
    
    left_rope = _rope_network('left')
    right_rope = _rope_network('right')
    
    # ── Scaling: 1/d_head per circuit ──
    s_left = Tensor(torch.full((d_head // 2,), 1.0 / d_head, **like),
                    inds=[f'{p}left:h'], tags=['S'])
    s_right = Tensor(torch.full((d_head // 2,), 1.0 / d_head, **like),
                     inds=[f'{p}right:h'], tags=['S'])
    
    # ── Causal mask ──
    mask_data = torch.tril(torch.ones(n_ctx, n_ctx, **like))
    mask = Tensor(mask_data, inds=['out:s', 'in:s'], tags=['M'])
    
    # ── Assemble ──
    # Connect QK circuits with RoPE
    left_qk = left_rope & q1 & k1 & s_left
    right_qk = right_rope & q2 & k2 & s_right
    
    return TensorNetwork([o, v, mask, left_qk, right_qk], check_collisions=False)


# ── Bra-ket contraction (inner product) ─────────────────────────────────────

def _make_bra(tn):
    """Create the bra (conjugate) copy by starring all input and output indices."""
    # Star all 'in:' and 'out:' indices
    reindex_map = {}
    for idx in tn.ind_map:
        if idx.startswith('in:') or idx.startswith('out:'):
            reindex_map[idx] = idx + '*'
    return tn.reindex(reindex_map)


def contract_bra_ket(bra_tn, ket_tn, gram=None, output_inds=None):
    """Contract bra and ket TNs with optional input gram matrices.
    
    If gram is provided, it's inserted between each pair of matching input indices.
    If output_inds is specified, those indices are kept free (returning a matrix).
    Otherwise, all indices are contracted (returning a scalar).
    
    Args:
        bra_tn: TensorNetwork (will be starred)
        ket_tn: TensorNetwork (kept as-is)
        gram: optional (d_in, d_in) matrix to insert between input pairs
        output_inds: list of index pairs to keep free, e.g. [('out:d', 'out:d*')]
    
    Returns: contracted result (scalar or tensor)
    """
    bra = _make_bra(bra_tn)
    
    tensors = [bra, ket_tn]
    
    if gram is not None:
        # Find all input indices in the ket
        input_inds = sorted(set(idx for idx in ket_tn.ind_map if idx.startswith('in:d')))
        for idx in input_inds:
            starred = idx + '*'
            if starred in bra.ind_map:
                g = Tensor(gram, inds=[starred, idx], tags=['G'])
                tensors.append(g)
    
    # For sequence indices: they should match between bra and ket
    # out:s* in bra matches out:s in ket (query positions sum)
    # in:s* in bra matches in:s in ket (key positions sum)
    # These are already handled by the mask tensor sharing
    
    full_tn = TensorNetwork(tensors, check_collisions=False)
    
    if output_inds:
        out_idx_list = []
        for pair in output_inds:
            out_idx_list.extend(pair)
        result = full_tn.contract(all, output_inds=out_idx_list)
        return result.data
    else:
        return full_tn.contract(all)


# ── Symmetrized attention inner product ──────────────────────────────────────

def build_symmetrized_attention_pair(layer_A, layer_B, gram, device=None, dtype=None):
    """Build the symmetrized bra-ket TN for two attention layers.
    
    Returns the symmetrized output gram G_attn[d_out_A, d_out_B] as a matrix.
    
    Symmetrization: average the direct matching (circuit1↔circuit1, circuit2↔circuit2)
    with the cross matching (circuit1↔circuit2, circuit2↔circuit1).
    
    For the cross matching, we swap Q1↔Q2 and K1↔K2 in the ket (model B).
    """
    if device is None:
        device = layer_A.o.weight.device
    if dtype is None:
        dtype = torch.float64
    like = dict(device=device, dtype=dtype)
    
    # Build bra (model A)
    tn_A = build_attention_network(layer_A, device=device, dtype=dtype)
    
    # Build ket (model B) — direct matching
    tn_B_direct = build_attention_network(layer_B, device=device, dtype=dtype)
    
    # Build ket (model B) — cross matching (swap circuits)
    # We swap the input indices: in:d1↔in:d2 (K1↔K2) and in:d3↔in:d4 (Q1↔Q2)
    swap_map = {'in:d1': 'in:d2', 'in:d2': 'in:d1', 'in:d3': 'in:d4', 'in:d4': 'in:d3'}
    tn_B_cross = build_attention_network(layer_B, device=device, dtype=dtype).reindex(swap_map)
    
    # Contract direct matching
    G_direct = contract_bra_ket(
        tn_A, tn_B_direct, gram=gram,
        output_inds=[('out:d*', 'out:d')]
    )
    
    # Contract cross matching
    G_cross = contract_bra_ket(
        tn_A, tn_B_cross, gram=gram,
        output_inds=[('out:d*', 'out:d')]
    )
    
    # Symmetrized average
    return 0.5 * (G_direct + G_cross)


# ── Full model similarity ───────────────────────────────────────────────────

def compute_initial_gram(embed_A, embed_B, device=None, dtype=torch.float64):
    """Compute initial gram matrix from embedding weights.
    Same as tn_similarity.py version.
    """
    E_A = embed_A.weight.detach().to(device=device, dtype=dtype)
    E_B = embed_B.weight.detach().to(device=device, dtype=dtype)
    V, d = E_A.shape
    
    G = torch.zeros(d + 1, d + 1, device=device, dtype=dtype)
    G[0, 0] = 1.0
    mu_A = E_A.mean(dim=0)
    mu_B = E_B.mean(dim=0)
    G[0, 1:] = mu_B
    G[1:, 0] = mu_A
    G[1:, 1:] = E_A.T @ E_B / V
    return G


def layer_inner_product_with_residual_quimb(layer_A, layer_B, G_in, device=None, dtype=torch.float64):
    """Compute output gram for one layer WITH residual using quimb contraction.
    
    y = x + scale * attn(x)
    G_out = G_residual + G_attn  (cross-terms ignored under independence)
    """
    d = layer_A.d_model
    
    # Attention path gram via quimb
    G_attn = build_symmetrized_attention_pair(layer_A, layer_B, G_in, device, dtype)
    
    # Build output gram
    G_out = torch.zeros(d + 1, d + 1, device=device, dtype=dtype)
    G_out[0, 0] = 1.0
    G_out[1:, 1:] = G_in[1:, 1:]  # residual
    G_out[1:, 1:] += G_attn        # attention
    G_out[0, 1:] = G_in[0, 1:]
    G_out[1:, 0] = G_in[1:, 0]
    
    return G_out


def compute_tn_similarity_quimb(model_A, model_B, device=None, dtype=torch.float64):
    """Compute tensor network cosine similarity using quimb contraction.
    
    This is functionally equivalent to compute_tn_similarity() but uses
    quimb TensorNetworks for the attention path contraction.
    """
    if device is None:
        device = next(model_A.parameters()).device
    
    n_layers = model_A.n_layers
    
    G_AB = compute_initial_gram(model_A.embed, model_B.embed, device, dtype)
    G_AA = compute_initial_gram(model_A.embed, model_A.embed, device, dtype)
    G_BB = compute_initial_gram(model_B.embed, model_B.embed, device, dtype)
    
    for i in range(n_layers):
        G_AB = layer_inner_product_with_residual_quimb(
            model_A.layers[i], model_B.layers[i], G_AB, device, dtype)
        G_AA = layer_inner_product_with_residual_quimb(
            model_A.layers[i], model_A.layers[i], G_AA, device, dtype)
        G_BB = layer_inner_product_with_residual_quimb(
            model_B.layers[i], model_B.layers[i], G_BB, device, dtype)
    
    U_A = model_A.unembed.weight.detach().to(device=device, dtype=dtype)
    U_B = model_B.unembed.weight.detach().to(device=device, dtype=dtype)
    
    ip_AB = (U_A @ G_AB[1:, 1:] @ U_B.T).trace().item()
    ip_AA = (U_A @ G_AA[1:, 1:] @ U_A.T).trace().item()
    ip_BB = (U_B @ G_BB[1:, 1:] @ U_B.T).trace().item()
    
    denom = np.sqrt(abs(ip_AA) * abs(ip_BB))
    if denom < 1e-30:
        return 0.0
    return ip_AB / denom
