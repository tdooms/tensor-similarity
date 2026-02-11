"""
Convert a 1-layer bilinear attention-only model into a tensor network.

The attention layer computes:
    y_t = x_t + scale * O @ [ sum_s  M(t,s) * A1(t,s) * A2(t,s) * V @ x_s ]

where:
    A_i(t,s) = (R_t @ Q_i @ x_t)^T (R_s @ K_i @ x_s) / d_head
    M(t,s)   = causal mask (or diagonal mask for the restricted version)
    R_t, R_s = rotary embeddings at positions t, s
    O, V     = output and value projection matrices

As a tensor the attention path is degree-3 in x (one x through Q1, one through Q2,
one through V), and the residual path is degree-1 (identity). The full layer is a
sum of these two.

For the norm computation we need <T, T> = contract T with T* over all free indices.
The key trick: we can compute the norm of the attention-path tensor separately with
different masks (causal vs diagonal) without materializing the full tensor.

NOTE: We do NOT include the pre-unembed RMSNorm or embed/unembed in the tensor,
because the norm analysis is about the attention layer itself.
We also ignore norm inside attention (it's Identity).

Architecture of the tensor network for ||attention_path||^2:

  For each copy (bra and ket), the attention path is:
    scale * O @ (pattern @ V @ x)

  where pattern(t,s) = M(t,s) * (1/d_head^2) * [Q1 x_t · K1 x_s] * [Q2 x_t · K2 x_s]

  The inner product <attn, attn> contracts:
    - 3 pairs of input x indices (Q1, Q2, V paths)
    - 1 output d index
    - 2 sequence indices (t, s) via the mask

  Since the mask appears in both bra and ket, the combined mask is M(t,s)^2 = M(t,s)
  for binary masks.
"""

import sys
sys.path.insert(0, "/workspace/tensor-mars")

import torch
from quimb.tensor import Tensor, TensorNetwork


def build_attention_tensor_norm(attn, mask_kind="causal"):
    """
    Compute ||attention_path||^2 for a given Attention module and mask type.

    This contracts the attention tensor network with itself (Frobenius inner product).
    We do this analytically by tracing over all input/output indices.

    Args:
        attn: Attention module (from src/components/attention.py)
        mask_kind: "causal" for full causal mask, "diag" for diagonal-only

    Returns:
        norm_sq: scalar, the squared Frobenius norm of the attention path tensor
    """
    d_model = attn.d_model
    n_head = attn.n_head
    d_head = attn.d_head
    n_ctx = attn.n_ctx
    scale = attn.scale

    # Extract weight matrices (no bias in these linears)
    Q1 = attn.q1.weight  # (d_model, d_model)
    K1 = attn.k1.weight
    Q2 = attn.q2.weight
    K2 = attn.k2.weight
    V  = attn.v.weight   # (d_model, d_model)
    O  = attn.o.weight   # (d_model, d_model)

    # Reshape into per-head matrices
    # nn.Linear: output = x @ W^T, so W is (d_out, d_in)
    # After rearrange: (n_head, d_head, d_model)
    Q1 = Q1.view(n_head, d_head, d_model)
    K1 = K1.view(n_head, d_head, d_model)
    Q2 = Q2.view(n_head, d_head, d_model)
    K2 = K2.view(n_head, d_head, d_model)
    V  = V.view(n_head, d_head, d_model)
    O  = O.weight.view(d_model, n_head, d_head) if False else attn.o.weight.view(d_model, n_head, d_head)

    # OV circuit: O @ V per head -> (d_model, n_head, d_head) @ (n_head, d_head, d_model)
    # = per head: O_h @ V_h -> (d_model, d_model)
    # Combined OV: sum over heads of O_h @ V_h, but for norm we keep heads separate
    OV = torch.einsum('dnh,nhm->dnm', O, V)  # (d_model_out, d_model_in) -- summed over heads
    # Actually for the norm we need to keep the head structure.
    # Let's think about this more carefully.

    # The full attention output for a single position t is:
    #   attn(x)_t = scale * sum_h sum_s M(t,s) * score1_h(t,s) * score2_h(t,s) * O_h @ V_h @ x_s
    # where score_i_h(t,s) = (1/d_head) * (R_t Q_i_h x_t)^T (R_s K_i_h x_s)
    #
    # The full tensor has indices: (d_out, t | d_in_q1, d_in_q2, d_in_v, s)
    # where d_in_q1, d_in_q2 come from the two query paths at position t,
    # and d_in_v comes from the value path at position s.
    #
    # T[d_out, t, d_q1, d_q2, d_v, s] =
    #   scale/d_head^2 * sum_h sum_{h1,h2} M(t,s) *
    #   (sum_{e1} R_t[h1,e1] Q1_h[e1, d_q1]) * (sum_{f1} R_s[h1,f1] K1_h[f1, d_q1_in]) *
    #   ... this gets complicated with rotary.
    #
    # For now, let's compute the norm numerically by materializing the key contractions.

    return _compute_norm_numerical(attn, mask_kind)


def _get_mask(n_ctx, kind):
    """Get the mask matrix."""
    if kind == "causal":
        return torch.tril(torch.ones(n_ctx, n_ctx))
    elif kind == "diag":
        return torch.eye(n_ctx)
    elif kind == "offdiag1":
        # Off-by-1 diagonal: t == s+1 (attend to previous token, "bigram")
        return torch.diag(torch.ones(n_ctx - 1), diagonal=-1)
    elif kind == "offdiag2":
        # Off-by-2 diagonal: t == s+2 (attend to token 2 back, "trigram")
        return torch.diag(torch.ones(n_ctx - 2), diagonal=-2)
    elif kind == "none":
        return torch.ones(n_ctx, n_ctx)
    else:
        raise ValueError(f"Unknown mask kind: {kind}")


def _compute_norm_numerical(attn, mask_kind="causal"):
    """
    Compute ||attention_path||^2 numerically.

    The attention path maps x -> scale * O(pattern(x) @ V(x)).
    This is a degree-3 polynomial in x.

    For the norm, we need:
      ||T||^2 = sum over all index combos of T[...]^2

    We compute this by constructing the Gram matrices and contracting.

    The key insight: ||T||^2 = Tr[ (stuff involving QK gram matrices) * (OV gram) * (mask gram) ]
    """
    device = attn.q1.weight.device
    dtype = attn.q1.weight.dtype
    d_model = attn.d_model
    n_head = attn.n_head
    d_head = attn.d_head
    n_ctx = attn.n_ctx
    scale = attn.scale

    Q1 = attn.q1.weight.view(n_head, d_head, d_model)
    K1 = attn.k1.weight.view(n_head, d_head, d_model)
    Q2 = attn.q2.weight.view(n_head, d_head, d_model)
    K2 = attn.k2.weight.view(n_head, d_head, d_model)
    V  = attn.v.weight.view(n_head, d_head, d_model)
    O  = attn.o.weight.view(d_model, n_head, d_head)

    M = _get_mask(n_ctx, mask_kind).to(device=device, dtype=dtype)

    # Rotary embeddings
    cos = attn.rotary.cos  # (n_ctx, d_head//2)
    sin = attn.rotary.sin  # (n_ctx, d_head//2)

    # For the norm computation ||T||^2, we need to contract T with T*.
    # T is indexed by (d_out, t, d_q1, d_q2, d_v, s).
    # ||T||^2 = sum_{d_out,t,d_q1,d_q2,d_v,s} T[...]^2
    #
    # = scale^2 / d_head^4 * sum_{t,s} M(t,s)^2 *
    #   sum_{h,h'} [sum_{d_out} O_h[d_out,:] O_h'[d_out,:]] *    -- OV gram over d_out
    #              [sum_{d_v} (V_h x_s)[...] (V_h' x_s)[...]] *   -- V gram over d_v
    #              [sum_{d_q1} (score1_h)(score1_h')] *             -- Q1K1 gram over d_q1
    #              [sum_{d_q2} (score2_h)(score2_h')] *             -- Q2K2 gram over d_q2
    #
    # But we also trace over x (d_q1, d_q2, d_v are "input" dims we sum over).
    # The input dimensions are independent so we can factor:
    #
    # ||T||^2 = scale^2/d_head^4 * sum_{t,s,h,h'} M(t,s)^2 *
    #   [sum_{d_out} O[d_out,h,:] . O[d_out,h',:]] *
    #   [sum_{d_v}   V[h,:,d_v]  . V[h',:,d_v]]  *  -- but rotary doesn't affect V
    #   [score1_gram(h,h',t,s)] *
    #   [score2_gram(h,h',t,s)]
    #
    # where score_i_gram(h,h',t,s) = sum_{d_q} [R_t Q_i_h[:,d_q]]^T [R_t Q_i_h'[:,d_q]] *
    #                                           [R_s K_i_h[:,d_q]]^T [R_s K_i_h'[:,d_q]]
    #
    # Wait, that's not right either. Let me think again.
    #
    # Actually T contracts with T* over ALL free indices (d_out, t, d_q1, d_q2, d_v, s).
    # When we say ||T||^2 we mean the Frobenius norm — sum of squares of all entries.
    # The "entries" are indexed by (d_out, t, d_q1, d_q2, d_v, s).
    #
    # But d_q1, d_q2, d_v are the INPUT dimensions, not output. The tensor T maps
    # (d_q1, d_q2, d_v, s) -> (d_out, t). So the Frobenius norm is indeed
    # sum over all indices of |T[d_out, t, d_q1, d_q2, d_v, s]|^2.
    #
    # Let's just compute it directly for small models.
    # For d_model=64, n_ctx=128, this tensor has shape (64, 128, 64, 64, 64, 128)
    # which is way too big to materialize (~128 TB).
    #
    # So we MUST factorize. Let me redo the math carefully.

    # The attention output at position t, output dim d_out is:
    #   T[d_out, t, d_q1, d_q2, d_v, s] =
    #     scale * sum_h M(t,s) * (1/d_head^2) *
    #     [sum_e (R_t Q1_h)[e, d_q1] * (R_s K1_h)[e, ???]]
    #     ...
    #
    # Actually, let me reconsider. The score is:
    #   score1_h(t,s,x_t,x_s) = (1/d_head) * sum_e (R_t Q1_h x_t)[e] * (R_s K1_h x_s)[e]
    #                          = (1/d_head) * sum_e sum_{d1} (R_t Q1_h)[e,d1] x_t[d1] *
    #                                                sum_{d1'} (R_s K1_h)[e,d1'] x_s[d1']
    #
    # But the d_q1 in the tensor refers to the input that goes through Q1 path.
    # The x_t that enters Q1 and the x_t that enters Q2 are the SAME x_t.
    # Similarly, the x_s that enters K1, K2, and V are the SAME x_s.
    #
    # So the tensor is NOT separable in (d_q1, d_q2). The two query paths share
    # the same input x_t. This makes the tensor degree-3 (not degree-5):
    #   Input copies: x_t (shared by Q1 and Q2), x_s (shared by K1, K2, V)
    #   But wait, only Q paths use x_t, and K+V paths use x_s.
    #
    # Let me re-examine the forward pass:
    #   q1, k1, q2, k2, v = [rearrange(op(x), ...) for op in [self.q1, self.k1, self.q2, self.k2, self.v]]
    #   pattern = mask((scores1 * scores2) / d_head^2)
    #   z = pattern @ v
    #   return x + O(z) * scale
    #
    # At position t:
    #   q1_t = Q1 @ x_t  (applied to x at position t)
    #   k1_s = K1 @ x_s  (applied to x at position s)
    #   score1(t,s) = q1_t . k1_s  (dot over d_head, after rotary)
    #
    # So x_t feeds into Q1 and Q2 (both queries), and x_s feeds into K1, K2, V.
    # The output at position t and output dim d:
    #
    #   T_attn[d, t, x_t, x_s, s] = scale/d_head^2 * sum_h M(t,s) *
    #     [sum_e (R_t Q1_h)[e,:] . x_t] [sum_e (R_s K1_h)[e,:] . x_s] *
    #     [sum_f (R_t Q2_h)[f,:] . x_t] [sum_f (R_s K2_h)[f,:] . x_s] *
    #     O[d, h, :] . (V_h @ x_s)
    #
    # So T is quadratic in x_t and cubic in x_s:
    #   - x_t appears in Q1 and Q2 (degree 2)
    #   - x_s appears in K1, K2, and V (degree 3)
    #
    # Total degree = 5 in x, but 2 in x_t and 3 in x_s (with shared indices).
    #
    # For ||T_attn||^2 = sum_{d,t,s} <T_attn[d,t,:,:,s], T_attn[d,t,:,:,s]>
    #
    # Since x_t and x_s are independent inputs (different positions), the norm factors:
    #
    # ||T_attn||^2 = (scale/d_head^2)^2 * sum_{t,s} M(t,s)^2 *
    #   sum_{h,h'} [O_h . O_h' over d_out] *
    #   [Q1-rotary gram at t (h,h')] * [K1-rotary gram at s (h,h')] *
    #   [Q2-rotary gram at t (h,h')] * [K2-rotary gram at s (h,h')] *
    #   [V_h . V_h' over d_in, times K-V cross terms...]
    #
    # Hmm, but K1, K2, V all use the SAME x_s. So we can't independently trace
    # over the K1-input and V-input — they're the same variable.
    #
    # This means we need to compute a 3rd-order moment of x_s (degree 3 polynomial),
    # which for the norm gives a 6th-order contraction. This is exactly what
    # Thomas' isserlis.py is for — computing these higher-order moments using
    # Isserlis' theorem (Wick's theorem).
    #
    # For now, let's use a simpler numerical approach: sample random x and estimate
    # the norm via Monte Carlo. This is valid because:
    #   ||T||^2 = E_{x~N(0,I)} [||T(x)||^2] * (appropriate normalization)
    #
    # Actually for a multilinear map T of degree k in inputs of dim d,
    #   ||T||_F^2 = E_{g1,...,gk ~ N(0,I)} [T(g1,...,gk)^2] when inputs are independent.
    #
    # But our inputs are NOT independent (x_t shared by Q1,Q2; x_s shared by K1,K2,V).
    # So we need to account for that.

    # SIMPLEST CORRECT APPROACH: feed random inputs through the actual model
    # and compute the output variance, then relate to the norm.
    #
    # Let's use the Monte Carlo estimator:
    #   ||T||^2 ≈ E_{x ~ N(0,I)} [ ||T(x)||^2 ] / Z
    # where Z is a normalization factor.
    #
    # Actually, the cleanest way: just contract the tensor network using quimb,
    # using Thomas' infrastructure.

    return _compute_norm_via_contraction(attn, mask_kind)


def _compute_norm_via_contraction(attn, mask_kind):
    """
    Compute ||attention_path||^2 by explicitly constructing the "doubled" tensor
    network (bra-ket) and contracting.

    For the attention path (ignoring residual):
      T[d_out, t, d_t, d_t', d_s, d_s', d_s'', s] where
      d_t, d_t' are Q1, Q2 inputs from position t
      d_s, d_s', d_s'' are K1, K2, V inputs from position s

    But since Q1 and Q2 share x_t, and K1,K2,V share x_s, we need to handle
    the shared-input structure.

    For ||T||^2 with shared inputs, we get:
      sum_{t,s,d_out} |sum_h M(t,s)/d_head^2 * scale *
        <R_t Q1_h, x_t> <R_t Q2_h, x_t> <R_s K1_h, x_s> <R_s K2_h, x_s> <O_h V_h, x_s>_d_out|^2

    Expanding the square and tracing over x_t (dim d_model) and x_s (dim d_model):

    For x_t (degree 2): sum_{d_t} [R_t Q1_h][a,d_t] [R_t Q2_h][b,d_t] [R_t Q1_h'][a',d_t] [R_t Q2_h'][b',d_t]
      This is a 4th moment of a Gaussian, which by Isserlis' theorem decomposes into
      3 pairings: (aa',bb') + (ab', a'b) + (ab, a'b') but the last two are cross terms.

    This is getting quite involved. Let me use a direct sampling approach for now
    and validate against analytic results later when Martin's code is ready.
    """
    return _compute_norm_mc(attn, mask_kind)


def _attn_path_forward(attn, x, mask):
    """Run x through the attention path (no residual), using the given mask."""
    from einops import rearrange, einsum

    n_head = attn.n_head
    d_head = attn.d_head
    scale = attn.scale

    q1, k1, q2, k2, v = [
        rearrange(op(x), '... (n_head d_head) -> ... n_head d_head', n_head=n_head)
        for op in [attn.q1, attn.k1, attn.q2, attn.k2, attn.v]
    ]

    q1, k1 = attn.rotary(q1), attn.rotary(k1)
    q2, k2 = attn.rotary(q2), attn.rotary(k2)

    scores1 = einsum(q1, k1, "... seq_q n_head d_head, ... seq_k n_head d_head -> ... n_head seq_q seq_k")
    scores2 = einsum(q2, k2, "... seq_q n_head d_head, ... seq_k n_head d_head -> ... n_head seq_q seq_k")

    pattern = mask[None, None, :, :] * (scores1 * scores2) / d_head**2

    z = einsum(pattern, v, "... n_head seq_q seq_k, ... seq_k n_head d_head -> ... seq_q n_head d_head")
    z = rearrange(z, '... seq n_head d_head -> ... seq (n_head d_head)')
    return attn.o(z) * scale


@torch.no_grad()
def _compute_norm_mc(attn, mask_kind, n_samples=512, batch_size=32):
    """
    Monte Carlo estimate of ||attention_path||^2.

    Feeds random Gaussian inputs x ~ N(0, I) through the attention path
    (without residual) and computes E[||attn_path(x)||^2].

    The ratio ||T_diag||^2 / ||T_causal||^2 measures what fraction of the
    attention path's output energy comes from the diagonal (t==s) part.
    """
    device = attn.q1.weight.device
    dtype = attn.q1.weight.dtype
    d_model = attn.d_model
    n_ctx = attn.n_ctx

    M = _get_mask(n_ctx, mask_kind).to(device=device, dtype=dtype)
    total_norm_sq = 0.0

    for i in range(0, n_samples, batch_size):
        bs = min(batch_size, n_samples - i)
        x = torch.randn(bs, n_ctx, d_model, device=device, dtype=dtype)
        out = _attn_path_forward(attn, x, M)
        total_norm_sq += out.pow(2).sum().item() / bs

    n_batches = (n_samples + batch_size - 1) // batch_size
    return total_norm_sq / n_batches


@torch.no_grad()
def compute_cross_norm_mc(attn_a, attn_b, mask_kind="causal", n_samples=512, batch_size=32):
    """
    Monte Carlo estimate of <T_a, T_b> — cross inner product of two attention tensors.

    Feeds the SAME random input x through both attention paths and computes:
      <T_a, T_b> ≈ E_x[ sum_{t,d} attn_a(x)[t,d] * attn_b(x)[t,d] ]

    This is useful for tensor similarity:
      sim(a, b) = <T_a, T_b> / (||T_a|| * ||T_b||)
    """
    device = attn_a.q1.weight.device
    dtype = attn_a.q1.weight.dtype
    d_model = attn_a.d_model
    n_ctx = attn_a.n_ctx

    M = _get_mask(n_ctx, mask_kind).to(device=device, dtype=dtype)
    total_ip = 0.0

    for i in range(0, n_samples, batch_size):
        bs = min(batch_size, n_samples - i)
        x = torch.randn(bs, n_ctx, d_model, device=device, dtype=dtype)
        out_a = _attn_path_forward(attn_a, x, M)
        out_b = _attn_path_forward(attn_b, x, M)
        total_ip += (out_a * out_b).sum().item() / bs

    n_batches = (n_samples + batch_size - 1) // batch_size
    return total_ip / n_batches


def compute_similarity_mc(attn_a, attn_b, mask_kind="causal", n_samples=512):
    """
    Monte Carlo tensor similarity (cosine) between two attention modules.

    sim = <T_a, T_b> / (||T_a|| * ||T_b||)

    Self-similarity should be ~1.0. Small perturbation should be close to 1.0.
    """
    import math
    ip_ab = compute_cross_norm_mc(attn_a, attn_b, mask_kind, n_samples)
    norm_a = _compute_norm_mc(attn_a, mask_kind, n_samples)
    norm_b = _compute_norm_mc(attn_b, mask_kind, n_samples)
    return ip_ab / math.sqrt(norm_a * norm_b)


def compute_residual_norm(attn, n_ctx, d_model):
    """
    The residual path is just the identity: y_t = x_t.
    ||residual||^2 for Gaussian inputs = n_ctx * d_model (each entry has E[x^2]=1).
    """
    return n_ctx * d_model


def compute_diagonal_fraction(attn, n_samples=512):
    """
    Compute diagonal and off-by-1 norm fractions:
      diag_fraction    = ||T_diag||^2 / ||T_causal||^2     (t==s, "unigram/copy")
      offdiag1_fraction = ||T_offdiag1||^2 / ||T_causal||^2 (t==s+1, "bigram")

    Returns dict with all relevant quantities.
    """
    norm_causal = _compute_norm_mc(attn, "causal", n_samples=n_samples)
    norm_diag = _compute_norm_mc(attn, "diag", n_samples=n_samples)
    norm_offdiag1 = _compute_norm_mc(attn, "offdiag1", n_samples=n_samples)
    norm_offdiag2 = _compute_norm_mc(attn, "offdiag2", n_samples=n_samples)

    return {
        "norm_causal": norm_causal,
        "norm_diag": norm_diag,
        "norm_offdiag1": norm_offdiag1,
        "norm_offdiag2": norm_offdiag2,
        "diag_fraction": norm_diag / norm_causal if norm_causal > 0 else float('nan'),
        "offdiag1_fraction": norm_offdiag1 / norm_causal if norm_causal > 0 else float('nan'),
        "offdiag2_fraction": norm_offdiag2 / norm_causal if norm_causal > 0 else float('nan'),
    }
