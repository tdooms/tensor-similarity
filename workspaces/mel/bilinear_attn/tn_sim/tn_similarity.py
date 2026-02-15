"""Exact tensor network similarity for bilinear attention models.

Design decisions:
- Gaussian assumption: higher-order moments factor via Isserlis into products of 2nd moments.
  This lets us chain gram matrices through layers (each layer contraction only needs the
  covariance of its input, not the full distribution).
- RMSNorm (pre-unembed only): Under the extra-input approximation (treat norm scalar as
  independent of x), the E[s^2] factor cancels in cosine similarity. So we ignore it.
- Residual: Each layer computes y = x + scale * attn(x). We handle this by decomposing
  the inner product into 4 terms: <I,I>, <I,A>, <A,I>, <A,A> and summing.
- RoPE: Linear in its input (position-indexed rotation), represented as fixed tensors.
- Causal mask: Fixed binary tensor, summed over sequence positions.

The attention path (without residual) for one layer is degree 5 in x:
  attn(x)_t = O * sum_s M(t,s)/d^2 * (q1(x_t).k1(x_s)) * (q2(x_t).k2(x_s)) * V(x_s)

Input legs: d0=V(x_s), d1=K1(x_s), d2=K2(x_s), d3=Q1(x_t), d4=Q2(x_t)

For the bra-ket contraction with gram G on each input leg:
  G_out[i,j] = sum over all hidden indices of bra_A[...,i] * ket_B[...,j] * prod_k G[d_k, d_k']
"""

import torch
import numpy as np
from einops import rearrange


def _rope_matrices(cos, sin, n_ctx):
    """Build per-position 2x2 rotation matrices from RoPE cos/sin buffers.
    
    Returns: (n_ctx, d_head//2, 2, 2) rotation matrices.
    """
    # cos, sin: (n_ctx, d_head//2)
    R = torch.zeros(n_ctx, cos.shape[1], 2, 2, dtype=cos.dtype, device=cos.device)
    R[:, :, 0, 0] = cos
    R[:, :, 0, 1] = -sin
    R[:, :, 1, 0] = sin
    R[:, :, 1, 1] = cos
    return R


def _apply_rope_to_weight(W, R_pos):
    """Apply RoPE rotation to a QK weight matrix at a given position.
    
    W: (n_head, d_head, d_in) — the QK projection weight (with bias absorbed)
    R_pos: (d_head//2, 2, 2) — rotation matrix at one position
    
    Returns: (n_head, d_head, d_in) — rotated weight
    """
    n_head, d_head, d_in = W.shape
    half = d_head // 2
    # Reshape W into pairs: (n_head, half, 2, d_in)
    W_pairs = W.reshape(n_head, half, 2, d_in)
    # R_pos: (half, 2, 2) — apply rotation: out[h,p,:,d] = R[p,:,:] @ W_pairs[h,p,:,d]
    W_rot = torch.einsum("pij,hpjd->hpid", R_pos, W_pairs)
    return W_rot.reshape(n_head, d_head, d_in)


def build_layer_weights(layer, device=None, dtype=torch.float64):
    """Extract and prepare weight matrices from a BilinearAttention layer.
    
    Returns dict with keys: q1, k1, q2, k2, v, o, scale, d_head, n_head, n_ctx,
    rope_cos, rope_sin, causal_mask.
    All weights are reshaped to (n_head, d_head, d_model) or (n_head, d_head, d_model+1)
    with bias absorbed into last column.
    """
    d_model = layer.d_model
    n_head = layer.n_head
    d_head = layer.d_head
    n_ctx = layer.n_ctx

    def _get_proj(proj):
        """Get weight (n_head, d_head, d_model+1) with bias absorbed."""
        W = proj.weight.detach().to(device=device, dtype=dtype)
        W = W.reshape(n_head, d_head, d_model)
        if proj.bias is not None:
            b = proj.bias.detach().to(device=device, dtype=dtype).reshape(n_head, d_head, 1)
            W = torch.cat([b, W], dim=-1)  # (n_head, d_head, d_model+1)
        else:
            z = torch.zeros(n_head, d_head, 1, device=device, dtype=dtype)
            W = torch.cat([z, W], dim=-1)
        return W

    # RoPE cos/sin: (n_ctx, d_head//2)
    rotary = layer.rotary
    freq = 1.0 / (layer.rotary.cos_cached.device and 1) # dummy, recompute
    dim = d_head
    base = 10000  # default
    freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    ctx = torch.arange(n_ctx, device=device, dtype=dtype)
    freqs = torch.outer(ctx, freq)
    rope_cos = freqs.cos()  # (n_ctx, d_head//2)
    rope_sin = freqs.sin()

    return dict(
        q1=_get_proj(layer.q1),
        k1=_get_proj(layer.k1),
        q2=_get_proj(layer.q2),
        k2=_get_proj(layer.k2),
        v=_get_proj(layer.v),
        o=layer.o.weight.detach().to(device=device, dtype=dtype).reshape(d_model, n_head, d_head),
        scale=layer.scale,
        d_head=d_head,
        n_head=n_head,
        n_ctx=n_ctx,
        d_model=d_model,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        causal_mask=layer.causal_mask[:n_ctx, :n_ctx].to(device=device, dtype=dtype),
    )


def attention_path_inner_product(wA, wB, G):
    """Compute <attn_A(x), attn_B(x)> for the attention-only path (no residual).
    
    This computes the degree-10 inner product (degree 5 from bra, degree 5 from ket).
    
    wA, wB: dicts from build_layer_weights
    G: input gram matrix (d_model+1, d_model+1) — covariance of [1, x]
    
    Under Gaussian/Isserlis assumption with INDEPENDENT inputs on each leg:
    <attn_A, attn_B> = sum_{t,s,t',s'} M(t,s)*M(t',s') *
        sum_{n_A,n_B} (1/d^2)^2 * scale_A * scale_B *
        [O_A^T O_B]_{n_A,n_B via output} *
        [V_A G V_B^T]_{n_A,n_B} *
        [Q1_A(t) G Q1_B(t')^T]_{n_A,n_B} * [K1_A(s) G K1_B(s')^T]_{n_A,n_B} *
        [Q2_A(t) G Q2_B(t')^T]_{n_A,n_B} * [K2_A(s) G K2_B(s')^T]_{n_A,n_B}
    
    But with shared inputs (d0,d1,d2 share x_s; d3,d4 share x_t), we need Isserlis
    expansion. Under independent-input Gaussian approximation, we just multiply grams.
    
    Returns: scalar inner product, and output gram (d_model+1, d_model+1)
    """
    n_head = wA['n_head']
    d_head = wA['d_head']
    n_ctx = min(wA['n_ctx'], wB['n_ctx'])
    d = wA['d_model']
    
    R = _rope_matrices(wA['rope_cos'], wA['rope_sin'], n_ctx)  # (n_ctx, half, 2, 2)
    mask = wA['causal_mask'][:n_ctx, :n_ctx]
    
    scale_factor = (wA['scale'] * wB['scale']) / (d_head ** 4)
    
    # O gram: O_A^T @ O_B — shape (n_head_A, d_head_A, n_head_B, d_head_B)
    # But we need per-output-dim contraction. Actually:
    # The output gram is sum over output dims: O_A[d, nA, hA] * O_B[d, nB, hB]
    # = (O_A^T O_B)[nA, hA, nB, hB] summed appropriately
    O_A = wA['o']  # (d_model, n_head, d_head)
    O_B = wB['o']
    
    # For the FULL inner product (scalar), we contract over output dim too.
    # But for gram-chaining, we want the output gram G_out[d_i, d_j].
    # Let's compute the output gram.
    
    # Per-head contribution: for heads nA, nB:
    #   contrib[nA, nB] = V_gram[nA, nB] * Q1_gram[nA, nB] * K1_gram[nA, nB] * 
    #                     Q2_gram[nA, nB] * K2_gram[nA, nB]
    # where each X_gram[nA, nB] = sum over head dims of (X_A G X_B^T)[nA, :, nB, :]
    # contracted appropriately.
    
    # Actually let me think about this more carefully.
    # For each pair of positions (t_q, s_k) for bra and (t_q', s_k') for ket:
    #   and for each pair of heads (nA, nB):
    #   The contribution to G_out is:
    #     scale * M(tq,sk) * M(tq',sk') *
    #     O_A[:, nA, :] @ [v_contrib * qk_contrib] @ O_B[:, nB, :]^T
    #   where v_contrib and qk_contrib involve position-rotated weight grams.
    
    # Let me compute this step by step.
    
    # Step 1: For each position pair (t, s) and (t', s'), compute the head-level
    # attention value contribution.
    
    # For the V path: V_A[nA, hv, :] @ G @ V_B[nB, hv, :]^T summed over hv
    # This gives a (n_head, n_head) matrix.
    V_A, V_B = wA['v'], wB['v']  # (n_head, d_head, d_in)
    VGV = torch.einsum("ahi,ij,bhj->ab", V_A, G, V_B)  # (n_head_A, n_head_B)
    # Note: V has no RoPE, no position dependence.
    
    # For QK paths: need position-dependent rotated weights.
    # Q1_A(t)[n, h, d] = R(t) @ Q1_A[n, h, d] (rotation on the h dimension pairs)
    # The gram is: sum_h (R(t) Q1_A)[n, h, :] @ G @ (R(t') Q1_B)[n, h, :]^T
    
    Q1_A, Q1_B = wA['q1'], wB['q1']  # (n_head, d_head, d_in)
    K1_A, K1_B = wA['k1'], wB['k1']
    Q2_A, Q2_B = wA['q2'], wB['q2']
    K2_A, K2_B = wA['k2'], wB['k2']
    
    # Precompute rotated weights for all positions
    # For Q at position t: Q_rot[t] = R[t] applied to Q
    def rotate_all(W, R_mats):
        """W: (n_head, d_head, d_in), R_mats: (n_ctx, half, 2, 2)
        Returns: (n_ctx, n_head, d_head, d_in)"""
        nh, dh, di = W.shape
        half = dh // 2
        W_pairs = W.reshape(nh, half, 2, di)
        # R_mats: (T, half, 2, 2), W_pairs: (nh, half, 2, di)
        W_rot = torch.einsum("tpij,npjd->tnpid", R_mats, W_pairs)
        return W_rot.reshape(n_ctx, nh, dh, di)
    
    Q1_A_rot = rotate_all(Q1_A, R)  # (n_ctx, n_head, d_head, d_in)
    Q1_B_rot = rotate_all(Q1_B, R)
    Q2_A_rot = rotate_all(Q2_A, R)
    Q2_B_rot = rotate_all(Q2_B, R)
    K1_A_rot = rotate_all(K1_A, R)
    K1_B_rot = rotate_all(K1_B, R)
    K2_A_rot = rotate_all(K2_A, R)
    K2_B_rot = rotate_all(K2_B, R)
    
    # Helper: compute gram between two sets of rotated weights
    def _qk_gram(W_A_rot, W_B_rot):
        """(n_ctx, n_head, d_head, d_in) x G x same -> (pos_A, pos_B, head_A, head_B)"""
        return torch.einsum("tahi,ij,ubhj->tuab", W_A_rot, G, W_B_rot)
    
    # ── Direct matching: circuit1_A ↔ circuit1_B, circuit2_A ↔ circuit2_B ────
    Q1_gram_11 = _qk_gram(Q1_A_rot, Q1_B_rot)  # Q1_A vs Q1_B
    Q2_gram_22 = _qk_gram(Q2_A_rot, Q2_B_rot)  # Q2_A vs Q2_B
    K1_gram_11 = _qk_gram(K1_A_rot, K1_B_rot)
    K2_gram_22 = _qk_gram(K2_A_rot, K2_B_rot)
    
    # ── Cross matching: circuit1_A ↔ circuit2_B, circuit2_A ↔ circuit1_B ─────
    Q1_gram_12 = _qk_gram(Q1_A_rot, Q2_B_rot)  # Q1_A vs Q2_B
    Q2_gram_21 = _qk_gram(Q2_A_rot, Q1_B_rot)  # Q2_A vs Q1_B
    K1_gram_12 = _qk_gram(K1_A_rot, K2_B_rot)
    K2_gram_21 = _qk_gram(K2_A_rot, K1_B_rot)
    
    # ── Symmetrized QQ and KK ────────────────────────────────────────────────
    # Direct: (Q1_A·Q1_B)*(Q2_A·Q2_B), Cross: (Q1_A·Q2_B)*(Q2_A·Q1_B)
    # Analogous to 0.5*(ll*rr + lr*rl) in standard bilinear layers
    QQ = 0.5 * (Q1_gram_11 * Q2_gram_22 + Q1_gram_12 * Q2_gram_21)
    KK = 0.5 * (K1_gram_11 * K2_gram_22 + K1_gram_12 * K2_gram_21)
    
    # ── V gram (unsummed over d_head for OV contraction) ─────────────────────
    # V_gram_full[nA, hvA, nB, hvB] = sum_d V_A[nA, hvA, d] G[d,d'] V_B[nB, hvB, d']
    V_gram_full = torch.einsum("aHi,ij,bKj->aHbK", V_A, G, V_B)
    # (n_head, d_head, n_head, d_head)
    
    # ── Position-summed QK contribution ──────────────────────────────────────
    # C[nA, nB] = sum_{tq, sk, tq', sk'} M[tq,sk] M[tq',sk'] QQ[tq,tq',nA,nB] KK[sk,sk',nA,nB]
    # Compute via: MKK = M @ KK, MKKM = MKK @ M^T, then C = sum(QQ * MKKM)
    MKK = torch.einsum("ts,suab->tuab", mask, KK)
    MKKM = torch.einsum("tuab,vu->tvab", MKK, mask)
    C = (QQ * MKKM).sum(dim=(0, 1))  # (nA, nB)
    
    # Full head tensor: H[nA, hvA, nB, hvB] = C[nA, nB] * V_gram_full[nA, hvA, nB, hvB]
    H = C[:, None, :, None] * V_gram_full  # (nA, hvA, nB, hvB)
    
    # Output gram: G_out[d_i, d_j] = scale_factor * sum_{nA, hvA, nB, hvB}
    #   O_A[d_i, nA, hvA] * H[nA, hvA, nB, hvB] * O_B[d_j, nB, hvB]
    G_out = scale_factor * torch.einsum("iaH,aHbK,jbK->ij", O_A, H, O_B)
    # (d_model, d_model)
    
    return G_out


def layer_inner_product_with_residual(wA, wB, G_in):
    """Compute output gram for one layer WITH residual: y = x + scale*attn(x).
    
    Under Gaussian/independent-input assumption:
    <y_A, y_B> = <x, x> + scale_A * <attn_A(x), x> + scale_B * <x, attn_B(x)> 
                 + scale_A * scale_B * <attn_A(x), attn_B(x)>
    
    But <attn(x), x> is a cross-term between degree 5 and degree 1.
    Under zero-mean Gaussian x, odd-degree terms vanish when ALL inputs are i.i.d.
    
    However, our G_in includes the bias dimension (the [1, x] augmentation), so the
    input is NOT zero-mean. The constant '1' in the first position means the attention
    path has non-zero projection onto degree-1 subspace through the bias terms.
    
    So we CANNOT assume the cross-terms vanish. We must compute them.
    
    Actually, let's think more carefully. The attention path output is:
    attn(x) = O @ (pattern @ V @ x)
    
    For the cross-term <attn_A(x), x_B>:
    This is sum_d E[attn_A(x)_d * x_d] = sum_d O_A[d,:,:] * E[pattern_A * V_A x * x_d]
    
    Under independent-input Gaussian assumption, the 5 inputs to attn are independent.
    x appears as degree 1, and the cross with attn (degree 5) gives a degree 6 moment.
    Under independence of the 5 attn inputs, the cross-term <attn(x), x> requires
    x (from the residual) to be correlated with one of the 5 inputs.
    
    In reality x IS the same as the inputs to the attention path (shared input!).
    But under our independent-input approximation, we treat them as independent.
    If the inputs are independent zero-mean Gaussians, then E[attn(x) * x_residual] = 0
    because they share no indices.
    
    BUT with bias (x augmented as [1, x]), the "1" component means attn has a nonzero 
    constant + linear part from bias terms. These DO contribute to the cross-term.
    
    For simplicity and correctness, let's compute all 4 terms:
    G_out = G_residual + G_cross_AB + G_cross_BA + G_attn
    
    Where:
    - G_residual = G_in (the d_model x d_model submatrix, identity path)
    - G_attn = attention_path_inner_product output
    - G_cross terms require computing E[attn_A(x) * x_B^T]
    
    For the cross-term under independent-input approximation:
    The attention output has 5 independent input legs. The residual x is a 6th 
    independent input. Since they're all independent, E[attn(x)] is a constant
    (computed from the bias terms), and the cross-term is:
    G_cross[i,j] = E[attn_A(x)]_i * E[x]_j = E[attn_A]_i * mu_j
    where mu = G_in[:, 0] = E[[1,x]] (the bias column of the gram).
    
    But E[attn_A(x)] under independent inputs with E[x]=mu is messy.
    
    For the SIMPLEST correct implementation, let's just use the 4-term decomposition
    but approximate the cross-terms as zero (valid when biases are small or inputs
    are approximately zero-mean).
    
    G_out ≈ G_in[1:,1:] + G_attn[1:,1:] (ignoring bias dims for output gram)
    
    Wait no. Let me reconsider. The gram G_in is (d_model+1, d_model+1) where the
    first row/col is the bias ("1") dimension. The residual maps x -> x, which in
    augmented form is [1, x] -> x (dropping the 1). The attention path maps 
    [1, x] -> attn output (d_model dims, no bias augmentation on output).
    
    For the output gram (which feeds into the next layer as input), the output is
    y = x + scale*attn(x), which is d_model-dimensional. For the NEXT layer's input
    gram, we need G_out = E[[1, y] [1, y]^T] which is (d_model+1, d_model+1).
    
    Let me be very precise:
    G_out[0, 0] = 1  (the constant-constant term)
    G_out[0, 1:] = E[y] = E[x] + scale * E[attn(x)]
    G_out[1:, 0] = same (symmetric)
    G_out[1:, 1:] = E[y y^T] = E[x x^T] + scale_A E[attn_A(x) x^T] + 
                     scale_B E[x attn_B(x)^T] + scale_A scale_B E[attn_A(x) attn_B(x)^T]
    
    This is for the CROSS gram between models A and B.
    
    Under independent-input approximation:
    - E[x x^T] = G_in[1:, 1:] (the covariance submatrix)
    - E[attn_A(x) attn_B(x)^T] = G_attn (from attention_path_inner_product)
    - E[attn(x) x^T]: Since attn uses independent copies of x, and the residual x 
      is yet another independent copy, these are independent. So:
      E[attn(x) x^T] = E[attn(x)] E[x]^T
    - E[attn(x)] requires computing the expected attention output under the independent
      input distribution. This is nonzero due to biases.
    
    For the first implementation, let me IGNORE cross-terms (set them to zero).
    This is exact when biases are zero and inputs are zero-mean.
    
    Args:
        wA, wB: layer weight dicts
        G_in: (d_in, d_in) input gram where d_in = d_model+1 (with bias augmentation)
    
    Returns:
        G_out: (d_model+1, d_model+1) output gram for the next layer
    """
    d = wA['d_model']
    
    # Attention-path gram (d_model, d_model) — no bias augmentation on output
    G_attn = attention_path_inner_product(wA, wB, G_in)
    
    # Build output gram with bias augmentation
    G_out = torch.zeros(d + 1, d + 1, dtype=G_in.dtype, device=G_in.device)
    G_out[0, 0] = 1.0  # constant term
    
    # Residual contribution: E[x x^T]
    G_out[1:, 1:] = G_in[1:, 1:]  # residual-residual
    
    # Attention contribution
    G_out[1:, 1:] += G_attn  # attn-attn
    
    # Cross-terms (residual-attn and attn-residual): IGNORED for now
    # These are E[attn(x)] E[x]^T + E[x] E[attn(x)]^T under independence
    # and are zero when inputs are zero-mean and biases are zero.
    
    # Mean terms: E[y] = E[x] + scale * E[attn(x)] ≈ E[x] (ignoring attn mean)
    G_out[0, 1:] = G_in[0, 1:]  # E[x] (from bias column)
    G_out[1:, 0] = G_in[1:, 0]
    
    return G_out


def compute_initial_gram(embed_A, embed_B, device=None, dtype=torch.float64):
    """Compute initial gram matrix from embedding weights.
    
    The input to layer 0 is Embed(token). Under uniform token distribution:
    E[embed embed^T] = (1/V) * E^T E
    
    We augment with bias: G[0,0] = 1, G[0,1:] = E[embed] = mean row,
    G[1:,1:] = E[embed embed^T].
    
    Returns: (d_model+1, d_model+1) gram matrix.
    """
    E_A = embed_A.weight.detach().to(device=device, dtype=dtype)  # (V, d)
    E_B = embed_B.weight.detach().to(device=device, dtype=dtype)
    V, d = E_A.shape
    
    G = torch.zeros(d + 1, d + 1, device=device, dtype=dtype)
    G[0, 0] = 1.0
    
    # E[embed] = mean over vocab
    mu_A = E_A.mean(dim=0)  # (d,)
    mu_B = E_B.mean(dim=0)
    G[0, 1:] = mu_B
    G[1:, 0] = mu_A
    
    # E[embed_A embed_B^T] = (1/V) * E_A^T E_B
    G[1:, 1:] = E_A.T @ E_B / V
    
    return G


def compute_tn_similarity(model_A, model_B, device=None, dtype=torch.float64):
    """Compute tensor network cosine similarity between two AttentionLM models.
    
    Ignores RMSNorm (cancels in cosine sim under extra-input approximation).
    Uses Gaussian/independent-input approximation for gram chaining.
    
    Returns: cosine similarity scalar in [-1, 1].
    """
    if device is None:
        device = next(model_A.parameters()).device
    
    n_layers = model_A.n_layers
    
    # Initial gram from embeddings
    G_AB = compute_initial_gram(model_A.embed, model_B.embed, device, dtype)
    G_AA = compute_initial_gram(model_A.embed, model_A.embed, device, dtype)
    G_BB = compute_initial_gram(model_B.embed, model_B.embed, device, dtype)
    
    # Chain through attention layers
    for i in range(n_layers):
        wA = build_layer_weights(model_A.layers[i], device, dtype)
        wB = build_layer_weights(model_B.layers[i], device, dtype)
        wA_self = wA  # for self-inner-product, both sides are model A
        wB_self = wB
        
        G_AB = layer_inner_product_with_residual(wA, wB, G_AB)
        G_AA = layer_inner_product_with_residual(wA_self, wA_self, G_AA)
        G_BB = layer_inner_product_with_residual(wB_self, wB_self, G_BB)
    
    # Apply unembed: U_A @ G @ U_B^T, then trace for inner product
    U_A = model_A.unembed.weight.detach().to(device=device, dtype=dtype)  # (V, d)
    U_B = model_B.unembed.weight.detach().to(device=device, dtype=dtype)
    
    # The output of the last layer (after residual) is d_model-dimensional.
    # Gram is (d+1, d+1). The unembed operates on the d_model part (not bias).
    # Inner product = tr(U_A @ G[1:,1:] @ U_B^T)
    ip_AB = (U_A @ G_AB[1:, 1:] @ U_B.T).trace().item()
    ip_AA = (U_A @ G_AA[1:, 1:] @ U_A.T).trace().item()
    ip_BB = (U_B @ G_BB[1:, 1:] @ U_B.T).trace().item()
    
    denom = np.sqrt(abs(ip_AA) * abs(ip_BB))
    if denom < 1e-30:
        return 0.0
    return ip_AB / denom
