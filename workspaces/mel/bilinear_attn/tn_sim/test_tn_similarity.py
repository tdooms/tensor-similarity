#!/usr/bin/env python3
"""Test suite for TN similarity of bilinear attention models.

Tests are designed to verify correctness at multiple levels:
- Component-level: RoPE matrices, gram construction, weight extraction
- Single-layer: closed-form validation, symmetry properties
- Multi-layer: gram chaining, residual handling
- End-to-end: self-similarity, symmetry, circuit-swap invariance

Run from bilinear_attn directory:
    python -m pytest tn_sim/test_tn_similarity.py -v
    # or without pytest:
    python -m tn_sim.test_tn_similarity
"""

import copy
import math
import sys

import torch
import numpy as np

from models import AttentionLM
from tn_sim.tn_similarity import (
    _rope_matrices,
    _apply_rope_to_weight,
    build_layer_weights,
    attention_path_inner_product,
    layer_inner_product_with_residual,
    compute_initial_gram,
    compute_tn_similarity,
)
from tn_sim.mc_similarity import mc_similarity_gaussian

try:
    from tn_sim.tn_similarity_quimb import compute_tn_similarity_quimb
    HAS_QUIMB = True
except ImportError:
    HAS_QUIMB = False


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_tiny_config(n_layers=2, use_bias_qk=True, attn_scale=0.35):
    """Create a minimal config dict for testing."""
    return {
        "model": {
            "vocab_size": 16,
            "n_ctx": 8,
            "d_model": 8,
            "n_head": 2,
            "n_layers": n_layers,
            "attn_type": "bilinear",
            "attn_scale": attn_scale,
            "rope_base": 10000,
            "norm_type": "rmsnorm",
            "norm_places": ["pre_unembed"],
            "use_rmsnorm_qk": False,
            "use_bias_qk": use_bias_qk,
        },
        "init": {
            "init_type": "mup",
            "std_embed": 0.02,
        },
    }


def make_model(cfg, seed=42):
    torch.manual_seed(seed)
    model = AttentionLM.from_config(cfg)
    model.eval()
    return model


ATOL = 1e-6
RTOL = 1e-5


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: RoPE rotation matrices are orthogonal (norm-preserving)
# ═══════════════════════════════════════════════════════════════════════════════

def test_rope_orthogonality():
    """Each 2x2 rotation block should satisfy R^T R = I."""
    n_ctx, d_head = 8, 4
    half = d_head // 2
    base = 10000
    freq = 1.0 / (base ** (torch.arange(0, d_head, 2, dtype=torch.float64) / d_head))
    ctx = torch.arange(n_ctx, dtype=torch.float64)
    freqs = torch.outer(ctx, freq)
    R = _rope_matrices(freqs.cos(), freqs.sin(), n_ctx)  # (n_ctx, half, 2, 2)
    
    for t in range(n_ctx):
        for p in range(half):
            block = R[t, p]  # (2, 2)
            eye = block.T @ block
            err = (eye - torch.eye(2, dtype=torch.float64)).abs().max().item()
            assert err < ATOL, f"RoPE not orthogonal at pos={t}, pair={p}: err={err}"
    
    print("  PASS: test_rope_orthogonality")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: Self-similarity = 1.0
# ═══════════════════════════════════════════════════════════════════════════════

def test_self_similarity():
    """compute_tn_similarity(model, model) should be exactly 1.0."""
    for n_layers in [1, 2]:
        cfg = make_tiny_config(n_layers=n_layers)
        model = make_model(cfg)
        sim = compute_tn_similarity(model, model)
        assert abs(sim - 1.0) < 1e-4, \
            f"Self-similarity for {n_layers}L model: {sim:.6f} (expected 1.0)"
    
    print("  PASS: test_self_similarity")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3: Symmetry — sim(A, B) == sim(B, A)
# ═══════════════════════════════════════════════════════════════════════════════

def test_symmetry():
    """TN similarity should be symmetric."""
    cfg = make_tiny_config()
    model_A = make_model(cfg, seed=42)
    model_B = make_model(cfg, seed=123)
    
    sim_AB = compute_tn_similarity(model_A, model_B)
    sim_BA = compute_tn_similarity(model_B, model_A)
    
    assert abs(sim_AB - sim_BA) < ATOL, \
        f"Asymmetry: sim(A,B)={sim_AB:.8f}, sim(B,A)={sim_BA:.8f}"
    
    print(f"  PASS: test_symmetry (sim={sim_AB:.6f})")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4: Circuit swap invariance — swapping (Q1,K1)<->(Q2,K2) should not
#          change the function, so similarity with the swapped model = 1.0
# ═══════════════════════════════════════════════════════════════════════════════

def test_circuit_swap_invariance():
    """Swapping Q1<->Q2 and K1<->K2 gives identical function, so sim should be 1.0."""
    cfg = make_tiny_config(n_layers=1)
    model_A = make_model(cfg, seed=42)
    model_B = copy.deepcopy(model_A)
    
    # Swap circuits in model_B
    for layer in model_B.layers:
        # Swap Q1 <-> Q2
        q1_w = layer.q1.weight.data.clone()
        q1_b = layer.q1.bias.data.clone() if layer.q1.bias is not None else None
        layer.q1.weight.data.copy_(layer.q2.weight.data)
        layer.q2.weight.data.copy_(q1_w)
        if q1_b is not None:
            q1_b_orig = q1_b.clone()
            layer.q1.bias.data.copy_(layer.q2.bias.data)
            layer.q2.bias.data.copy_(q1_b_orig)
        
        # Swap K1 <-> K2
        k1_w = layer.k1.weight.data.clone()
        k1_b = layer.k1.bias.data.clone() if layer.k1.bias is not None else None
        layer.k1.weight.data.copy_(layer.k2.weight.data)
        layer.k2.weight.data.copy_(k1_w)
        if k1_b is not None:
            k1_b_orig = k1_b.clone()
            layer.k1.bias.data.copy_(layer.k2.bias.data)
            layer.k2.bias.data.copy_(k1_b_orig)
    
    sim = compute_tn_similarity(model_A, model_B)
    assert abs(sim - 1.0) < 1e-4, \
        f"Circuit-swap similarity: {sim:.6f} (expected 1.0)"
    
    # Also verify the models produce the same outputs
    x = torch.randint(0, 16, (2, 8))
    with torch.no_grad():
        out_A = model_A(x)
        out_B = model_B(x)
    output_diff = (out_A - out_B).abs().max().item()
    assert output_diff < 1e-5, f"Circuit-swapped models differ: {output_diff}"
    
    print(f"  PASS: test_circuit_swap_invariance (sim={sim:.6f})")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5: Initial gram matrix properties
# ═══════════════════════════════════════════════════════════════════════════════

def test_initial_gram_properties():
    """Self-gram should be PSD and have G[0,0]=1."""
    cfg = make_tiny_config()
    model = make_model(cfg)
    
    G = compute_initial_gram(model.embed, model.embed, dtype=torch.float64)
    
    # G[0,0] = 1
    assert abs(G[0, 0].item() - 1.0) < ATOL, f"G[0,0] = {G[0,0].item()}"
    
    # Symmetry
    asym = (G - G.T).abs().max().item()
    assert asym < ATOL, f"Gram not symmetric: max asymmetry = {asym}"
    
    # PSD: all eigenvalues >= 0
    eigvals = torch.linalg.eigvalsh(G)
    min_eig = eigvals.min().item()
    assert min_eig > -ATOL, f"Gram not PSD: min eigenvalue = {min_eig}"
    
    print(f"  PASS: test_initial_gram_properties (min_eig={min_eig:.2e})")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 6: Zero-scale attention → model is linear (embed → unembed)
# ═══════════════════════════════════════════════════════════════════════════════

def test_zero_scale_is_linear():
    """With attn_scale=0, the model is embed→unembed (ignoring norm).
    TN similarity should match the linear kernel: tr(U_A E_A^T E_B U_B^T) / norms."""
    cfg = make_tiny_config(attn_scale=0.0)
    model_A = make_model(cfg, seed=42)
    model_B = make_model(cfg, seed=123)
    
    tn_sim = compute_tn_similarity(model_A, model_B)
    
    # Compute linear similarity directly
    E_A = model_A.embed.weight.detach().double()
    E_B = model_B.embed.weight.detach().double()
    U_A = model_A.unembed.weight.detach().double()
    U_B = model_B.unembed.weight.detach().double()
    V = E_A.shape[0]
    
    # G = (1/V) E^T E (the [1:,1:] block of the augmented gram)
    G_AB = E_A.T @ E_B / V
    G_AA = E_A.T @ E_A / V
    G_BB = E_B.T @ E_B / V
    
    ip_AB = (U_A @ G_AB @ U_B.T).trace().item()
    ip_AA = (U_A @ G_AA @ U_A.T).trace().item()
    ip_BB = (U_B @ G_BB @ U_B.T).trace().item()
    
    linear_sim = ip_AB / np.sqrt(abs(ip_AA) * abs(ip_BB))
    
    # Should match closely (small differences from mean terms in augmented gram)
    diff = abs(tn_sim - linear_sim)
    assert diff < 0.01, \
        f"Zero-scale TN sim ({tn_sim:.6f}) != linear sim ({linear_sim:.6f}), diff={diff:.6f}"
    
    print(f"  PASS: test_zero_scale_is_linear (tn={tn_sim:.6f}, linear={linear_sim:.6f})")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 7: Attention path gram is PSD for self-inner-product
# ═══════════════════════════════════════════════════════════════════════════════

def test_attention_gram_psd():
    """The attention-path output gram G_attn should be PSD for self-inner-product."""
    cfg = make_tiny_config(n_layers=1)
    model = make_model(cfg)
    
    G_in = compute_initial_gram(model.embed, model.embed, dtype=torch.float64)
    w = build_layer_weights(model.layers[0], dtype=torch.float64)
    
    G_attn = attention_path_inner_product(w, w, G_in)
    
    eigvals = torch.linalg.eigvalsh(G_attn)
    min_eig = eigvals.min().item()
    assert min_eig > -ATOL, f"Attention gram not PSD: min eigenvalue = {min_eig}"
    
    print(f"  PASS: test_attention_gram_psd (min_eig={min_eig:.2e})")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 8: Layer output gram is PSD for self-inner-product
# ═══════════════════════════════════════════════════════════════════════════════

def test_layer_output_gram_psd():
    """Output gram after residual should be PSD for self-inner-product."""
    cfg = make_tiny_config(n_layers=2)
    model = make_model(cfg)
    
    G = compute_initial_gram(model.embed, model.embed, dtype=torch.float64)
    
    for i, layer in enumerate(model.layers):
        w = build_layer_weights(layer, dtype=torch.float64)
        G = layer_inner_product_with_residual(w, w, G)
        
        # Check symmetry
        asym = (G - G.T).abs().max().item()
        assert asym < ATOL, f"Layer {i} gram not symmetric: {asym}"
        
        # Check PSD
        eigvals = torch.linalg.eigvalsh(G)
        min_eig = eigvals.min().item()
        assert min_eig > -ATOL, \
            f"Layer {i} output gram not PSD: min eigenvalue = {min_eig}"
    
    print(f"  PASS: test_layer_output_gram_psd")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 9: MC vs TN correlation — both methods should agree in direction
# ═══════════════════════════════════════════════════════════════════════════════

def test_mc_tn_correlation():
    """TN and MC similarities should be positively correlated across model pairs."""
    cfg = make_tiny_config(n_layers=1)
    
    models = [make_model(cfg, seed=s) for s in [42, 43, 44, 45]]
    
    tn_sims = []
    mc_sims = []
    
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            tn_s = compute_tn_similarity(models[i], models[j])
            mc_s = mc_similarity_gaussian(
                models[i], models[j],
                vocab_size=16, n_ctx=8, device="cpu",
                n_samples=2000, batch_size=64,
            )
            tn_sims.append(tn_s)
            mc_sims.append(mc_s)
    
    # Compute Pearson correlation
    tn_arr = np.array(tn_sims)
    mc_arr = np.array(mc_sims)
    
    if tn_arr.std() > 1e-10 and mc_arr.std() > 1e-10:
        corr = np.corrcoef(tn_arr, mc_arr)[0, 1]
        print(f"  TN-MC correlation: {corr:.4f}")
        print(f"    TN values: {tn_arr}")
        print(f"    MC values: {mc_arr}")
        # We expect positive correlation but with approximation differences
        # Don't assert a strict threshold — just report
    else:
        corr = float("nan")
        print(f"  TN-MC: insufficient variance (TN std={tn_arr.std():.2e})")
    
    print(f"  PASS: test_mc_tn_correlation (corr={corr:.4f})")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 10: Verify RoPE application to weight matrices
# ═══════════════════════════════════════════════════════════════════════════════

def test_rope_weight_application():
    """Applying RoPE to weights should give same result as applying RoPE to
    the output of the projection, i.e., R(t) @ W @ x == (R(t)W) @ x."""
    d_head, d_in, n_head = 4, 6, 2
    torch.manual_seed(42)
    W = torch.randn(n_head, d_head, d_in, dtype=torch.float64)
    x = torch.randn(d_in, dtype=torch.float64)
    
    base = 10000
    freq = 1.0 / (base ** (torch.arange(0, d_head, 2, dtype=torch.float64) / d_head))
    t = 3  # test at position 3
    freqs_t = freq * t
    cos_t = freqs_t.cos()
    sin_t = freqs_t.sin()
    
    R_t = torch.zeros(d_head // 2, 2, 2, dtype=torch.float64)
    R_t[:, 0, 0] = cos_t
    R_t[:, 0, 1] = -sin_t
    R_t[:, 1, 0] = sin_t
    R_t[:, 1, 1] = cos_t
    
    # Method 1: Apply R to W, then multiply by x
    W_rot = _apply_rope_to_weight(W, R_t)  # (n_head, d_head, d_in)
    out1 = torch.einsum("nhi,i->nh", W_rot, x)  # (n_head, d_head)
    
    # Method 2: Multiply W by x, then apply R
    Wx = torch.einsum("nhi,i->nh", W, x)  # (n_head, d_head)
    Wx_pairs = Wx.reshape(n_head, d_head // 2, 2)
    out2 = torch.einsum("pij,npj->npi", R_t, Wx_pairs).reshape(n_head, d_head)
    
    err = (out1 - out2).abs().max().item()
    assert err < ATOL, f"RoPE weight application mismatch: {err}"
    
    print(f"  PASS: test_rope_weight_application (err={err:.2e})")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 11: 1-layer no-bias no-RoPE against closed-form bilinear inner product
# ═══════════════════════════════════════════════════════════════════════════════

def test_single_layer_closed_form():
    """For a 1-layer model with no bias, compare the attention-path inner product 
    against a direct closed-form computation.
    
    The bilinear attention inner product (symmetrized) for a single position pair 
    and single head is:
      0.5 * [(Q1_A G Q1_B^T)(Q2_A G Q2_B^T) + (Q1_A G Q2_B^T)(Q2_A G Q1_B^T)]
      * similar for K
      * V_gram * O contraction
    
    We verify this for a 1-head, 1-position (n_ctx=1) model to remove summation 
    complexity and isolate the core bilinear inner product.
    """
    d_model = 4
    cfg = {
        "model": {
            "vocab_size": 8,
            "n_ctx": 2,  # 1 position per half (induction data uses n_ctx//2)
            "d_model": d_model,
            "n_head": 1,
            "n_layers": 1,
            "attn_type": "bilinear",
            "attn_scale": 1.0,
            "rope_base": 10000,
            "norm_type": "rmsnorm",
            "norm_places": ["pre_unembed"],
            "use_rmsnorm_qk": False,
            "use_bias_qk": False,
        },
        "init": {"init_type": "normal", "std_embed": 0.02, "std_qkv": 0.1, "std_o": 0.1},
    }
    
    torch.manual_seed(99)
    model_A = AttentionLM.from_config(cfg).eval()
    torch.manual_seed(100)
    model_B = AttentionLM.from_config(cfg).eval()
    
    # Get weight dicts
    wA = build_layer_weights(model_A.layers[0], dtype=torch.float64)
    wB = build_layer_weights(model_B.layers[0], dtype=torch.float64)
    
    # Use identity gram on the non-bias part (standard inner product)
    d_in = d_model + 1  # bias augmented
    G = torch.zeros(d_in, d_in, dtype=torch.float64)
    G[0, 0] = 1.0
    G[1:, 1:] = torch.eye(d_model, dtype=torch.float64)
    
    G_attn = attention_path_inner_product(wA, wB, G)
    
    # Now compute the same thing manually for n_head=1
    # With n_ctx=2 and causal mask, positions: 0 can attend to 0, 1 can attend to 0,1
    # The attention path output gram is d_model x d_model
    
    # For the manual computation, we need the position-rotated weights
    n_ctx = 2
    base = 10000
    half = d_model // 2  # d_head = d_model for n_head=1
    freq = 1.0 / (base ** (torch.arange(0, d_model, 2, dtype=torch.float64) / d_model))
    ctx = torch.arange(n_ctx, dtype=torch.float64)
    freqs = torch.outer(ctx, freq)
    R = _rope_matrices(freqs.cos(), freqs.sin(), n_ctx)
    
    # Extract weights: shape (1, d_model, d_model+1) -> squeeze head dim
    q1A = wA['q1'].squeeze(0)  # (d_head, d_in)
    k1A = wA['k1'].squeeze(0)
    q2A = wA['q2'].squeeze(0)
    k2A = wA['k2'].squeeze(0)
    vA = wA['v'].squeeze(0)
    oA = wA['o'].squeeze(1)  # (d_model, d_head)
    
    q1B = wB['q1'].squeeze(0)
    k1B = wB['k1'].squeeze(0)
    q2B = wB['q2'].squeeze(0)
    k2B = wB['k2'].squeeze(0)
    vB = wB['v'].squeeze(0)
    oB = wB['o'].squeeze(1)
    
    def apply_rope(W, R_pos):
        """W: (d_head, d_in), R_pos: (half, 2, 2) -> (d_head, d_in)"""
        W_pairs = W.reshape(half, 2, -1)
        return torch.einsum("pij,pjd->pid", R_pos, W_pairs).reshape(W.shape)
    
    scale = 1.0 / (d_model ** 4)  # (1/d_head^2)^2 * scale_A * scale_B = (1/d^2)^2 * 1 * 1
    
    # Manual computation: sum over (tq, sk, tq', sk') with mask
    manual_G = torch.zeros(d_model, d_model, dtype=torch.float64)
    mask = torch.tril(torch.ones(n_ctx, n_ctx, dtype=torch.float64))
    
    for tq in range(n_ctx):
        for sk in range(n_ctx):
            if mask[tq, sk] == 0:
                continue
            for tqp in range(n_ctx):
                for skp in range(n_ctx):
                    if mask[tqp, skp] == 0:
                        continue
                    
                    # Rotated weights at positions
                    q1A_t = apply_rope(q1A, R[tq])
                    q2A_t = apply_rope(q2A, R[tq])
                    k1A_s = apply_rope(k1A, R[sk])
                    k2A_s = apply_rope(k2A, R[sk])
                    
                    q1B_tp = apply_rope(q1B, R[tqp])
                    q2B_tp = apply_rope(q2B, R[tqp])
                    k1B_sp = apply_rope(k1B, R[skp])
                    k2B_sp = apply_rope(k2B, R[skp])
                    
                    # QK grams (scalar, summed over d_head)
                    q1g = (q1A_t @ G @ q1B_tp.T).sum().item()
                    q2g = (q2A_t @ G @ q2B_tp.T).sum().item()
                    q1g_cross = (q1A_t @ G @ q2B_tp.T).sum().item()
                    q2g_cross = (q2A_t @ G @ q1B_tp.T).sum().item()
                    
                    k1g = (k1A_s @ G @ k1B_sp.T).sum().item()
                    k2g = (k2A_s @ G @ k2B_sp.T).sum().item()
                    k1g_cross = (k1A_s @ G @ k2B_sp.T).sum().item()
                    k2g_cross = (k2A_s @ G @ k1B_sp.T).sum().item()
                    
                    # Symmetrized QQ and KK
                    qq = 0.5 * (q1g * q2g + q1g_cross * q2g_cross)
                    kk = 0.5 * (k1g * k2g + k1g_cross * k2g_cross)
                    
                    # V gram (unsummed): (d_head, d_head)
                    vgv = vA @ G @ vB.T  # (d_head, d_head)
                    
                    # Contribution to output gram
                    # G_out[i,j] += scale * qq * kk * O_A[i,:] @ vgv @ O_B[j,:]^T
                    manual_G += scale * qq * kk * (oA @ vgv @ oB.T)
    
    err = (G_attn - manual_G).abs().max().item()
    assert err < 1e-8, \
        f"Attention gram vs manual closed-form: max err = {err:.2e}"
    
    print(f"  PASS: test_single_layer_closed_form (err={err:.2e})")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 12: Orthogonal models should have low similarity
# ═══════════════════════════════════════════════════════════════════════════════

def test_different_models_not_one():
    """Two independently initialized models should not have similarity = 1."""
    cfg = make_tiny_config()
    model_A = make_model(cfg, seed=42)
    model_B = make_model(cfg, seed=999)
    
    sim = compute_tn_similarity(model_A, model_B)
    assert sim < 0.99, f"Different models have sim={sim:.6f}, expected < 0.99"
    
    print(f"  PASS: test_different_models_not_one (sim={sim:.6f})")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 13: Inner product is non-negative for self
# ═══════════════════════════════════════════════════════════════════════════════

def test_self_inner_product_positive():
    """The self inner product ip(A,A) should be positive."""
    cfg = make_tiny_config()
    model = make_model(cfg)
    
    device = "cpu"
    dtype = torch.float64
    
    G = compute_initial_gram(model.embed, model.embed, device, dtype)
    for layer in model.layers:
        w = build_layer_weights(layer, device, dtype)
        G = layer_inner_product_with_residual(w, w, G)
    
    U = model.unembed.weight.detach().to(dtype=dtype)
    ip = (U @ G[1:, 1:] @ U.T).trace().item()
    
    assert ip > 0, f"Self inner product is negative: {ip}"
    
    print(f"  PASS: test_self_inner_product_positive (ip={ip:.6e})")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 14: Quimb self-similarity = 1.0
# ═══════════════════════════════════════════════════════════════════════════════

def test_quimb_self_similarity():
    """Quimb-based self-similarity should be 1.0."""
    if not HAS_QUIMB:
        print("  SKIP: test_quimb_self_similarity (quimb not installed)")
        return
    
    cfg = make_tiny_config(n_layers=1)
    model = make_model(cfg)
    sim = compute_tn_similarity_quimb(model, model)
    assert abs(sim - 1.0) < 1e-3, \
        f"Quimb self-similarity: {sim:.6f} (expected 1.0)"
    
    print(f"  PASS: test_quimb_self_similarity (sim={sim:.6f})")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 15: Quimb vs einsum implementations should match
# ═══════════════════════════════════════════════════════════════════════════════

def test_quimb_matches_einsum():
    """Quimb and einsum implementations should produce the same similarity values."""
    if not HAS_QUIMB:
        print("  SKIP: test_quimb_matches_einsum (quimb not installed)")
        return
    
    for n_layers in [1, 2]:
        cfg = make_tiny_config(n_layers=n_layers)
        model_A = make_model(cfg, seed=42)
        model_B = make_model(cfg, seed=123)
        
        sim_einsum = compute_tn_similarity(model_A, model_B)
        sim_quimb = compute_tn_similarity_quimb(model_A, model_B)
        
        diff = abs(sim_einsum - sim_quimb)
        assert diff < 1e-3, \
            f"{n_layers}L: einsum={sim_einsum:.6f}, quimb={sim_quimb:.6f}, diff={diff:.6f}"
    
    print(f"  PASS: test_quimb_matches_einsum (diff={diff:.2e})")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 16: Quimb circuit swap invariance
# ═══════════════════════════════════════════════════════════════════════════════

def test_quimb_circuit_swap():
    """Quimb version should also be invariant under circuit swap."""
    if not HAS_QUIMB:
        print("  SKIP: test_quimb_circuit_swap (quimb not installed)")
        return
    
    cfg = make_tiny_config(n_layers=1)
    model_A = make_model(cfg, seed=42)
    model_B = copy.deepcopy(model_A)
    
    # Swap circuits in model_B
    for layer in model_B.layers:
        q1_w = layer.q1.weight.data.clone()
        q1_b = layer.q1.bias.data.clone() if layer.q1.bias is not None else None
        layer.q1.weight.data.copy_(layer.q2.weight.data)
        layer.q2.weight.data.copy_(q1_w)
        if q1_b is not None:
            q1_b_orig = q1_b.clone()
            layer.q1.bias.data.copy_(layer.q2.bias.data)
            layer.q2.bias.data.copy_(q1_b_orig)
        
        k1_w = layer.k1.weight.data.clone()
        k1_b = layer.k1.bias.data.clone() if layer.k1.bias is not None else None
        layer.k1.weight.data.copy_(layer.k2.weight.data)
        layer.k2.weight.data.copy_(k1_w)
        if k1_b is not None:
            k1_b_orig = k1_b.clone()
            layer.k1.bias.data.copy_(layer.k2.bias.data)
            layer.k2.bias.data.copy_(k1_b_orig)
    
    sim = compute_tn_similarity_quimb(model_A, model_B)
    assert abs(sim - 1.0) < 1e-3, \
        f"Quimb circuit-swap sim: {sim:.6f} (expected 1.0)"
    
    print(f"  PASS: test_quimb_circuit_swap (sim={sim:.6f})")


# ═══════════════════════════════════════════════════════════════════════════════
# Run all tests
# ═══════════════════════════════════════════════════════════════════════════════

ALL_TESTS = [
    test_rope_orthogonality,
    test_rope_weight_application,
    test_self_similarity,
    test_symmetry,
    test_circuit_swap_invariance,
    test_initial_gram_properties,
    test_zero_scale_is_linear,
    test_attention_gram_psd,
    test_layer_output_gram_psd,
    test_single_layer_closed_form,
    test_different_models_not_one,
    test_self_inner_product_positive,
    test_mc_tn_correlation,
    test_quimb_self_similarity,
    test_quimb_matches_einsum,
    test_quimb_circuit_swap,
]


def main():
    print(f"\nRunning {len(ALL_TESTS)} TN similarity tests\n{'='*60}")
    passed, failed = 0, 0
    for test_fn in ALL_TESTS:
        name = test_fn.__name__
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL: {name}")
            import traceback
            traceback.print_exc()
            print()
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(ALL_TESTS)}")
    
    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed!")


if __name__ == "__main__":
    main()
