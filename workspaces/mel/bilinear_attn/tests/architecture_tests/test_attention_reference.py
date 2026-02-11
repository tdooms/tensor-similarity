"""Test 5: Quadratic attention math matches naive reference."""
import pytest
import torch

from models.attention_kernels.bilinear import QuadraticAttention
from tests.architecture_tests.conftest import B, T, D_MODEL, N_HEAD, N_CTX, ATOL, RTOL


def naive_quadratic_attention(q, k, v, d_head):
    """Naive reference implementation with explicit loops.
    
    Args:
        q: (B, T, n_head, d_head)
        k: (B, T, n_head, d_head)
        v: (B, T, n_head, d_head)
        d_head: head dimension for scaling
        
    Returns:
        pattern: (B, n_head, T, T)
        z: (B, T, n_head, d_head)
    """
    B, T, n_head, _ = q.shape
    
    pattern = torch.zeros(B, n_head, T, T)
    z = torch.zeros_like(v)
    
    for b in range(B):
        for h in range(n_head):
            for t in range(T):
                for s in range(T):
                    score = (q[b, t, h] * k[b, s, h]).sum()
                    
                    scaled_score = score / d_head
                    squared = scaled_score ** 2
                    
                    if s <= t:
                        pattern[b, h, t, s] = squared
                    else:
                        pattern[b, h, t, s] = 0.0
            
            for t in range(T):
                for i in range(v.shape[-1]):
                    acc = 0.0
                    for s in range(T):
                        acc += pattern[b, h, t, s] * v[b, s, h, i]
                    z[b, t, h, i] = acc
    
    return pattern, z


def test_pattern_matches_reference():
    """Test that pattern matches naive reference implementation."""
    d_head = D_MODEL // N_HEAD
    
    attn = QuadraticAttention(
        d_model=D_MODEL,
        n_head=N_HEAD,
        n_ctx=N_CTX,
        scale=0.2,
        use_rmsnorm_qk=False,
    )
    
    torch.manual_seed(42)
    x = torch.randn(B, T, D_MODEL)
    
    _, debug = attn(x, return_debug=True)
    
    q = debug["q"]
    k = debug["k"]
    v = debug["v"]
    pattern = debug["pattern"]
    z = debug["z"]
    
    pattern_ref, z_ref = naive_quadratic_attention(q, k, v, d_head)
    
    assert torch.allclose(pattern, pattern_ref, atol=ATOL, rtol=RTOL), \
        f"Pattern mismatch. Max diff: {(pattern - pattern_ref).abs().max()}"


def test_z_matches_reference():
    """Test that z (attended values) matches naive reference."""
    d_head = D_MODEL // N_HEAD
    
    attn = QuadraticAttention(
        d_model=D_MODEL,
        n_head=N_HEAD,
        n_ctx=N_CTX,
        scale=0.2,
        use_rmsnorm_qk=False,
    )
    
    torch.manual_seed(42)
    x = torch.randn(B, T, D_MODEL)
    
    _, debug = attn(x, return_debug=True)
    
    q = debug["q"]
    k = debug["k"]
    v = debug["v"]
    z = debug["z"]
    
    pattern_ref, z_ref = naive_quadratic_attention(q, k, v, d_head)
    
    assert torch.allclose(z, z_ref, atol=ATOL, rtol=RTOL), \
        f"Z mismatch. Max diff: {(z - z_ref).abs().max()}"


def test_scaling_is_correct():
    """Test that scaling by 1/d_head (not sqrt) is used."""
    d_head = D_MODEL // N_HEAD
    
    attn = QuadraticAttention(
        d_model=D_MODEL,
        n_head=N_HEAD,
        n_ctx=N_CTX,
        scale=0.2,
        use_rmsnorm_qk=False,
    )
    
    torch.manual_seed(42)
    x = torch.randn(B, T, D_MODEL)
    
    _, debug = attn(x, return_debug=True)
    
    scores = debug["scores"]
    pattern = debug["pattern"]
    
    expected_pattern = (scores / d_head).square()
    mask = torch.tril(torch.ones(T, T))[None, None, :, :]
    expected_pattern = expected_pattern * mask
    
    assert torch.allclose(pattern, expected_pattern, atol=ATOL, rtol=RTOL), \
        "Pattern should be (scores/d_head)^2 * mask"
