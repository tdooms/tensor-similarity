"""Test 4: Causal masking - strict future-zero pattern."""
import pytest
import torch

from models.attention_kernels.bilinear import QuadraticAttention
from tests.architecture_tests.conftest import B, T, D_MODEL, N_HEAD, N_CTX


def test_causal_masking_future_zero():
    """Test that attention pattern has exact zeros for future positions."""
    attn = QuadraticAttention(
        d_model=D_MODEL,
        n_head=N_HEAD,
        n_ctx=N_CTX,
        scale=0.2,
        use_rmsnorm_qk=False,
    )
    
    x = torch.randn(B, T, D_MODEL)
    
    _, debug = attn(x, return_debug=True)
    pattern = debug["pattern"]
    
    for t in range(T):
        for s in range(t + 1, T):
            assert (pattern[:, :, t, s] == 0).all(), \
                f"pattern[..., {t}, {s}] should be exactly 0 (future leakage)"


def test_causal_masking_past_nonzero():
    """Test that attention pattern can have non-zeros for past/present positions."""
    attn = QuadraticAttention(
        d_model=D_MODEL,
        n_head=N_HEAD,
        n_ctx=N_CTX,
        scale=0.2,
        use_rmsnorm_qk=False,
    )
    
    torch.manual_seed(42)
    x = torch.randn(B, T, D_MODEL) * 10
    
    _, debug = attn(x, return_debug=True)
    pattern = debug["pattern"]
    
    lower_tri_mask = torch.tril(torch.ones(T, T))
    lower_tri_values = pattern[:, :, lower_tri_mask.bool()]
    
    assert (lower_tri_values != 0).any(), \
        "Some past/present attention values should be non-zero"


def test_causal_masking_shape():
    """Test that pattern has correct shape."""
    attn = QuadraticAttention(
        d_model=D_MODEL,
        n_head=N_HEAD,
        n_ctx=N_CTX,
        scale=0.2,
    )
    
    x = torch.randn(B, T, D_MODEL)
    
    _, debug = attn(x, return_debug=True)
    pattern = debug["pattern"]
    
    assert pattern.shape == (B, N_HEAD, T, T), \
        f"Expected {(B, N_HEAD, T, T)}, got {pattern.shape}"


def test_causal_masking_different_sequence_lengths():
    """Test causal masking works for different sequence lengths."""
    attn = QuadraticAttention(
        d_model=D_MODEL,
        n_head=N_HEAD,
        n_ctx=N_CTX,
        scale=0.2,
    )
    
    for seq_len in [1, 2, 4, 8]:
        x = torch.randn(B, seq_len, D_MODEL)
        _, debug = attn(x, return_debug=True)
        pattern = debug["pattern"]
        
        for t in range(seq_len):
            for s in range(t + 1, seq_len):
                assert (pattern[:, :, t, s] == 0).all(), \
                    f"seq_len={seq_len}: pattern[..., {t}, {s}] should be 0"
