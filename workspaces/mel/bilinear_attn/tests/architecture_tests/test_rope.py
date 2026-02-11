"""Test 3: RoPE is norm-preserving."""
import pytest
import torch

from models.attention_kernels.rotary import Rotary
from tests.architecture_tests.conftest import B, T, N_HEAD, D_MODEL, ATOL, RTOL


def test_rope_preserves_norm():
    """Test that RoPE preserves L2 norm of each (d_head,) vector."""
    d_head = D_MODEL // N_HEAD
    n_ctx = 32
    
    rotary = Rotary(dim=d_head, n_ctx=n_ctx)
    
    x = torch.randn(B, T, N_HEAD, d_head)
    
    y = rotary(x)
    
    x_norms = x.norm(dim=-1)
    y_norms = y.norm(dim=-1)
    
    assert torch.allclose(x_norms, y_norms, atol=ATOL, rtol=RTOL), \
        f"RoPE should preserve L2 norm. Max diff: {(x_norms - y_norms).abs().max()}"


def test_rope_different_positions_different_rotations():
    """Test that different positions get different rotations."""
    d_head = D_MODEL // N_HEAD
    n_ctx = 32
    
    rotary = Rotary(dim=d_head, n_ctx=n_ctx)
    
    x = torch.ones(1, 4, 1, d_head)
    
    y = rotary(x)
    
    for t1 in range(4):
        for t2 in range(t1 + 1, 4):
            assert not torch.allclose(y[0, t1], y[0, t2]), \
                f"Positions {t1} and {t2} should have different rotations"


def test_rope_deterministic():
    """Test that RoPE is deterministic."""
    d_head = D_MODEL // N_HEAD
    n_ctx = 32
    
    rotary = Rotary(dim=d_head, n_ctx=n_ctx)
    
    x = torch.randn(B, T, N_HEAD, d_head)
    
    y1 = rotary(x)
    y2 = rotary(x)
    
    assert torch.equal(y1, y2), "RoPE should be deterministic"


def test_rope_batch_independence():
    """Test that RoPE applies independently across batch dimension."""
    d_head = D_MODEL // N_HEAD
    n_ctx = 32
    
    rotary = Rotary(dim=d_head, n_ctx=n_ctx)
    
    x1 = torch.randn(1, T, N_HEAD, d_head)
    x2 = torch.randn(1, T, N_HEAD, d_head)
    x_batch = torch.cat([x1, x2], dim=0)
    
    y1 = rotary(x1)
    y2 = rotary(x2)
    y_batch = rotary(x_batch)
    
    assert torch.allclose(y_batch[0], y1[0], atol=ATOL, rtol=RTOL)
    assert torch.allclose(y_batch[1], y2[0], atol=ATOL, rtol=RTOL)
