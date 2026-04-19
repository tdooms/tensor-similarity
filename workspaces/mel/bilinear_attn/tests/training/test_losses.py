"""Tests for loss functions."""
import pytest
import torch

from train.losses import compute_loss, next_token_ce, per_position_ce
from tests.training.conftest import B, T, V


def test_next_token_ce_shape():
    """Test that next_token_ce returns scalar loss."""
    logits = torch.randn(B, T, V)
    input_ids = torch.randint(0, V, (B, T))
    
    loss = next_token_ce(logits, input_ids)
    
    assert loss.shape == (), "Loss should be scalar"
    assert loss.dtype == torch.float32


def test_next_token_ce_finite():
    """Test that loss is finite for valid inputs."""
    logits = torch.randn(B, T, V)
    input_ids = torch.randint(0, V, (B, T))
    
    loss = next_token_ce(logits, input_ids)
    
    assert torch.isfinite(loss), "Loss should be finite"


def test_next_token_ce_positive():
    """Test that cross entropy loss is positive."""
    logits = torch.randn(B, T, V)
    input_ids = torch.randint(0, V, (B, T))
    
    loss = next_token_ce(logits, input_ids)
    
    assert loss > 0, "Cross entropy loss should be positive"


def test_next_token_ce_shift_correct():
    """Test that loss uses correct shift (predicting next token)."""
    logits = torch.zeros(B, T, V)
    input_ids = torch.zeros(B, T, dtype=torch.long)
    
    # Set logits to strongly predict token 0 at all positions
    logits[:, :, 0] = 10.0
    
    # Targets are all 0s, so shifted targets are input_ids[:, 1:] = all 0s
    # Logits used are logits[:, :-1] which predict 0 strongly
    loss = next_token_ce(logits, input_ids)
    
    # Loss should be very low since predictions match targets
    assert loss < 0.1, f"Loss should be low when predictions match, got {loss}"


def test_per_position_ce_shape():
    """Test per-position CE returns correct shape."""
    logits = torch.randn(B, T, V)
    input_ids = torch.randint(0, V, (B, T))
    
    loss = per_position_ce(logits, input_ids)
    
    assert loss.shape == (T - 1,), f"Expected shape ({T-1},), got {loss.shape}"


def test_compute_loss_dispatch():
    """Test that compute_loss dispatches correctly."""
    logits = torch.randn(B, T, V)
    input_ids = torch.randint(0, V, (B, T))
    
    loss = compute_loss(logits, input_ids, loss_type="next_token_ce")
    
    assert torch.isfinite(loss)


def test_compute_loss_invalid_type():
    """Test that invalid loss type raises error."""
    logits = torch.randn(B, T, V)
    input_ids = torch.randint(0, V, (B, T))
    
    with pytest.raises(ValueError):
        compute_loss(logits, input_ids, loss_type="invalid_loss")


def test_label_smoothing():
    """Test that label smoothing affects loss."""
    logits = torch.randn(B, T, V)
    input_ids = torch.randint(0, V, (B, T))
    
    loss_no_smooth = next_token_ce(logits, input_ids, label_smoothing=0.0)
    loss_smooth = next_token_ce(logits, input_ids, label_smoothing=0.1)
    
    # Label smoothing should change the loss value
    assert not torch.isclose(loss_no_smooth, loss_smooth), \
        "Label smoothing should affect loss value"
