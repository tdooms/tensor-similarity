"""Test 6: End-to-end backward pass - grads exist and are finite."""
import pytest
import torch
import torch.nn.functional as F

from models import AttentionLM
from train.losses import compute_loss
from tests.models.conftest import B, T, V


def test_gradients_exist_and_finite(test_config, random_input_ids):
    """Test that gradients exist and are finite after backward pass."""
    torch.manual_seed(test_config["seed"])
    model = AttentionLM.from_config(test_config)
    
    logits = model(random_input_ids)
    
    loss = compute_loss(logits, random_input_ids)
    
    loss.backward()
    
    assert model.embed.weight.grad is not None, "Embedding grad should exist"
    assert torch.isfinite(model.embed.weight.grad).all(), "Embedding grad should be finite"
    
    assert model.unembed.weight.grad is not None, "Unembed grad should exist"
    assert torch.isfinite(model.unembed.weight.grad).all(), "Unembed grad should be finite"
    
    for i, layer in enumerate(model.layers):
        assert layer.q.weight.grad is not None, f"Layer {i} Wq grad should exist"
        assert torch.isfinite(layer.q.weight.grad).all(), f"Layer {i} Wq grad should be finite"
        
        assert layer.k.weight.grad is not None, f"Layer {i} Wk grad should exist"
        assert torch.isfinite(layer.k.weight.grad).all(), f"Layer {i} Wk grad should be finite"
        
        assert layer.v.weight.grad is not None, f"Layer {i} Wv grad should exist"
        assert torch.isfinite(layer.v.weight.grad).all(), f"Layer {i} Wv grad should be finite"
        
        assert layer.o.weight.grad is not None, f"Layer {i} Wo grad should exist"
        assert torch.isfinite(layer.o.weight.grad).all(), f"Layer {i} Wo grad should be finite"
    
    assert torch.isfinite(loss), "Loss should be finite"


def test_loss_decreases_with_step(test_config, random_input_ids):
    """Test that loss can decrease with a gradient step."""
    torch.manual_seed(test_config["seed"])
    model = AttentionLM.from_config(test_config)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    logits = model(random_input_ids)
    loss_before = compute_loss(logits, random_input_ids)
    
    loss_before.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    logits = model(random_input_ids)
    loss_after = compute_loss(logits, random_input_ids)
    
    assert loss_after < loss_before, \
        f"Loss should decrease. Before: {loss_before.item()}, After: {loss_after.item()}"


def test_gradients_flow_through_all_layers(test_config, random_input_ids):
    """Test that gradients flow through all attention layers."""
    torch.manual_seed(test_config["seed"])
    model = AttentionLM.from_config(test_config)
    
    logits = model(random_input_ids)
    loss = compute_loss(logits, random_input_ids)
    loss.backward()
    
    for i, layer in enumerate(model.layers):
        grad_norm = layer.q.weight.grad.norm().item()
        assert grad_norm > 0, f"Layer {i} should have non-zero gradients"
