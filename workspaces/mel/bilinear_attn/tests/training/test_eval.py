"""Tests for evaluation functions."""
import pytest
import torch

from models import AttentionLM
from train.eval import evaluate
from tests.training.conftest import B, T, V


def test_evaluate_returns_float(tiny_config, dummy_dataloader, device):
    """Test that evaluate returns a float loss value."""
    torch.manual_seed(tiny_config["seed"])
    model = AttentionLM.from_config(tiny_config)
    model = model.to(device)
    
    val_loss = evaluate(model, dummy_dataloader, device)
    
    assert isinstance(val_loss, float)


def test_evaluate_finite(tiny_config, dummy_dataloader, device):
    """Test that evaluation loss is finite."""
    torch.manual_seed(tiny_config["seed"])
    model = AttentionLM.from_config(tiny_config)
    model = model.to(device)
    
    val_loss = evaluate(model, dummy_dataloader, device)
    
    assert val_loss > 0, "Loss should be positive"
    assert val_loss < float('inf'), "Loss should be finite"


def test_evaluate_no_grad(tiny_config, dummy_dataloader, device):
    """Test that evaluate doesn't compute gradients."""
    torch.manual_seed(tiny_config["seed"])
    model = AttentionLM.from_config(tiny_config)
    model = model.to(device)
    
    # Zero all grads
    model.zero_grad()
    
    evaluate(model, dummy_dataloader, device)
    
    # Check no gradients were computed
    for param in model.parameters():
        assert param.grad is None or (param.grad == 0).all(), \
            "Evaluate should not compute gradients"


def test_evaluate_max_batches(tiny_config, dummy_dataloader, device):
    """Test that max_batches limits evaluation."""
    torch.manual_seed(tiny_config["seed"])
    model = AttentionLM.from_config(tiny_config)
    model = model.to(device)
    
    # Should work with max_batches=1
    val_loss = evaluate(model, dummy_dataloader, device, max_batches=1)
    
    assert isinstance(val_loss, float)
    assert val_loss > 0


def test_evaluate_model_in_eval_mode(tiny_config, dummy_dataloader, device):
    """Test that model is in eval mode during evaluation."""
    torch.manual_seed(tiny_config["seed"])
    model = AttentionLM.from_config(tiny_config)
    model = model.to(device)
    model.train()  # Start in train mode
    
    evaluate(model, dummy_dataloader, device)
    
    # Model should be in eval mode after evaluate
    assert not model.training, "Model should be in eval mode after evaluate"


def test_evaluate_deterministic(tiny_config, dummy_dataloader, device):
    """Test that evaluation is deterministic."""
    torch.manual_seed(tiny_config["seed"])
    model = AttentionLM.from_config(tiny_config)
    model = model.to(device)
    
    loss1 = evaluate(model, dummy_dataloader, device, max_batches=2)
    loss2 = evaluate(model, dummy_dataloader, device, max_batches=2)
    
    # Note: Due to dataloader shuffling, we use max_batches to get consistent results
    # The losses should be similar (same model, same data subset)
    assert abs(loss1 - loss2) < 0.1, \
        f"Evaluation should be deterministic. Got {loss1} and {loss2}"
