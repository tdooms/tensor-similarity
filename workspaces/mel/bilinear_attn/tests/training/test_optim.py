"""Tests for optimizer and scheduler."""
import pytest
import torch
import torch.nn as nn

from train.optim import create_optimizer, create_scheduler


class DummyModel(nn.Module):
    """Simple model for testing optimizer."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.norm = nn.LayerNorm(10)
    
    def forward(self, x):
        return self.norm(self.linear(x))


def test_create_optimizer_returns_adamw():
    """Test that create_optimizer returns AdamW."""
    model = DummyModel()
    optimizer = create_optimizer(model, lr=1e-3, use_muon=False)
    
    assert isinstance(optimizer, torch.optim.AdamW)


def test_create_optimizer_param_groups():
    """Test that optimizer has correct param groups (decay vs no decay)."""
    model = DummyModel()
    optimizer = create_optimizer(model, lr=1e-3, weight_decay=0.1, use_muon=False)
    
    # Should have 2 param groups: decay and no_decay
    assert len(optimizer.param_groups) == 2
    
    # One group should have weight_decay=0.1, other should have 0.0
    weight_decays = [g["weight_decay"] for g in optimizer.param_groups]
    assert 0.1 in weight_decays
    assert 0.0 in weight_decays


def test_create_optimizer_lr():
    """Test that optimizer uses specified learning rate."""
    model = DummyModel()
    lr = 5e-4
    optimizer = create_optimizer(model, lr=lr, use_muon=False)
    
    for group in optimizer.param_groups:
        assert group["lr"] == lr


def test_create_scheduler_warmup():
    """Test that scheduler implements warmup."""
    model = DummyModel()
    optimizer = create_optimizer(model, lr=1e-3, use_muon=False)
    scheduler = create_scheduler(optimizer, warmup_steps=10, max_steps=100)
    
    # At step 0, lr should be 0 (or very small)
    initial_lr = scheduler.get_last_lr()[0]
    
    # Step through warmup
    for _ in range(5):
        optimizer.step()
        scheduler.step()
    
    mid_warmup_lr = scheduler.get_last_lr()[0]
    
    # LR should increase during warmup
    assert mid_warmup_lr > initial_lr


def test_create_scheduler_decay():
    """Test that scheduler decays after warmup."""
    model = DummyModel()
    optimizer = create_optimizer(model, lr=1e-3, use_muon=False)
    scheduler = create_scheduler(optimizer, warmup_steps=10, max_steps=100)
    
    # Complete warmup
    for _ in range(10):
        optimizer.step()
        scheduler.step()
    
    post_warmup_lr = scheduler.get_last_lr()[0]
    
    # Step through some decay
    for _ in range(50):
        optimizer.step()
        scheduler.step()
    
    decayed_lr = scheduler.get_last_lr()[0]
    
    # LR should decrease after warmup
    assert decayed_lr < post_warmup_lr


def test_scheduler_reaches_floor():
    """Test that scheduler reaches lr_decay_frac at max_steps."""
    model = DummyModel()
    optimizer = create_optimizer(model, lr=1e-3, use_muon=False)
    scheduler = create_scheduler(optimizer, warmup_steps=10, max_steps=100, lr_decay_frac=0.1)
    
    # Step to max_steps
    for _ in range(100):
        optimizer.step()
        scheduler.step()
    
    final_lr = scheduler.get_last_lr()[0]
    
    # LR should be at lr_decay_frac * lr = 0.1 * 1e-3 = 1e-4
    assert abs(final_lr - 1e-4) < 1e-7
