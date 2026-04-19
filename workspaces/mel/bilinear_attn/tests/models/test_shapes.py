"""Test 1: Forward shapes."""
import pytest
import torch

from models import AttentionLM
from tests.models.conftest import B, T, V, D_MODEL, N_HEAD, N_LAYERS, N_CTX


def test_forward_shapes(test_config, random_input_ids):
    """Test that forward pass produces correct output shapes."""
    torch.manual_seed(test_config["seed"])
    model = AttentionLM.from_config(test_config)
    
    logits = model(random_input_ids)
    
    assert logits.shape == (B, T, V), f"Expected {(B, T, V)}, got {logits.shape}"
    assert logits.dtype == torch.float32, f"Expected float32, got {logits.dtype}"
    assert logits.device.type == "cpu", f"Expected cpu, got {logits.device}"


def test_forward_with_debug(test_config, random_input_ids):
    """Test that debug output contains expected keys."""
    torch.manual_seed(test_config["seed"])
    model = AttentionLM.from_config(test_config)
    
    logits, debug_list = model(random_input_ids, return_debug=True)
    
    assert len(debug_list) == N_LAYERS
    
    for debug in debug_list:
        assert "q" in debug
        assert "k" in debug
        assert "v" in debug
        assert "scores" in debug
        assert "pattern" in debug
        assert "z" in debug
        
        d_head = D_MODEL // N_HEAD
        assert debug["q"].shape == (B, T, N_HEAD, d_head)
        assert debug["k"].shape == (B, T, N_HEAD, d_head)
        assert debug["v"].shape == (B, T, N_HEAD, d_head)
        assert debug["scores"].shape == (B, N_HEAD, T, T)
        assert debug["pattern"].shape == (B, N_HEAD, T, T)
        assert debug["z"].shape == (B, T, N_HEAD, d_head)


def test_variable_sequence_length(test_config):
    """Test that model handles different sequence lengths."""
    torch.manual_seed(test_config["seed"])
    model = AttentionLM.from_config(test_config)
    
    for seq_len in [1, 4, 8, N_CTX]:
        input_ids = torch.randint(0, V, (B, seq_len))
        logits = model(input_ids)
        assert logits.shape == (B, seq_len, V)
