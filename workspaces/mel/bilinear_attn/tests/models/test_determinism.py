"""Test 7: Determinism - same seed → same init → same logits."""
import pytest
import torch

from models import AttentionLM
from tests.models.conftest import B, T, V


def test_same_seed_same_init(test_config):
    """Test that same seed produces identical model initialization."""
    seed = test_config["seed"]
    
    torch.manual_seed(seed)
    model_a = AttentionLM.from_config(test_config)
    
    torch.manual_seed(seed)
    model_b = AttentionLM.from_config(test_config)
    
    for (name_a, param_a), (name_b, param_b) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        assert name_a == name_b, f"Parameter names should match: {name_a} vs {name_b}"
        assert torch.equal(param_a, param_b), \
            f"Parameter {name_a} should be identical with same seed"


def test_same_seed_same_logits(test_config, random_input_ids):
    """Test that same seed produces identical logits."""
    seed = test_config["seed"]
    
    torch.manual_seed(seed)
    model_a = AttentionLM.from_config(test_config)
    
    torch.manual_seed(seed)
    model_b = AttentionLM.from_config(test_config)
    
    torch.manual_seed(0)
    input_ids = random_input_ids
    
    logits_a = model_a(input_ids)
    logits_b = model_b(input_ids)
    
    assert torch.equal(logits_a, logits_b), \
        "Logits should be exactly equal with same seed"


def test_different_seed_different_init(test_config):
    """Test that different seeds produce different model initialization."""
    torch.manual_seed(42)
    model_a = AttentionLM.from_config(test_config)
    
    torch.manual_seed(123)
    model_b = AttentionLM.from_config(test_config)
    
    all_equal = True
    for (name_a, param_a), (name_b, param_b) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        if not torch.equal(param_a, param_b):
            all_equal = False
            break
    
    assert not all_equal, "Different seeds should produce different parameters"


def test_forward_deterministic(test_config, random_input_ids):
    """Test that forward pass is deterministic."""
    torch.manual_seed(test_config["seed"])
    model = AttentionLM.from_config(test_config)
    
    logits_1 = model(random_input_ids)
    logits_2 = model(random_input_ids)
    
    assert torch.equal(logits_1, logits_2), \
        "Multiple forward passes should produce identical results"
