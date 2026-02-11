"""Test 2: Embeddings (not tied in this architecture)."""
import pytest
import torch

from models import AttentionLM


def test_separate_embed_unembed(test_config):
    """Test that embed and unembed are separate parameters."""
    torch.manual_seed(test_config["seed"])
    model = AttentionLM.from_config(test_config)
    
    assert hasattr(model, "embed"), "Model should have embed attribute"
    assert hasattr(model, "unembed"), "Model should have unembed attribute"
    
    assert model.embed.weight is not model.unembed.weight, \
        "embed and unembed weights should be separate (not tied)"
    
    assert model.embed.weight.shape[0] == test_config["model"]["vocab_size"]
    assert model.embed.weight.shape[1] == test_config["model"]["d_model"]
    
    assert model.unembed.weight.shape[0] == test_config["model"]["vocab_size"]
    assert model.unembed.weight.shape[1] == test_config["model"]["d_model"]


def test_embed_perturbation_affects_logits(test_config, random_input_ids):
    """Test that perturbing embed weights changes logits."""
    torch.manual_seed(test_config["seed"])
    model = AttentionLM.from_config(test_config)
    
    logits_before = model(random_input_ids).clone()
    
    # Perturb embedding for a token that's actually in the input
    token_to_perturb = random_input_ids[0, 0].item()
    with torch.no_grad():
        model.embed.weight[token_to_perturb, 0] += 1e-3
    
    logits_after = model(random_input_ids)
    
    assert not torch.allclose(logits_before, logits_after), \
        "Logits should change after perturbing embed weights"


def test_unembed_perturbation_affects_logits(test_config, random_input_ids):
    """Test that perturbing unembed weights changes logits."""
    torch.manual_seed(test_config["seed"])
    model = AttentionLM.from_config(test_config)
    
    logits_before = model(random_input_ids).clone()
    
    with torch.no_grad():
        model.unembed.weight[0, 0] += 1e-3
    
    logits_after = model(random_input_ids)
    
    assert not torch.allclose(logits_before, logits_after), \
        "Logits should change after perturbing unembed weights"
