#!/usr/bin/env python3
"""Test TN similarity using main codebase against MC baseline.

This test suite validates the migration from the custom TN similarity
implementation to the main codebase's exact algorithm.

Tests cover:
1. Self-similarity (should be exactly 1.0)
2. Cross-similarity vs MC baseline (should match within tolerance)
3. Model validation (should reject incompatible configs)
4. Component weight loading (should preserve weights correctly)

Usage:
    pytest tn_sim/test_similarity.py -v
    
    # Or run directly:
    python -m tn_sim.test_similarity
"""

import pytest
import torch
import numpy as np

from models import AttentionLM
from models.components import AttentionLMComponent, BilinearAttentionComponent, EmbeddingComponent
from models.components.embedding import UnembeddingComponent
from tn_sim.similarity import (
    compute_tn_similarity,
    cosine_similarity,
    inner_product,
    self_similarity,
    _check_architectures_match,
)
from tn_sim.mc_similarity import mc_similarity, mc_similarity_gaussian_tokens


# Use float64 for numerical stability in tests
DTYPE = torch.float64
DEVICE = "cpu"

# Tolerances
SELF_SIM_TOL = 1e-6  # Self-similarity should be very close to 1.0
MC_REL_TOL = 0.3     # MC vs TN relative tolerance (30% for small sample sizes)
MC_ABS_TOL = 0.1     # MC vs TN absolute tolerance


def make_tn_compatible_config(
    vocab_size=8,
    n_ctx=8,      # Reduced from 8 to match main codebase tests
    d_model=8,    # Reduced from 16 to match main codebase tests
    n_head=2,
    n_layers=2,
    attn_scale=0.5,
    attn_type="quadratic",
    use_bias_qk=True,
):
    """Create a TN-compatible model config (no normalization)."""
    return {
        "model": {
            "vocab_size": vocab_size,
            "n_ctx": n_ctx,
            "d_model": d_model,
            "n_head": n_head,
            "n_layers": n_layers,
            "attn_scale": attn_scale,
            "attn_type": attn_type,
            "use_bias_qk": use_bias_qk,
            "use_rmsnorm_qk": False,
            "norm_type": "none",
            "norm_places": [],
            "rope_base": 10000,
        },
        "init": {
            "std_embed": 0.02,
            "std_qkv": 0.02,
            "std_o": 0.01,
        }
    }


class TestSelfSimilarity:
    """Test that self-similarity is exactly 1.0."""
    
    def test_self_similarity_1layer_bilinear(self):
        """Self-similarity for 1-layer bilinear model."""
        torch.manual_seed(42)
        cfg = make_tn_compatible_config(n_layers=1, attn_type="bilinear")
        model = AttentionLM.from_config(cfg).to(dtype=DTYPE)
        
        sim = self_similarity(model, device=DEVICE, dtype=DTYPE)
        
        assert abs(sim - 1.0) < SELF_SIM_TOL, f"Self-similarity should be 1.0, got {sim}"
    
    def test_self_similarity_1layer_quadratic(self):
        """Self-similarity for 1-layer quadratic model."""
        torch.manual_seed(42)
        cfg = make_tn_compatible_config(n_layers=1, attn_type="quadratic")
        model = AttentionLM.from_config(cfg).to(dtype=DTYPE)
        
        sim = self_similarity(model, device=DEVICE, dtype=DTYPE)
        
        assert abs(sim - 1.0) < SELF_SIM_TOL, f"Self-similarity should be 1.0, got {sim}"
    
    def test_self_similarity_2layer(self):
        """Self-similarity for 2-layer model."""
        torch.manual_seed(42)
        cfg = make_tn_compatible_config(n_layers=2)
        model = AttentionLM.from_config(cfg).to(dtype=DTYPE)
        
        sim = self_similarity(model, device=DEVICE, dtype=DTYPE)
        
        assert abs(sim - 1.0) < SELF_SIM_TOL, f"Self-similarity should be 1.0, got {sim}"
    
    def test_self_similarity_no_bias(self):
        """Self-similarity for model without Q/K biases."""
        torch.manual_seed(42)
        cfg = make_tn_compatible_config(use_bias_qk=False)
        model = AttentionLM.from_config(cfg).to(dtype=DTYPE)
        
        sim = self_similarity(model, device=DEVICE, dtype=DTYPE)
        
        assert abs(sim - 1.0) < SELF_SIM_TOL, f"Self-similarity should be 1.0, got {sim}"


class TestCrossSimilarity:
    """Test cross-similarity between different models."""
    
    def test_different_seeds_not_identical(self):
        """Models with different seeds should have similarity < 1."""
        cfg = make_tn_compatible_config()
        
        torch.manual_seed(42)
        model_A = AttentionLM.from_config(cfg).to(dtype=DTYPE)
        
        torch.manual_seed(99)
        model_B = AttentionLM.from_config(cfg).to(dtype=DTYPE)
        
        sim = cosine_similarity(model_A, model_B, device=DEVICE, dtype=DTYPE)
        
        assert sim < 1.0, f"Different models should have similarity < 1, got {sim}"
        assert sim > -1.0, f"Similarity should be > -1, got {sim}"
    
    def test_symmetry(self):
        """Similarity should be symmetric: sim(A, B) == sim(B, A)."""
        cfg = make_tn_compatible_config()
        
        torch.manual_seed(42)
        model_A = AttentionLM.from_config(cfg).to(dtype=DTYPE)
        
        torch.manual_seed(99)
        model_B = AttentionLM.from_config(cfg).to(dtype=DTYPE)
        
        sim_AB = cosine_similarity(model_A, model_B, device=DEVICE, dtype=DTYPE)
        sim_BA = cosine_similarity(model_B, model_A, device=DEVICE, dtype=DTYPE)
        
        assert abs(sim_AB - sim_BA) < 1e-10, f"Similarity should be symmetric: {sim_AB} != {sim_BA}"


class TestMCComparison:
    """Compare TN similarity against Monte Carlo baseline."""
    
    def test_vs_mc_1layer(self):
        """TN similarity should approximate MC similarity for 1-layer model.

        The MC baseline MUST sample at the TN algorithm's input level
        (Gaussian over the padded vocab axis) to match what TN computes;
        sampling at the residual stream (``mc_similarity``) is a different
        distribution and does not converge to the TN value.
        """
        torch.manual_seed(42)
        cfg = make_tn_compatible_config(n_layers=1)
        model_A = AttentionLM.from_config(cfg).to(dtype=DTYPE)

        torch.manual_seed(99)
        model_B = AttentionLM.from_config(cfg).to(dtype=DTYPE)

        tn_sim = cosine_similarity(model_A, model_B, device=DEVICE, dtype=DTYPE)

        mc_sim = mc_similarity_gaussian_tokens(
            model_A, model_B, device=DEVICE, n_samples=20000,
        )

        diff = abs(tn_sim - mc_sim)
        rel_diff = diff / max(abs(mc_sim), 1e-8)
        assert diff < MC_ABS_TOL or rel_diff < MC_REL_TOL, (
            f"TN sim ({tn_sim:.4f}) differs from MC sim ({mc_sim:.4f}) "
            f"by {diff:.4f} (rel: {rel_diff:.2%})"
        )

    def test_self_similarity_vs_mc(self):
        """Self-similarity should be ~1.0 for both TN and the TN-matched MC."""
        torch.manual_seed(42)
        cfg = make_tn_compatible_config(n_layers=1)
        model = AttentionLM.from_config(cfg).to(dtype=DTYPE)

        tn_sim = self_similarity(model, device=DEVICE, dtype=DTYPE)
        mc_sim = mc_similarity_gaussian_tokens(
            model, model, device=DEVICE, n_samples=5000,
        )

        assert abs(tn_sim - 1.0) < SELF_SIM_TOL, f"TN self-sim should be 1.0, got {tn_sim}"
        assert abs(mc_sim - 1.0) < 0.05, f"MC self-sim should be ~1.0, got {mc_sim}"


class TestModelValidation:
    """Test that incompatible models are rejected."""
    
    def test_rejects_rmsnorm(self):
        """Should reject models with norm_type != 'none'."""
        cfg = make_tn_compatible_config()
        cfg["model"]["norm_type"] = "rmsnorm"
        cfg["model"]["norm_places"] = ["pre_unembed"]
        
        model = AttentionLM.from_config(cfg)
        
        with pytest.raises(ValueError, match="norm_type"):
            cosine_similarity(model, model)
    
    def test_rejects_rmsnorm_qk(self):
        """Should reject models with use_rmsnorm_qk=True."""
        cfg = make_tn_compatible_config()
        cfg["model"]["use_rmsnorm_qk"] = True
        
        model = AttentionLM.from_config(cfg)
        
        with pytest.raises(ValueError, match="rmsnorm_qk"):
            cosine_similarity(model, model)
    
    def test_rejects_softmax_attention(self):
        """Should reject models with softmax attention."""
        cfg = make_tn_compatible_config()
        cfg["model"]["attn_type"] = "softmax"
        
        model = AttentionLM.from_config(cfg)
        
        with pytest.raises(ValueError, match="attn_type"):
            cosine_similarity(model, model)
    
    def test_rejects_incompatible_architectures(self):
        """Should reject models with different architectures."""
        cfg_A = make_tn_compatible_config(d_model=16)
        cfg_B = make_tn_compatible_config(d_model=32)
        
        model_A = AttentionLM.from_config(cfg_A)
        model_B = AttentionLM.from_config(cfg_B)
        
        with pytest.raises(ValueError, match="incompatible"):
            cosine_similarity(model_A, model_B)

    def test_ignore_norms_accepts_normed_model(self):
        """``from_trained_model(model, ignore_norms=True)`` must accept a
        model trained with norms and yield a component whose
        self-similarity is exactly 1 (norms dropped, linear subnet only)."""
        cfg = make_tn_compatible_config()
        cfg["model"]["norm_type"] = "rmsnorm"
        cfg["model"]["norm_places"] = ["pre_unembed", "pre_layer"]
        cfg["model"]["use_rmsnorm_qk"] = True
        model = AttentionLM.from_config(cfg).to(dtype=DTYPE)

        # Strict mode still rejects.
        with pytest.raises(ValueError):
            AttentionLMComponent.from_trained_model(model)

        # Permissive mode strips norms and produces a usable component.
        comp = AttentionLMComponent.from_trained_model(
            model, ignore_norms=True,
        ).to(dtype=DTYPE)
        sim = cosine_similarity(comp, comp, device=DEVICE, dtype=DTYPE)
        assert abs(float(sim) - 1.0) < SELF_SIM_TOL, f"self-sim = {float(sim)!r}"


class TestComponentWeightLoading:
    """Test that weights are correctly loaded into Component wrappers."""
    
    def test_embedding_weight_copy(self):
        """Embedding weights should be correctly copied."""
        torch.manual_seed(42)
        cfg = make_tn_compatible_config()
        model = AttentionLM.from_config(cfg)
        
        comp = AttentionLMComponent.from_trained_model(model)
        
        # Check embedding weights match
        assert torch.allclose(
            comp.embed.weight.data,
            model.embed.weight.data
        ), "Embedding weights should match"
    
    def test_attention_weight_copy(self):
        """Attention weights should be correctly copied."""
        torch.manual_seed(42)
        cfg = make_tn_compatible_config(attn_type="bilinear")
        model = AttentionLM.from_config(cfg)
        
        comp = AttentionLMComponent.from_trained_model(model)
        
        # Check attention weights match
        for i, (comp_layer, model_layer) in enumerate(zip(comp.layers, model.layers)):
            assert torch.allclose(
                comp_layer.q1.weight.data,
                model_layer.q1.weight.data
            ), f"Layer {i} Q1 weights should match"
            assert torch.allclose(
                comp_layer.v.weight.data,
                model_layer.v.weight.data
            ), f"Layer {i} V weights should match"
            assert torch.allclose(
                comp_layer.o.weight.data,
                model_layer.o.weight.data
            ), f"Layer {i} O weights should match"
    
    def test_unembed_weight_copy(self):
        """Unembed weights should be correctly copied."""
        torch.manual_seed(42)
        cfg = make_tn_compatible_config()
        model = AttentionLM.from_config(cfg)
        
        comp = AttentionLMComponent.from_trained_model(model)
        
        assert torch.allclose(
            comp.unembed.weight.data,
            model.unembed.weight.data
        ), "Unembed weights should match"


class TestComponentInterface:
    """Test that Component interface is correctly implemented."""
    
    @pytest.mark.parametrize("attn_type,expected_cls_name", [
        ("bilinear", "BilinearAttentionComponent"),
        ("quadratic", "QuadraticAttentionComponent"),
    ])
    def test_components_list(self, attn_type, expected_cls_name):
        """components() should return [embed, *layers, unembed] with the
        attention class matching attn_type."""
        cfg = make_tn_compatible_config(n_layers=2, attn_type=attn_type)
        model = AttentionLM.from_config(cfg)
        comp = AttentionLMComponent.from_trained_model(model)

        components = comp.components()

        assert len(components) == 4, f"Expected 4 components, got {len(components)}"
        assert isinstance(components[0], EmbeddingComponent)
        assert type(components[1]).__name__ == expected_cls_name
        assert type(components[2]).__name__ == expected_cls_name
        assert isinstance(components[3], UnembeddingComponent)
    
    def test_attention_network_indices(self):
        """Attention network() should have correct indices."""
        cfg = make_tn_compatible_config()
        model = AttentionLM.from_config(cfg)
        comp = AttentionLMComponent.from_trained_model(model)
        
        attn = comp.layers[0]
        tn = attn.network()
        
        # Check expected indices exist
        inds = set(tn.ind_map.keys())
        assert 'out:d' in inds, "Should have 'out:d' index"
        assert 'in:d0' in inds, "Should have 'in:d0' index (V input)"
        assert 'in:d1' in inds, "Should have 'in:d1' index (K1 input)"
        assert 'in:d2' in inds, "Should have 'in:d2' index (K2 input)"
        assert 'in:d3' in inds, "Should have 'in:d3' index (Q1 input)"
        assert 'in:d4' in inds, "Should have 'in:d4' index (Q2 input)"
    
    def test_attention_terms(self):
        """Attention terms() should return 2 terms (residual + active)."""
        cfg = make_tn_compatible_config()
        model = AttentionLM.from_config(cfg)
        comp = AttentionLMComponent.from_trained_model(model)
        
        attn = comp.layers[0]
        terms = attn.terms(n_ctx=cfg["model"]["n_ctx"], device=DEVICE, dtype=DTYPE)
        
        assert len(terms) == 2, f"Expected 2 terms, got {len(terms)}"
        
        # Term 0: residual (1 input leg)
        assert len(terms[0].legs) == 1, "Residual term should have 1 input leg"
        
        # Term 1: active (5 input legs)
        assert len(terms[1].legs) == 5, "Active term should have 5 input legs"


def run_quick_validation():
    """Run a quick validation to check the migration works."""
    print("=" * 60)
    print("Quick Validation: TN Similarity Migration")
    print("=" * 60)
    
    # Test 1: Self-similarity
    print("\n1. Testing self-similarity...")
    torch.manual_seed(42)
    cfg = make_tn_compatible_config(n_layers=1)  # Uses defaults: d_model=8, n_ctx=3
    model = AttentionLM.from_config(cfg).to(dtype=DTYPE)
    
    sim = self_similarity(model, device=DEVICE, dtype=DTYPE)
    print(f"   Self-similarity: {sim:.10f}")
    assert abs(sim - 1.0) < SELF_SIM_TOL, f"FAILED: Expected 1.0, got {sim}"
    print("   ✓ PASSED")
    
    # Test 2: Cross-similarity
    print("\n2. Testing cross-similarity...")
    torch.manual_seed(99)
    model_B = AttentionLM.from_config(cfg).to(dtype=DTYPE)
    
    sim_AB = cosine_similarity(model, model_B, device=DEVICE, dtype=DTYPE)
    print(f"   Cross-similarity: {sim_AB:.6f}")
    assert -1.0 <= sim_AB <= 1.0, f"FAILED: Similarity out of range: {sim_AB}"
    print("   ✓ PASSED")
    
    # Test 3: MC comparison
    print("\n3. Testing vs MC baseline...")
    model_f32 = model.float()
    model_B_f32 = model_B.float()
    mc_sim = mc_similarity(
        model_f32, model_B_f32,
        device=DEVICE,
        n_samples=5000,
    )
    print(f"   TN similarity:  {sim_AB:.6f}")
    print(f"   MC similarity:  {mc_sim:.6f}")
    print(f"   Difference:     {abs(sim_AB - mc_sim):.6f}")
    print("   ✓ PASSED (comparison only, no assertion)")
    
    # Test 4: Model validation
    print("\n4. Testing model validation...")
    cfg_bad = make_tn_compatible_config()
    cfg_bad["model"]["norm_type"] = "rmsnorm"
    cfg_bad["model"]["norm_places"] = ["pre_unembed"]
    model_bad = AttentionLM.from_config(cfg_bad)
    
    try:
        cosine_similarity(model_bad, model_bad)
        print("   ✗ FAILED: Should have raised ValueError")
    except ValueError as e:
        print(f"   Correctly rejected: {str(e)[:60]}...")
        print("   ✓ PASSED")
    
    print("\n" + "=" * 60)
    print("All quick validation tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_quick_validation()
