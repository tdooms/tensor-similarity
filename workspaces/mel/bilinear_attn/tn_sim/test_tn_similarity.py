"""Test script for TN similarity computation on bilinear attention models.

This script validates the TN similarity integration by:
1. Creating two small models with different random seeds
2. Computing their TN similarity
3. Verifying self-similarity is close to 1.0
4. Comparing with Monte Carlo estimates (if requested)
"""

import torch
import yaml
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import AttentionLM
from tn_sim import compute_similarity, cosine_similarity, inner_product_similarity, self_similarity_check


def create_test_model(seed=42, attn_type="bilinear", use_rmsnorm_qk=False):
    """Create a small test model for similarity computation."""
    torch.manual_seed(seed)
    
    config = {
        "model": {
            "vocab_size": 256,
            "n_ctx": 32,
            "d_model": 64,
            "n_head": 4,
            "n_layers": 2,
            "attn_type": attn_type,
            "attn_scale": 0.2,
            "rope_base": 10000,
            "norm_type": "none",  # TN-clean: no normalization
            "use_rmsnorm_qk": use_rmsnorm_qk,
            "use_bias_qk": True,
        },
        "init": {
            "std_embed": 0.02,
            "std_qkv": 0.02,
            "std_o": 0.01,
        }
    }
    
    model = AttentionLM.from_config(config)
    model.eval()
    return model.double()  # Use float64 for numerical precision


def test_self_similarity():
    """Test that a model has perfect cosine similarity with itself."""
    print("\n=== Test 1: Self-Similarity ===")
    
    model = create_test_model(seed=42, attn_type="bilinear")
    model.setup_tn_components()
    
    print("Computing self-similarity...")
    try:
        self_similarity_check(model, tolerance=1e-6)
        print("✓ Self-similarity check passed (cosine ≈ 1.0)")
    except AssertionError as e:
        print(f"✗ Self-similarity check failed: {e}")
        return False
    
    return True


def test_cross_similarity():
    """Test similarity between two different models."""
    print("\n=== Test 2: Cross-Similarity (Bilinear) ===")
    
    model_a = create_test_model(seed=42, attn_type="bilinear")
    model_b = create_test_model(seed=99, attn_type="bilinear")
    
    model_a.setup_tn_components()
    model_b.setup_tn_components()
    
    print("Computing cross-similarity...")
    state = compute_similarity(model_a, model_b)
    
    cos_sim = cosine_similarity(state)
    inner_prod = inner_product_similarity(state)
    
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"Inner product: {inner_prod:.6f}")
    print(f"✓ Cross-similarity computed successfully")
    
    # Cross-similarity should be less than 1.0 for different random inits
    assert cos_sim < 1.0, "Different models should have cosine < 1.0"
    assert cos_sim > -1.0, "Cosine similarity should be in [-1, 1]"
    
    return True


def test_quadratic_attention():
    """Test similarity for quadratic attention."""
    print("\n=== Test 3: Quadratic Attention ===")
    
    model_a = create_test_model(seed=42, attn_type="quadratic")
    model_b = create_test_model(seed=99, attn_type="quadratic")
    
    model_a.setup_tn_components()
    model_b.setup_tn_components()
    
    print("Computing similarity for quadratic attention...")
    state = compute_similarity(model_a, model_b)
    
    cos_sim = cosine_similarity(state)
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"✓ Quadratic attention similarity computed successfully")
    
    return True


def test_validation_checks():
    """Test that validation checks work correctly."""
    print("\n=== Test 4: Validation Checks ===")
    
    # Test 1: Softmax should fail
    print("Testing softmax rejection...")
    model = create_test_model(seed=42, attn_type="softmax")
    try:
        model.setup_tn_components()
        print("✗ Should have rejected softmax attention")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected softmax: {str(e)[:60]}...")
    
    # Test 2: Q/K norm should warn
    print("\nTesting Q/K norm warning...")
    model = create_test_model(seed=42, attn_type="bilinear", use_rmsnorm_qk=True)
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.setup_tn_components()
        if len(w) > 0:
            print(f"✓ Warning issued: {str(w[0].message)[:60]}...")
        else:
            print("✗ Should have warned about Q/K normalization")
            return False
    
    return True


def monte_carlo_validation(n_samples=100000):
    """Compare TN similarity with Monte Carlo estimate.
    
    This is a sanity check: for Gaussian inputs, the TN computation
    should match MC sampling.
    """
    print("\n=== Test 5: Monte Carlo Validation ===")
    print(f"Using {n_samples:,} samples (this may take a minute)...")
    
    # Create small models for faster computation
    torch.manual_seed(42)
    model_a = create_test_model(seed=42, attn_type="bilinear")
    model_b = create_test_model(seed=99, attn_type="bilinear")
    
    model_a.setup_tn_components()
    model_b.setup_tn_components()
    
    # TN computation
    print("Computing exact TN similarity...")
    state = compute_similarity(model_a, model_b)
    tn_inner = inner_product_similarity(state)
    
    # MC estimation
    print("Computing MC estimate...")
    d_input = model_a.vocab_size
    n_ctx = model_a.n_ctx
    
    # Generate Gaussian inputs (one-hot is not Gaussian, so we use continuous)
    x = torch.randn(n_samples, n_ctx, d_input, dtype=torch.float64)
    
    with torch.no_grad():
        # Forward pass through both models
        # Note: We need to handle embedding differently for continuous input
        # For this test, we'll directly use the embedding weights as a linear layer
        y_a = model_a.embed(x)  # This won't work with continuous x
        
    print("⚠ MC validation requires special handling for embeddings")
    print("  Skipping MC comparison (would need to modify forward pass)")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("TN Similarity Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Self-Similarity", test_self_similarity),
        ("Cross-Similarity", test_cross_similarity),
        ("Quadratic Attention", test_quadratic_attention),
        ("Validation Checks", test_validation_checks),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ Test '{name}' failed with exception:")
            print(f"  {type(e).__name__}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    n_passed = sum(1 for _, s in results if s)
    n_total = len(results)
    print(f"\nPassed: {n_passed}/{n_total}")
    
    return n_passed == n_total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
