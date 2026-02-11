#!/usr/bin/env python3
"""Quick standalone tests to verify core functionality."""
import sys
import torch

print("Testing imports...")
try:
    from models import AttentionLM, QuadraticAttention, Rotary
    from models.attention_kernels.softmax import SoftmaxAttention
    from train.losses import compute_loss, next_token_ce
    from train.optim import create_optimizer, create_scheduler
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

B, T, V = 2, 6, 97
D_MODEL, N_HEAD, N_LAYERS, N_CTX = 32, 4, 2, 16
ATOL, RTOL = 1e-6, 1e-5

test_config = {
    "model": {
        "vocab_size": V,
        "n_ctx": N_CTX,
        "d_model": D_MODEL,
        "n_head": N_HEAD,
        "n_layers": N_LAYERS,
        "attn_scale": 0.2,
        "rope_base": 10000,
        "use_rmsnorm_qk": False,
        "use_bias_qkv": True,
        "use_bias_o": True,
    },
    "init": {"std_embed": 0.02, "std_qkv": 0.02, "std_o": 0.01},
}

def test_forward_shapes():
    print("\nTest 1: Forward shapes...")
    torch.manual_seed(42)
    model = AttentionLM.from_config(test_config)
    input_ids = torch.randint(0, V, (B, T))
    logits = model(input_ids)
    
    assert logits.shape == (B, T, V), f"Expected {(B, T, V)}, got {logits.shape}"
    assert logits.dtype == torch.float32
    print("✓ Forward shapes correct")

def test_rope_preserves_norm():
    print("\nTest 2: RoPE preserves norm...")
    d_head = D_MODEL // N_HEAD
    rotary = Rotary(dim=d_head, n_ctx=32)
    x = torch.randn(B, T, N_HEAD, d_head)
    y = rotary(x)
    
    x_norms = x.norm(dim=-1)
    y_norms = y.norm(dim=-1)
    assert torch.allclose(x_norms, y_norms, atol=ATOL, rtol=RTOL)
    print("✓ RoPE preserves L2 norm")

def test_causal_masking():
    print("\nTest 3: Causal masking...")
    attn = QuadraticAttention(d_model=D_MODEL, n_head=N_HEAD, n_ctx=N_CTX, scale=0.2)
    x = torch.randn(B, T, D_MODEL)
    _, debug = attn(x, return_debug=True)
    pattern = debug["pattern"]
    
    for t in range(T):
        for s in range(t + 1, T):
            assert (pattern[:, :, t, s] == 0).all(), f"Future leakage at ({t}, {s})"
    print("✓ Causal masking correct (no future leakage)")

def test_quadratic_attention_math():
    print("\nTest 4: Quadratic attention math...")
    d_head = D_MODEL // N_HEAD
    attn = QuadraticAttention(d_model=D_MODEL, n_head=N_HEAD, n_ctx=N_CTX, scale=0.2)
    
    torch.manual_seed(42)
    x = torch.randn(B, T, D_MODEL)
    _, debug = attn(x, return_debug=True)
    
    q, k, v = debug["q"], debug["k"], debug["v"]
    pattern, z = debug["pattern"], debug["z"]
    
    # Naive reference
    pattern_ref = torch.zeros(B, N_HEAD, T, T)
    z_ref = torch.zeros_like(v)
    
    for b in range(B):
        for h in range(N_HEAD):
            for t in range(T):
                for s in range(T):
                    score = (q[b, t, h] * k[b, s, h]).sum()
                    if s <= t:
                        pattern_ref[b, h, t, s] = (score / d_head) ** 2
            for t in range(T):
                for i in range(d_head):
                    z_ref[b, t, h, i] = (pattern_ref[b, h, t, :] * v[b, :, h, i]).sum()
    
    assert torch.allclose(pattern, pattern_ref, atol=ATOL, rtol=RTOL), \
        f"Pattern diff: {(pattern - pattern_ref).abs().max()}"
    assert torch.allclose(z, z_ref, atol=ATOL, rtol=RTOL), \
        f"Z diff: {(z - z_ref).abs().max()}"
    print("✓ Quadratic attention matches reference")

def test_gradients():
    print("\nTest 5: Gradients exist and are finite...")
    torch.manual_seed(42)
    model = AttentionLM.from_config(test_config)
    input_ids = torch.randint(0, V, (B, T))
    
    logits = model(input_ids)
    loss = compute_loss(logits, input_ids)
    loss.backward()
    
    assert model.embed.weight.grad is not None
    assert torch.isfinite(model.embed.weight.grad).all()
    assert torch.isfinite(loss)
    print("✓ Gradients exist and are finite")

def test_determinism():
    print("\nTest 6: Determinism...")
    torch.manual_seed(42)
    model_a = AttentionLM.from_config(test_config)
    torch.manual_seed(42)
    model_b = AttentionLM.from_config(test_config)
    
    input_ids = torch.randint(0, V, (B, T))
    logits_a = model_a(input_ids)
    logits_b = model_b(input_ids)
    
    assert torch.equal(logits_a, logits_b)
    print("✓ Same seed produces identical results")

def test_separate_embeddings():
    print("\nTest 7: Separate embed/unembed...")
    torch.manual_seed(42)
    model = AttentionLM.from_config(test_config)
    
    assert model.embed.weight is not model.unembed.weight
    print("✓ Embeddings are separate (not tied)")

if __name__ == "__main__":
    print("=" * 50)
    print("Running Quick Tests for Quadratic Attention LM")
    print("=" * 50)
    
    try:
        test_forward_shapes()
        test_rope_preserves_norm()
        test_causal_masking()
        test_quadratic_attention_math()
        test_gradients()
        test_determinism()
        test_separate_embeddings()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED ✓")
        print("=" * 50)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
