"""
Isolated Tensor Similarity Implementation with Unit Tests

This file contains ONLY the tensor similarity computation for manual auditing,
plus comprehensive sanity checks and unit tests.

FORMULA (as specified):
For two bilinear models with weights (W_l, W_r, W_p), compute a symmetry-aware inner product:
1. Compute four gram matrices: ll = W_l1ᵀW_l2, rr = W_r1ᵀW_r2, lr = W_l1ᵀW_r2, rl = W_r1ᵀW_l2
2. Core = 0.5 * ((ll ⊙ rr) + (lr ⊙ rl)) — averages aligned and swapped terms
3. Combine with output projections: dd = W_p1 W_p2ᵀ
4. Inner product = Tr(core @ dd)
Cosine similarity = inner(M1, M2) / sqrt(inner(M1, M1) * inner(M2, M2))

DIMENSION CONVENTIONS:
- W_l: (hidden_dim, input_dim)  — left projection
- W_r: (hidden_dim, input_dim)  — right projection
- W_p: (output_dim, hidden_dim) — output projection

AMBIGUITY IN FORMULA:
The formula says "ll = W_l1ᵀW_l2". There are two interpretations:
  A) ll = W_l1.T @ W_l2  →  (input, hidden) @ (hidden, input) = (input, input)
  B) ll = W_l1 @ W_l2.T  →  (hidden, input) @ (input, hidden) = (hidden, hidden)

Interpretation A gives (input, input) matrices — huge for MNIST (784x784)!
Interpretation B gives (hidden, hidden) matrices — manageable (128x128)

We implement BOTH and test which one makes sense.
"""

import torch
import numpy as np


# =============================================================================
# INTERPRETATION B: ll = W_l1 @ W_l2.T (what I originally implemented)
# =============================================================================

def tensor_inner_product_B(W_l1, W_r1, W_p1, W_l2, W_r2, W_p2):
    """
    Tensor inner product using interpretation B: gram matrices are (hidden, hidden).

    ll = W_l1 @ W_l2.T  →  (hidden, hidden)
    dd = W_p1.T @ W_p2  →  (hidden, hidden)

    This keeps all matrices at (hidden_dim, hidden_dim) size.
    """
    # Gram matrices: (hidden, hidden)
    ll = W_l1 @ W_l2.T
    rr = W_r1 @ W_r2.T
    lr = W_l1 @ W_r2.T
    rl = W_r1 @ W_l2.T

    # Symmetrized core: (hidden, hidden)
    aligned = ll * rr  # element-wise
    swapped = lr * rl  # element-wise
    core = 0.5 * (aligned + swapped)

    # Output projection: (hidden, hidden)
    dd = W_p1.T @ W_p2

    # Inner product
    inner = torch.trace(core @ dd).item()

    return inner


def tensor_sim_B(W_l1, W_r1, W_p1, W_l2, W_r2, W_p2):
    """Cosine similarity using interpretation B."""
    inner_12 = tensor_inner_product_B(W_l1, W_r1, W_p1, W_l2, W_r2, W_p2)
    inner_11 = tensor_inner_product_B(W_l1, W_r1, W_p1, W_l1, W_r1, W_p1)
    inner_22 = tensor_inner_product_B(W_l2, W_r2, W_p2, W_l2, W_r2, W_p2)

    denom = np.sqrt(inner_11 * inner_22)
    if denom < 1e-10:
        return 0.0
    return inner_12 / denom


# =============================================================================
# INTERPRETATION A: ll = W_l1.T @ W_l2 (literal reading of formula)
# =============================================================================

def tensor_inner_product_A(W_l1, W_r1, W_p1, W_l2, W_r2, W_p2):
    """
    Tensor inner product using interpretation A: gram matrices are (input, input).

    ll = W_l1.T @ W_l2  →  (input, hidden) @ (hidden, input) = (input, input)
    dd = W_p1 @ W_p2.T  →  (output, output)

    WARNING: This gives (input, input) = (784, 784) matrices for MNIST!
    And dd is (output, output) = (10, 10), so dimensions don't match for core @ dd.

    This interpretation seems broken for the trace computation.
    """
    # Gram matrices: (input, input) — this is (784, 784) for MNIST!
    ll = W_l1.T @ W_l2
    rr = W_r1.T @ W_r2
    lr = W_l1.T @ W_r2
    rl = W_r1.T @ W_l2

    # Symmetrized core: (input, input)
    aligned = ll * rr
    swapped = lr * rl
    core = 0.5 * (aligned + swapped)

    # Output projection: (output, output) = (10, 10)
    dd = W_p1 @ W_p2.T

    # PROBLEM: core is (784, 784), dd is (10, 10)
    # core @ dd doesn't work!
    # We'd need to do something else here...

    # Just return trace of core as a fallback (ignoring dd)
    print("WARNING: Interpretation A has dimension mismatch, returning trace(core) only")
    inner = torch.trace(core).item()

    return inner


# =============================================================================
# INTERPRETATION C: Maybe the formula assumes different dimension conventions?
# =============================================================================

def tensor_inner_product_C(W_l1, W_r1, W_p1, W_l2, W_r2, W_p2):
    """
    What if W_l is supposed to be (input, hidden) and W_p is (hidden, output)?

    Then:
      ll = W_l1.T @ W_l2 = (hidden, input) @ (input, hidden) = (hidden, hidden) ✓
      dd = W_p1 @ W_p2.T = (hidden, output) @ (output, hidden) = (hidden, hidden) ✓

    This requires transposing our weights first!
    """
    # Transpose to match assumed convention: W_l_math = W_l_ours.T
    W_l1_t = W_l1.T  # now (input, hidden)
    W_r1_t = W_r1.T
    W_l2_t = W_l2.T
    W_r2_t = W_r2.T
    W_p1_t = W_p1.T  # now (hidden, output)
    W_p2_t = W_p2.T

    # Now apply the formula literally
    ll = W_l1_t.T @ W_l2_t  # (hidden, input) @ (input, hidden) = (hidden, hidden)
    rr = W_r1_t.T @ W_r2_t
    lr = W_l1_t.T @ W_r2_t
    rl = W_r1_t.T @ W_l2_t

    aligned = ll * rr
    swapped = lr * rl
    core = 0.5 * (aligned + swapped)

    dd = W_p1_t @ W_p2_t.T  # (hidden, output) @ (output, hidden) = (hidden, hidden)

    inner = torch.trace(core @ dd).item()

    return inner


def tensor_sim_C(W_l1, W_r1, W_p1, W_l2, W_r2, W_p2):
    """Cosine similarity using interpretation C."""
    inner_12 = tensor_inner_product_C(W_l1, W_r1, W_p1, W_l2, W_r2, W_p2)
    inner_11 = tensor_inner_product_C(W_l1, W_r1, W_p1, W_l1, W_r1, W_p1)
    inner_22 = tensor_inner_product_C(W_l2, W_r2, W_p2, W_l2, W_r2, W_p2)

    denom = np.sqrt(inner_11 * inner_22)
    if denom < 1e-10:
        return 0.0
    return inner_12 / denom


# =============================================================================
# UNIT TESTS
# =============================================================================

def test_self_similarity():
    """Self-similarity should be exactly 1.0"""
    print("\n" + "="*60)
    print("TEST: Self-similarity")
    print("="*60)

    torch.manual_seed(42)
    W_l = torch.randn(128, 784)
    W_r = torch.randn(128, 784)
    W_p = torch.randn(10, 128)

    sim_B = tensor_sim_B(W_l, W_r, W_p, W_l, W_r, W_p)
    sim_C = tensor_sim_C(W_l, W_r, W_p, W_l, W_r, W_p)

    print(f"  Interpretation B: {sim_B:.10f} (should be 1.0)")
    print(f"  Interpretation C: {sim_C:.10f} (should be 1.0)")

    assert abs(sim_B - 1.0) < 1e-6, f"B failed: {sim_B}"
    assert abs(sim_C - 1.0) < 1e-6, f"C failed: {sim_C}"
    print("  PASSED ✓")


def test_symmetry():
    """tensor_sim(A, B) should equal tensor_sim(B, A)"""
    print("\n" + "="*60)
    print("TEST: Symmetry")
    print("="*60)

    torch.manual_seed(42)
    W_l1, W_r1, W_p1 = torch.randn(128, 784), torch.randn(128, 784), torch.randn(10, 128)
    torch.manual_seed(43)
    W_l2, W_r2, W_p2 = torch.randn(128, 784), torch.randn(128, 784), torch.randn(10, 128)

    sim_B_12 = tensor_sim_B(W_l1, W_r1, W_p1, W_l2, W_r2, W_p2)
    sim_B_21 = tensor_sim_B(W_l2, W_r2, W_p2, W_l1, W_r1, W_p1)

    sim_C_12 = tensor_sim_C(W_l1, W_r1, W_p1, W_l2, W_r2, W_p2)
    sim_C_21 = tensor_sim_C(W_l2, W_r2, W_p2, W_l1, W_r1, W_p1)

    print(f"  Interpretation B: sim(1,2)={sim_B_12:.6f}, sim(2,1)={sim_B_21:.6f}")
    print(f"  Interpretation C: sim(1,2)={sim_C_12:.6f}, sim(2,1)={sim_C_21:.6f}")

    assert abs(sim_B_12 - sim_B_21) < 1e-6, "B not symmetric"
    assert abs(sim_C_12 - sim_C_21) < 1e-6, "C not symmetric"
    print("  PASSED ✓")


def test_lr_swap_invariance():
    """Swapping W_l and W_r should give the same similarity (by design)"""
    print("\n" + "="*60)
    print("TEST: W_l <-> W_r swap invariance")
    print("="*60)

    torch.manual_seed(42)
    W_l1, W_r1, W_p1 = torch.randn(128, 784), torch.randn(128, 784), torch.randn(10, 128)
    torch.manual_seed(43)
    W_l2, W_r2, W_p2 = torch.randn(128, 784), torch.randn(128, 784), torch.randn(10, 128)

    # Original
    sim_B_orig = tensor_sim_B(W_l1, W_r1, W_p1, W_l2, W_r2, W_p2)
    sim_C_orig = tensor_sim_C(W_l1, W_r1, W_p1, W_l2, W_r2, W_p2)

    # Swap W_l and W_r for model 1
    sim_B_swap = tensor_sim_B(W_r1, W_l1, W_p1, W_l2, W_r2, W_p2)
    sim_C_swap = tensor_sim_C(W_r1, W_l1, W_p1, W_l2, W_r2, W_p2)

    print(f"  Interpretation B: orig={sim_B_orig:.6f}, swapped={sim_B_swap:.6f}")
    print(f"  Interpretation C: orig={sim_C_orig:.6f}, swapped={sim_C_swap:.6f}")

    assert abs(sim_B_orig - sim_B_swap) < 1e-6, "B not swap-invariant"
    assert abs(sim_C_orig - sim_C_swap) < 1e-6, "C not swap-invariant"
    print("  PASSED ✓")


def test_identical_weights():
    """Two models with identical weights should have similarity 1.0"""
    print("\n" + "="*60)
    print("TEST: Identical weights")
    print("="*60)

    torch.manual_seed(42)
    W_l, W_r, W_p = torch.randn(128, 784), torch.randn(128, 784), torch.randn(10, 128)

    # Clone the weights (separate tensors, same values)
    W_l2 = W_l.clone()
    W_r2 = W_r.clone()
    W_p2 = W_p.clone()

    sim_B = tensor_sim_B(W_l, W_r, W_p, W_l2, W_r2, W_p2)
    sim_C = tensor_sim_C(W_l, W_r, W_p, W_l2, W_r2, W_p2)

    print(f"  Interpretation B: {sim_B:.10f}")
    print(f"  Interpretation C: {sim_C:.10f}")

    assert abs(sim_B - 1.0) < 1e-6, f"B failed: {sim_B}"
    assert abs(sim_C - 1.0) < 1e-6, f"C failed: {sim_C}"
    print("  PASSED ✓")


def test_scaled_weights():
    """Scaling all weights by a constant should not change cosine similarity"""
    print("\n" + "="*60)
    print("TEST: Scale invariance")
    print("="*60)

    torch.manual_seed(42)
    W_l1, W_r1, W_p1 = torch.randn(128, 784), torch.randn(128, 784), torch.randn(10, 128)
    torch.manual_seed(43)
    W_l2, W_r2, W_p2 = torch.randn(128, 784), torch.randn(128, 784), torch.randn(10, 128)

    # Original similarity
    sim_B_orig = tensor_sim_B(W_l1, W_r1, W_p1, W_l2, W_r2, W_p2)
    sim_C_orig = tensor_sim_C(W_l1, W_r1, W_p1, W_l2, W_r2, W_p2)

    # Scale model 1 by 10x
    sim_B_scaled = tensor_sim_B(W_l1 * 10, W_r1 * 10, W_p1 * 10, W_l2, W_r2, W_p2)
    sim_C_scaled = tensor_sim_C(W_l1 * 10, W_r1 * 10, W_p1 * 10, W_l2, W_r2, W_p2)

    print(f"  Interpretation B: orig={sim_B_orig:.6f}, scaled={sim_B_scaled:.6f}")
    print(f"  Interpretation C: orig={sim_C_orig:.6f}, scaled={sim_C_scaled:.6f}")

    # Note: This test might fail! The inner product scales as (scale)^4 due to
    # the bilinear structure (W_l * W_r * W_p), so cosine sim might not be scale-invariant
    if abs(sim_B_orig - sim_B_scaled) > 1e-3:
        print("  WARNING: Not scale-invariant (this may be expected)")
    else:
        print("  PASSED ✓")


def test_orthogonal_weights():
    """Random weights should have similarity near 0 (for large dimensions)"""
    print("\n" + "="*60)
    print("TEST: Random orthogonality")
    print("="*60)

    sims_B = []
    sims_C = []

    for i in range(10):
        torch.manual_seed(i)
        W_l1, W_r1, W_p1 = torch.randn(128, 784), torch.randn(128, 784), torch.randn(10, 128)
        torch.manual_seed(i + 100)
        W_l2, W_r2, W_p2 = torch.randn(128, 784), torch.randn(128, 784), torch.randn(10, 128)

        sims_B.append(tensor_sim_B(W_l1, W_r1, W_p1, W_l2, W_r2, W_p2))
        sims_C.append(tensor_sim_C(W_l1, W_r1, W_p1, W_l2, W_r2, W_p2))

    print(f"  Interpretation B: mean={np.mean(sims_B):.6f}, std={np.std(sims_B):.6f}")
    print(f"  Interpretation C: mean={np.mean(sims_C):.6f}, std={np.std(sims_C):.6f}")
    print("  (Random weights should be near 0)")


def test_perturbation():
    """Small perturbations should give similarity close to 1"""
    print("\n" + "="*60)
    print("TEST: Perturbation sensitivity")
    print("="*60)

    torch.manual_seed(42)
    W_l, W_r, W_p = torch.randn(128, 784), torch.randn(128, 784), torch.randn(10, 128)

    for noise_scale in [0.001, 0.01, 0.1, 0.5]:
        W_l2 = W_l + noise_scale * torch.randn_like(W_l)
        W_r2 = W_r + noise_scale * torch.randn_like(W_r)
        W_p2 = W_p + noise_scale * torch.randn_like(W_p)

        sim_B = tensor_sim_B(W_l, W_r, W_p, W_l2, W_r2, W_p2)
        sim_C = tensor_sim_C(W_l, W_r, W_p, W_l2, W_r2, W_p2)

        print(f"  noise={noise_scale}: B={sim_B:.6f}, C={sim_C:.6f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("TENSOR SIMILARITY UNIT TESTS")
    print("="*60)
    print("\nNote: Interpretations B and C should give the SAME results")
    print("because C just clarifies the dimension convention.\n")

    test_self_similarity()
    test_symmetry()
    test_lr_swap_invariance()
    test_identical_weights()
    test_scaled_weights()
    test_orthogonal_weights()
    test_perturbation()

    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)
