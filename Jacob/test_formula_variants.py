"""
Test different variants of the tensor sim formula to identify the discrepancy.
"""

import torch
import numpy as np

def load_model_weights(seed, epoch):
    """Load weights from checkpoint."""
    from tensor_sim_experiment import BilinearMLP
    path = f"checkpoints/seed_{seed}/epoch_{epoch:02d}.pt"
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model = BilinearMLP(input_dim=784, hidden_dim=128, output_dim=10)
    model.load_state_dict(ckpt['model_state_dict'])
    return model.W_l.detach(), model.W_r.detach(), model.W_p.detach()


def original_formula(W_l1, W_r1, W_p1, W_l2, W_r2, W_p2):
    """Original formula: trace(core @ dd)"""
    ll = W_l1 @ W_l2.T
    rr = W_r1 @ W_r2.T
    lr = W_l1 @ W_r2.T
    rl = W_r1 @ W_l2.T

    aligned = ll * rr
    swapped = lr * rl
    core = 0.5 * (aligned + swapped)

    dd = W_p1.T @ W_p2

    # Original: matrix multiply then trace
    inner = torch.trace(core @ dd).item()
    return inner


def variant_A(W_l1, W_r1, W_p1, W_l2, W_r2, W_p2):
    """Variant A: sum(core * dd) - element-wise, no transpose issue"""
    ll = W_l1 @ W_l2.T
    rr = W_r1 @ W_r2.T
    lr = W_l1 @ W_r2.T
    rl = W_r1 @ W_l2.T

    aligned = ll * rr
    swapped = lr * rl
    core = 0.5 * (aligned + swapped)

    dd = W_p1.T @ W_p2

    # Fix: element-wise product then sum
    inner = torch.sum(core * dd).item()
    return inner


def variant_B(W_l1, W_r1, W_p1, W_l2, W_r2, W_p2):
    """Variant B: trace(core @ dd.T) - explicitly transpose dd"""
    ll = W_l1 @ W_l2.T
    rr = W_r1 @ W_r2.T
    lr = W_l1 @ W_r2.T
    rl = W_r1 @ W_l2.T

    aligned = ll * rr
    swapped = lr * rl
    core = 0.5 * (aligned + swapped)

    dd = W_p1.T @ W_p2

    # Use dd.T explicitly
    inner = torch.trace(core @ dd.T).item()
    return inner


def variant_C(W_l1, W_r1, W_p1, W_l2, W_r2, W_p2):
    """Variant C: No swapped term (just aligned), element-wise sum"""
    ll = W_l1 @ W_l2.T
    rr = W_r1 @ W_r2.T

    # Only aligned term, no swap symmetrization
    core = ll * rr

    dd = W_p1.T @ W_p2

    inner = torch.sum(core * dd).item()
    return inner


def new_interaction_formula(W_l1, W_r1, W_p1, W_l2, W_r2, W_p2):
    """New formula with explicit interaction matrices"""
    M1 = W_l1[:, :, None] * W_r1[:, None, :]
    M2 = W_l2[:, :, None] * W_r2[:, None, :]

    hidden_dim = M1.shape[0]
    M1_flat = M1.reshape(hidden_dim, -1)
    M2_flat = M2.reshape(hidden_dim, -1)
    G = M1_flat @ M2_flat.T

    P = W_p1.T @ W_p2

    inner = torch.sum(P * G).item()
    return inner


def cosine_sim(inner_func, W_l1, W_r1, W_p1, W_l2, W_r2, W_p2):
    """Compute cosine similarity using given inner product function."""
    inner_12 = inner_func(W_l1, W_r1, W_p1, W_l2, W_r2, W_p2)
    inner_11 = inner_func(W_l1, W_r1, W_p1, W_l1, W_r1, W_p1)
    inner_22 = inner_func(W_l2, W_r2, W_p2, W_l2, W_r2, W_p2)

    denom = np.sqrt(inner_11 * inner_22)
    if denom < 1e-10:
        return 0.0
    return inner_12 / denom


if __name__ == "__main__":
    print("="*70)
    print("FORMULA VARIANT COMPARISON")
    print("="*70)

    # Load trained models
    print("\nLoading models...")
    W_l0, W_r0, W_p0 = load_model_weights(0, 20)
    W_l1, W_r1, W_p1 = load_model_weights(1, 20)

    formulas = [
        ("Original: trace(core @ dd)", original_formula),
        ("Variant A: sum(core * dd)", variant_A),
        ("Variant B: trace(core @ dd.T)", variant_B),
        ("Variant C: no swap, sum(core * dd)", variant_C),
        ("New: interaction matrices", new_interaction_formula),
    ]

    print("\n" + "-"*70)
    print("INNER PRODUCTS (not normalized)")
    print("-"*70)

    for name, func in formulas:
        inner_11 = func(W_l0, W_r0, W_p0, W_l0, W_r0, W_p0)
        inner_12 = func(W_l0, W_r0, W_p0, W_l1, W_r1, W_p1)
        print(f"{name:40s}: self={inner_11:.6e}, cross={inner_12:.6e}")

    print("\n" + "-"*70)
    print("COSINE SIMILARITIES (normalized)")
    print("-"*70)

    for name, func in formulas:
        sim = cosine_sim(func, W_l0, W_r0, W_p0, W_l1, W_r1, W_p1)
        print(f"{name:40s}: {sim:.6f}")

    print("\n" + "-"*70)
    print("CHECKING MATHEMATICAL EQUIVALENCES")
    print("-"*70)

    # Check if variant_C equals new_interaction_formula
    inner_C = variant_C(W_l0, W_r0, W_p0, W_l1, W_r1, W_p1)
    inner_new = new_interaction_formula(W_l0, W_r0, W_p0, W_l1, W_r1, W_p1)
    print(f"\nVariant C (no swap) vs New interaction:")
    print(f"  Variant C:   {inner_C:.6e}")
    print(f"  New:         {inner_new:.6e}")
    print(f"  Equal? {np.isclose(inner_C, inner_new)}")

    # Check relationship between trace(A@B) and sum(A*B.T)
    ll = W_l0 @ W_l1.T
    rr = W_r0 @ W_r1.T
    core = 0.5 * ((ll * rr) + (W_l0 @ W_r1.T) * (W_r0 @ W_l1.T))
    dd = W_p0.T @ W_p1

    print(f"\nTrace vs element-wise sum relationship:")
    print(f"  trace(core @ dd)   = {torch.trace(core @ dd).item():.6e}")
    print(f"  sum(core * dd.T)   = {torch.sum(core * dd.T).item():.6e}")
    print(f"  sum(core * dd)     = {torch.sum(core * dd).item():.6e}")
    print(f"  trace(core @ dd.T) = {torch.trace(core @ dd.T).item():.6e}")

    print(f"\nIs dd symmetric? max|dd - dd.T| = {torch.max(torch.abs(dd - dd.T)).item():.6e}")
