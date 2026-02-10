"""
Tensor Similarity with Symmetrized Interaction Matrices

The hypothesis: different seeds learn the same symmetric (functional) part
but different antisymmetric (irrelevant) parts of the interaction matrix.

For each hidden unit i, the interaction matrix is:
    M_i[j,k] = W_l[i,j] * W_r[i,k]

The symmetric part (the only part that affects output):
    M_i_sym[j,k] = (M_i[j,k] + M_i[k,j]) / 2
                 = (W_l[i,j]*W_r[i,k] + W_l[i,k]*W_r[i,j]) / 2

We compare ONLY the symmetric parts.
"""

import torch
import numpy as np
from pathlib import Path


def compute_interaction_matrices(W_l, W_r, symmetrize=True):
    """
    Compute interaction matrices for all hidden units.

    Args:
        W_l: (hidden_dim, input_dim)
        W_r: (hidden_dim, input_dim)
        symmetrize: If True, return only the symmetric part

    Returns:
        M: (hidden_dim, input_dim, input_dim) tensor of interaction matrices
    """
    hidden_dim, input_dim = W_l.shape

    # M[i,j,k] = W_l[i,j] * W_r[i,k]
    # This is outer product for each hidden unit
    # W_l[:, :, None] is (hidden, input, 1)
    # W_r[:, None, :] is (hidden, 1, input)
    # Product is (hidden, input, input)
    M = W_l[:, :, None] * W_r[:, None, :]

    if symmetrize:
        # M_sym[i,j,k] = (M[i,j,k] + M[i,k,j]) / 2
        M = (M + M.transpose(1, 2)) / 2

    return M


def tensor_sim_via_interaction(W_l1, W_r1, W_p1, W_l2, W_r2, W_p2, symmetrize=True):
    """
    Compute tensor similarity by explicitly building interaction matrices.

    Args:
        symmetrize: If True, symmetrize interaction matrices before comparing

    Returns:
        Cosine similarity between models
    """
    # Compute interaction matrices: (hidden, input, input)
    M1 = compute_interaction_matrices(W_l1, W_r1, symmetrize=symmetrize)
    M2 = compute_interaction_matrices(W_l2, W_r2, symmetrize=symmetrize)

    # Weight by output projection: W_p is (output, hidden)
    # We want to weight each hidden unit's contribution by how much it affects output
    #
    # Full tensor would be T[o,j,k] = sum_i W_p[o,i] * M[i,j,k]
    # Inner product: sum_{o,j,k} T1[o,j,k] * T2[o,j,k]
    #              = sum_{o,j,k} (sum_i W_p1[o,i] M1[i,j,k]) * (sum_i' W_p2[o,i'] M2[i',j,k])
    #
    # This is expensive. Let's simplify:
    # = sum_{o,i,i',j,k} W_p1[o,i] W_p2[o,i'] M1[i,j,k] M2[i',j,k]
    # = sum_{i,i'} (sum_o W_p1[o,i] W_p2[o,i']) * (sum_{j,k} M1[i,j,k] M2[i',j,k])
    # = sum_{i,i'} (W_p1.T @ W_p2)[i,i'] * <M1[i], M2[i']>

    # Gram matrix of output projections: (hidden, hidden)
    P = W_p1.T @ W_p2

    # Gram matrix of interaction matrices: (hidden, hidden)
    # G[i,i'] = sum_{j,k} M1[i,j,k] * M2[i',j,k]
    # Flatten M to (hidden, input*input), then G = M1_flat @ M2_flat.T
    hidden_dim = M1.shape[0]
    M1_flat = M1.reshape(hidden_dim, -1)
    M2_flat = M2.reshape(hidden_dim, -1)
    G = M1_flat @ M2_flat.T

    # Inner product: trace of element-wise product
    inner = torch.sum(P * G).item()

    return inner


def tensor_sim_cosine(W_l1, W_r1, W_p1, W_l2, W_r2, W_p2, symmetrize=True):
    """Cosine similarity using interaction matrix approach."""
    inner_12 = tensor_sim_via_interaction(W_l1, W_r1, W_p1, W_l2, W_r2, W_p2, symmetrize)
    inner_11 = tensor_sim_via_interaction(W_l1, W_r1, W_p1, W_l1, W_r1, W_p1, symmetrize)
    inner_22 = tensor_sim_via_interaction(W_l2, W_r2, W_p2, W_l2, W_r2, W_p2, symmetrize)

    denom = np.sqrt(inner_11 * inner_22)
    if denom < 1e-10:
        return 0.0
    return inner_12 / denom


# =============================================================================
# TESTS
# =============================================================================

def load_model_weights(seed, epoch):
    """Load weights from checkpoint."""
    from tensor_sim_experiment import BilinearMLP
    path = f"checkpoints/seed_{seed}/epoch_{epoch:02d}.pt"
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model = BilinearMLP(input_dim=784, hidden_dim=128, output_dim=10)
    model.load_state_dict(ckpt['model_state_dict'])
    return model.W_l.detach(), model.W_r.detach(), model.W_p.detach()


if __name__ == "__main__":
    print("="*60)
    print("TENSOR SIM WITH SYMMETRIZED INTERACTION MATRICES")
    print("="*60)

    # Test 1: Self-similarity
    print("\n1. Self-similarity test:")
    torch.manual_seed(42)
    W_l = torch.randn(128, 784)
    W_r = torch.randn(128, 784)
    W_p = torch.randn(10, 128)

    sim_sym = tensor_sim_cosine(W_l, W_r, W_p, W_l, W_r, W_p, symmetrize=True)
    sim_nosym = tensor_sim_cosine(W_l, W_r, W_p, W_l, W_r, W_p, symmetrize=False)
    print(f"   With symmetrization:    {sim_sym:.6f} (should be 1.0)")
    print(f"   Without symmetrization: {sim_nosym:.6f} (should be 1.0)")

    # Test 2: Random models
    print("\n2. Random model similarity:")
    torch.manual_seed(42)
    W_l1, W_r1, W_p1 = torch.randn(128, 784), torch.randn(128, 784), torch.randn(10, 128)
    torch.manual_seed(43)
    W_l2, W_r2, W_p2 = torch.randn(128, 784), torch.randn(128, 784), torch.randn(10, 128)

    sim_sym = tensor_sim_cosine(W_l1, W_r1, W_p1, W_l2, W_r2, W_p2, symmetrize=True)
    sim_nosym = tensor_sim_cosine(W_l1, W_r1, W_p1, W_l2, W_r2, W_p2, symmetrize=False)
    print(f"   With symmetrization:    {sim_sym:.6f}")
    print(f"   Without symmetrization: {sim_nosym:.6f}")

    # Test 3: Trained models - cross seed
    print("\n3. Trained models (epoch 20) - CROSS SEED:")
    W_l0, W_r0, W_p0 = load_model_weights(0, 20)
    W_l1, W_r1, W_p1 = load_model_weights(1, 20)
    W_l2, W_r2, W_p2 = load_model_weights(2, 20)

    print("   Seed 0 vs Seed 1:")
    sim_sym = tensor_sim_cosine(W_l0, W_r0, W_p0, W_l1, W_r1, W_p1, symmetrize=True)
    sim_nosym = tensor_sim_cosine(W_l0, W_r0, W_p0, W_l1, W_r1, W_p1, symmetrize=False)
    print(f"      With symmetrization:    {sim_sym:.6f}")
    print(f"      Without symmetrization: {sim_nosym:.6f}")

    print("   Seed 0 vs Seed 2:")
    sim_sym = tensor_sim_cosine(W_l0, W_r0, W_p0, W_l2, W_r2, W_p2, symmetrize=True)
    sim_nosym = tensor_sim_cosine(W_l0, W_r0, W_p0, W_l2, W_r2, W_p2, symmetrize=False)
    print(f"      With symmetrization:    {sim_sym:.6f}")
    print(f"      Without symmetrization: {sim_nosym:.6f}")

    print("   Seed 1 vs Seed 2:")
    sim_sym = tensor_sim_cosine(W_l1, W_r1, W_p1, W_l2, W_r2, W_p2, symmetrize=True)
    sim_nosym = tensor_sim_cosine(W_l1, W_r1, W_p1, W_l2, W_r2, W_p2, symmetrize=False)
    print(f"      With symmetrization:    {sim_sym:.6f}")
    print(f"      Without symmetrization: {sim_nosym:.6f}")

    # Test 4: Same seed, different epochs
    print("\n4. Same seed (0), different epochs:")
    W_l_e1, W_r_e1, W_p_e1 = load_model_weights(0, 1)
    W_l_e17, W_r_e17, W_p_e17 = load_model_weights(0, 17)
    W_l_e20, W_r_e20, W_p_e20 = load_model_weights(0, 20)

    print("   Epoch 17 vs Epoch 20:")
    sim_sym = tensor_sim_cosine(W_l_e17, W_r_e17, W_p_e17, W_l_e20, W_r_e20, W_p_e20, symmetrize=True)
    sim_nosym = tensor_sim_cosine(W_l_e17, W_r_e17, W_p_e17, W_l_e20, W_r_e20, W_p_e20, symmetrize=False)
    print(f"      With symmetrization:    {sim_sym:.6f}")
    print(f"      Without symmetrization: {sim_nosym:.6f}")

    print("   Epoch 1 vs Epoch 20:")
    sim_sym = tensor_sim_cosine(W_l_e1, W_r_e1, W_p_e1, W_l_e20, W_r_e20, W_p_e20, symmetrize=True)
    sim_nosym = tensor_sim_cosine(W_l_e1, W_r_e1, W_p_e1, W_l_e20, W_r_e20, W_p_e20, symmetrize=False)
    print(f"      With symmetrization:    {sim_sym:.6f}")
    print(f"      Without symmetrization: {sim_nosym:.6f}")

    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print("\nIf symmetrization helps, cross-seed similarity should increase")
    print("while within-seed similarity should stay similar.")
