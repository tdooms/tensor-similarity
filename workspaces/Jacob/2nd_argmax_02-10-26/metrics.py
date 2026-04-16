"""
Tensor similarity and functional similarity metrics.
Works with BilinearStack models from bilinear-2nd-argmax.py.

Tensor for one bilinear layer: T_kij = sum_r D_kr L_ri R_rj
Symmetrized:                   T^sym_kij = 0.5 * (T_kij + T_kji)

Inner product of symmetrized tensors:
  <T1, T2> = sum(dd * 0.5*(ll*rr + lr*rl))
where ll = L1 @ L2.T, rr = R1 @ R2.T, lr = L1 @ R2.T, rl = R1 @ L2.T
      dd = D1.T @ D2

Generalized (with covariance Sigma of input distribution):
  Replace L1 @ L2.T with L1 @ Sigma @ L2.T, etc.

For multi-layer BilinearStack: sum per-layer inner products.
(This is approximate — layers compose, so the full network is not a single tensor.)
"""

import torch
import torch.nn.functional as F
import math


# --- Single-layer tensor inner product ---

def tensor_inner_product(L1, R1, D1, L2, R2, D2, Sigma=None):
    """
    Tensor inner product for one bilinear layer.

    Args:
        L1, R1: (rank, n) input projections for model 1
        D1: (n_out, rank) output projection for model 1
        L2, R2, D2: same for model 2
        Sigma: (n, n) input covariance, or None for identity
    """
    if Sigma is not None:
        ll = L1 @ Sigma @ L2.T
        rr = R1 @ Sigma @ R2.T
        lr = L1 @ Sigma @ R2.T
        rl = R1 @ Sigma @ L2.T
    else:
        ll = L1 @ L2.T
        rr = R1 @ R2.T
        lr = L1 @ R2.T
        rl = R1 @ L2.T

    core = 0.5 * (ll * rr + lr * rl)
    dd = D1.T @ D2
    return torch.sum(core * dd).item()


def tensor_inner_product_bruteforce(L1, R1, D1, L2, R2, D2):
    """
    Brute-force: explicitly build T_kij = sum_r D_kr L_ri R_rj,
    symmetrize, then dot product. For verification only.
    """
    T1 = torch.einsum('kr,ri,rj->kij', D1, L1, R1)
    T2 = torch.einsum('kr,ri,rj->kij', D2, L2, R2)
    T1 = 0.5 * (T1 + T1.transpose(1, 2))
    T2 = 0.5 * (T2 + T2.transpose(1, 2))
    return torch.sum(T1 * T2).item()


# --- Full-model tensor similarity ---

def _model_inner_product(model1, model2, Sigma=None):
    """Sum of per-layer tensor inner products."""
    total = 0.0
    for i in range(model1.num_layers):
        total += tensor_inner_product(
            model1.Ls[i].detach(), model1.Rs[i].detach(), model1.Ds[i].detach(),
            model2.Ls[i].detach(), model2.Rs[i].detach(), model2.Ds[i].detach(),
            Sigma
        )
    return total


def tensor_similarity(model1, model2, Sigma=None):
    """Cosine tensor similarity between two BilinearStack models."""
    i12 = _model_inner_product(model1, model2, Sigma)
    i11 = _model_inner_product(model1, model1, Sigma)
    i22 = _model_inner_product(model2, model2, Sigma)
    denom = math.sqrt(i11 * i22)
    if denom < 1e-10:
        return 0.0
    return i12 / denom


# --- Functional similarity ---

def functional_similarity(model1, model2, dist_fn, n=4, num_samples=100000):
    """Output agreement: fraction of inputs where argmax predictions match."""
    model1.eval()
    model2.eval()
    with torch.no_grad():
        x = dist_fn(num_samples, n)
        agreement = (model1(x).argmax(1) == model2(x).argmax(1)).float().mean().item()
    return agreement


# --- Covariance estimation ---

def estimate_covariance(dist_fn, n=4, num_samples=100000):
    """Estimate covariance matrix of a distribution by sampling."""
    x = dist_fn(num_samples, n)
    return torch.cov(x.T)


# --- Verification ---

def verify():
    """Confirm formula matches brute-force on random weights."""
    torch.manual_seed(42)
    L1 = torch.randn(8, 4)
    R1 = torch.randn(8, 4)
    D1 = torch.randn(4, 8)
    L2 = torch.randn(8, 4)
    R2 = torch.randn(8, 4)
    D2 = torch.randn(4, 8)

    formula = tensor_inner_product(L1, R1, D1, L2, R2, D2)
    brute = tensor_inner_product_bruteforce(L1, R1, D1, L2, R2, D2)
    print(f"Formula:     {formula:.6f}")
    print(f"Brute force: {brute:.6f}")
    print(f"Match: {abs(formula - brute) < 1e-4}")

    # Self-similarity should be 1.0
    self_inner = tensor_inner_product(L1, R1, D1, L1, R1, D1)
    self_brute = tensor_inner_product_bruteforce(L1, R1, D1, L1, R1, D1)
    cos = self_inner / math.sqrt(self_inner * self_inner)
    print(f"\nSelf-similarity: {cos:.6f} (should be 1.0)")
    print(f"Self inner (formula): {self_inner:.6f}")
    print(f"Self inner (brute):   {self_brute:.6f}")
    print(f"Match: {abs(self_inner - self_brute) < 1e-4}")


if __name__ == "__main__":
    verify()
