"""Tensor similarity metrics (standard and generalized) and functional similarity metrics."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple

from model import BilinearMLP


def one_hot_covariance(p: int) -> torch.Tensor:
    """
    Covariance matrix for uniform one-hot distribution over {0, ..., p-1}.

    For one-hot vectors uniformly distributed:
    - E[x_i] = 1/p
    - Cov(x_i, x_j) = E[x_i * x_j] - E[x_i]*E[x_j]
                    = (1/p)δ_ij - 1/p²

    So: Σ = (1/p)I - (1/p²)11ᵀ

    Args:
        p: Dimension (number of classes)

    Returns:
        (p, p) covariance matrix
    """
    I = torch.eye(p)
    ones = torch.ones(p, p)
    return (1.0 / p) * I - (1.0 / (p * p)) * ones


def tensor_inner_product_standard(model1: BilinearMLP, model2: BilinearMLP) -> float:
    """
    Standard tensor inner product (M = I).

    For bilinear networks with tensor T = W_p ⊗ W_l ⊗ W_r, the inner product is:
    <T1, T2> = Σ_ijk T1_ijk * T2_ijk
             = Σ_k (W_p1)_k · (W_p2)_k * [0.5 * ((W_l1·W_l2)(W_r1·W_r2) + (W_l1·W_r2)(W_r1·W_l2))]_k

    This simplifies to: trace(D @ Core) where D_kl = (W_p1.T @ W_p2)_kl and
    Core = 0.5 * (LL * RR + LR * RL) with LL = W_l1 @ W_l2.T, etc.

    Args:
        model1, model2: BilinearMLP models

    Returns:
        Scalar inner product value
    """
    W_l1 = model1.W_l.detach().cpu()
    W_r1 = model1.W_r.detach().cpu()
    W_p1 = model1.W_p.detach().cpu()
    W_l2 = model2.W_l.detach().cpu()
    W_r2 = model2.W_r.detach().cpu()
    W_p2 = model2.W_p.detach().cpu()

    # Gram matrices for input weights
    ll = W_l1 @ W_l2.T  # (hidden, hidden)
    rr = W_r1 @ W_r2.T  # (hidden, hidden)
    lr = W_l1 @ W_r2.T  # (hidden, hidden)
    rl = W_r1 @ W_l2.T  # (hidden, hidden)

    # Core bilinear structure (accounts for symmetry in product)
    core = 0.5 * (ll * rr + lr * rl)

    # Output weight Gram matrix
    dd = W_p1.T @ W_p2  # (hidden, hidden)

    return torch.sum(core * dd).item()


def tensor_inner_product_generalized(
    model1: BilinearMLP,
    model2: BilinearMLP,
    Sigma_l: torch.Tensor,
    Sigma_r: torch.Tensor
) -> float:
    """
    Generalized tensor inner product with metric tensors.

    Instead of standard Gram matrices W1 @ W2.T, we use Σ-weighted versions:
    W1 @ Σ @ W2.T

    This accounts for the actual input distribution (one-hot covariance).

    Args:
        model1, model2: BilinearMLP models
        Sigma_l: Covariance matrix for left input (p x p)
        Sigma_r: Covariance matrix for right input (p x p)

    Returns:
        Scalar inner product value
    """
    W_l1 = model1.W_l.detach().cpu()
    W_r1 = model1.W_r.detach().cpu()
    W_p1 = model1.W_p.detach().cpu()
    W_l2 = model2.W_l.detach().cpu()
    W_r2 = model2.W_r.detach().cpu()
    W_p2 = model2.W_p.detach().cpu()

    # Σ-weighted Gram matrices
    ll = W_l1 @ Sigma_l @ W_l2.T  # (hidden, hidden)
    rr = W_r1 @ Sigma_r @ W_r2.T  # (hidden, hidden)
    lr = W_l1 @ Sigma_l @ W_r2.T  # Cross-term: use Σ_l since W_l is "from"
    rl = W_r1 @ Sigma_r @ W_l2.T  # Cross-term: use Σ_r since W_r is "from"

    # Core bilinear structure
    core = 0.5 * (ll * rr + lr * rl)

    # Output weight Gram matrix (no change - outputs don't have distribution weighting)
    dd = W_p1.T @ W_p2

    return torch.sum(core * dd).item()


def tensor_similarity_standard(model1: BilinearMLP, model2: BilinearMLP) -> float:
    """
    Normalized tensor similarity (standard, M = I).

    sim(A, B) = <A, B> / sqrt(<A, A> * <B, B>)

    Args:
        model1, model2: BilinearMLP models

    Returns:
        Cosine similarity in [-1, 1]
    """
    i12 = tensor_inner_product_standard(model1, model2)
    i11 = tensor_inner_product_standard(model1, model1)
    i22 = tensor_inner_product_standard(model2, model2)

    if i11 <= 0 or i22 <= 0:
        return 0.0

    return i12 / np.sqrt(i11 * i22)


def tensor_similarity_generalized(
    model1: BilinearMLP,
    model2: BilinearMLP,
    Sigma_l: torch.Tensor,
    Sigma_r: torch.Tensor
) -> float:
    """
    Normalized tensor similarity (generalized, M = Σ).

    sim(A, B) = <A, B>_Σ / sqrt(<A, A>_Σ * <B, B>_Σ)

    Args:
        model1, model2: BilinearMLP models
        Sigma_l: Covariance matrix for left input
        Sigma_r: Covariance matrix for right input

    Returns:
        Cosine similarity in [-1, 1]
    """
    i12 = tensor_inner_product_generalized(model1, model2, Sigma_l, Sigma_r)
    i11 = tensor_inner_product_generalized(model1, model1, Sigma_l, Sigma_r)
    i22 = tensor_inner_product_generalized(model2, model2, Sigma_l, Sigma_r)

    if i11 <= 0 or i22 <= 0:
        return 0.0

    return i12 / np.sqrt(i11 * i22)


def compute_output_agreement(
    model1: BilinearMLP,
    model2: BilinearMLP,
    test_loader,
    device: str
) -> float:
    """
    Compute output agreement (fraction of inputs where argmax predictions match).

    Args:
        model1, model2: BilinearMLP models
        test_loader: DataLoader for evaluation
        device: Device string

    Returns:
        Agreement fraction in [0, 1]
    """
    model1.eval()
    model2.eval()
    total, agree = 0, 0

    with torch.no_grad():
        for x_a, x_b, _ in test_loader:
            x_a, x_b = x_a.to(device), x_b.to(device)
            pred1 = model1(x_a, x_b).argmax(dim=1)
            pred2 = model2(x_a, x_b).argmax(dim=1)
            agree += (pred1 == pred2).sum().item()
            total += x_a.size(0)

    return agree / total


def compute_logit_cosine(
    model1: BilinearMLP,
    model2: BilinearMLP,
    test_loader,
    device: str
) -> float:
    """
    Compute average logit cosine similarity across test set.

    Args:
        model1, model2: BilinearMLP models
        test_loader: DataLoader for evaluation
        device: Device string

    Returns:
        Mean cosine similarity of logit vectors
    """
    model1.eval()
    model2.eval()
    cosines = []

    with torch.no_grad():
        for x_a, x_b, _ in test_loader:
            x_a, x_b = x_a.to(device), x_b.to(device)
            logits1 = model1(x_a, x_b)
            logits2 = model2(x_a, x_b)
            # Per-sample cosine similarity
            cos = F.cosine_similarity(logits1, logits2, dim=1)
            cosines.append(cos)

    return torch.cat(cosines).mean().item()


def compute_kl_divergence(
    model1: BilinearMLP,
    model2: BilinearMLP,
    test_loader,
    device: str
) -> float:
    """
    Compute average KL divergence from model1's predictions to model2's.

    KL(P1 || P2) where P1, P2 are softmax distributions.

    Args:
        model1, model2: BilinearMLP models
        test_loader: DataLoader for evaluation
        device: Device string

    Returns:
        Mean KL divergence (non-negative)
    """
    model1.eval()
    model2.eval()
    kl_divs = []

    with torch.no_grad():
        for x_a, x_b, _ in test_loader:
            x_a, x_b = x_a.to(device), x_b.to(device)
            logits1 = model1(x_a, x_b)
            logits2 = model2(x_a, x_b)

            log_probs1 = F.log_softmax(logits1, dim=1)
            probs1 = F.softmax(logits1, dim=1)
            log_probs2 = F.log_softmax(logits2, dim=1)

            # KL(P1 || P2) = sum(P1 * (log P1 - log P2))
            kl = (probs1 * (log_probs1 - log_probs2)).sum(dim=1)
            kl_divs.append(kl)

    return torch.cat(kl_divs).mean().item()
