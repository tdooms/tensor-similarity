"""
Tensor similarity metrics for BilinearStack models.

Single-layer formula (symmetrized):
  <T1, T2>_Σ = Σ_k [ 0.5*(L1ΣL2ᵀ ⊙ R1ΣR2ᵀ + L1ΣR2ᵀ ⊙ R1ΣL2ᵀ) ⊙ D1ᵀD2 ]

Multi-layer extension (approximate — ignores cross-layer composition terms):
  <f1, f2> ≈ Σ_i <T1_i, T2_i>_{Σ_i}
  where Σ_i = Cov[h_{i-1}] = covariance of inputs to layer i.

Layer-sigma variants:
  Layer 0 Σ: input distribution covariance (same for all models on same dist)
  Layer 1 Σ: Cov[x + T0(x,x)], estimated per model, then averaged for cross-model inner products
"""

import torch
import math


def tensor_inner_product(L1, R1, D1, L2, R2, D2, Sigma=None):
    """Symmetrized bilinear tensor inner product, optionally weighted by Sigma."""
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


def _model_inner_product(model1, model2, sigmas):
    """Sum of per-layer tensor inner products using provided list of Sigmas."""
    return sum(
        tensor_inner_product(
            model1.Ls[i].detach(), model1.Rs[i].detach(), model1.Ds[i].detach(),
            model2.Ls[i].detach(), model2.Rs[i].detach(), model2.Ds[i].detach(),
            sigmas[i],
        )
        for i in range(model1.num_layers)
    )


def tensor_similarity(model1, model2, Sigma=None):
    """
    Cosine tensor similarity with the same Sigma applied to all layers.
    Sigma=None: identity (standard metric).
    Pass estimated input covariance for the generalized variant.
    """
    sigmas = [Sigma] * model1.num_layers
    ip12 = _model_inner_product(model1, model2, sigmas)
    ip11 = _model_inner_product(model1, model1, sigmas)
    ip22 = _model_inner_product(model2, model2, sigmas)
    denom = math.sqrt(max(ip11 * ip22, 0.0))
    return ip12 / denom if denom > 1e-10 else 0.0


def tensor_similarity_layer_sigma(model1, model2, sigmas_m1, sigmas_m2):
    """
    Per-layer Sigma tensor similarity.

    sigmas_m1: list [Σ_0, Σ_1, ...] where Σ_i = Cov[h_{i-1}] estimated under model1.
    sigmas_m2: same for model2.

    Cross inner product uses average Sigmas; self-norms use each model's own Sigmas
    (consistent with each model's hidden-state geometry).
    """
    sigmas_cross = [0.5 * (s1 + s2) for s1, s2 in zip(sigmas_m1, sigmas_m2)]
    ip12 = _model_inner_product(model1, model2, sigmas_cross)
    ip11 = _model_inner_product(model1, model1, sigmas_m1)
    ip22 = _model_inner_product(model2, model2, sigmas_m2)
    denom = math.sqrt(max(ip11 * ip22, 0.0))
    return ip12 / denom if denom > 1e-10 else 0.0


def estimate_layer_input_covariance(model, dist_fn, layer_idx, n, num_samples=100000):
    """
    Estimate Cov[h_{layer_idx-1}] = covariance of the input to layer `layer_idx`.
      layer_idx=0 → Cov[x]   (raw input distribution)
      layer_idx=k → Cov[h_{k-1}]  (output after k-1 layers)
    """
    with torch.no_grad():
        x = dist_fn(num_samples, n)
        states = model.hidden_states(x)   # [x, h1, h2, ...]
    return torch.cov(states[layer_idx].T)


def estimate_covariance(dist_fn, n, num_samples=100000):
    x = dist_fn(num_samples, n)
    return torch.cov(x.T)


def functional_similarity(model1, model2, dist_fn, n, num_samples=50000):
    """Fraction of inputs where argmax predictions agree."""
    model1.eval()
    model2.eval()
    with torch.no_grad():
        x = dist_fn(num_samples, n)
        return (model1(x).argmax(1) == model2(x).argmax(1)).float().mean().item()
