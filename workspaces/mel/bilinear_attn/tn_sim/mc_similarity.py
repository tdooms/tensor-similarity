"""Empirical similarity measures for bilinear attention models.

MC similarity samples Gaussian residual-stream inputs (torch.randn) and
computes cosine similarity from model logits. This mirrors Logan's baseline:
the samples are applied post-embed (residual stream), then the attention
stack + norms + unembedding are evaluated explicitly.

Random similarity instead samples discrete-uniform token IDs over the vocab
and uses the full model forward pass.

For cosine similarity:
  sim(A, B) = <T_A, T_B> / sqrt(<T_A, T_A> * <T_B, T_B>)

Numerical notes
---------------
Quadratic/bilinear attention produces degree-4 polynomials per layer, so an
n-layer stack is degree-(4^n) in the input. On *trained* weights this makes
the distribution of ||f(x)||^2 very heavy-tailed: individual MC samples can
differ by many orders of magnitude, and fp32 summation overflows long before
the mean stabilises. All three estimators below therefore:

  * run the forward + accumulators in a configurable `dtype` (default fp64),
  * accumulate on-device in tensors (not Python floats) to preserve
    precision and avoid silent fp32 sum overflow,
  * return `nan` (with a RuntimeWarning) when the denominator underflows,
    instead of silently returning 0.0.

On quadratic/bilinear stacks with trained weights, prefer TN similarity
(`tn_sim.cosine_similarity`), which computes the exact Gaussian cosine in
closed form via second-moment propagation and is not subject to MC variance.
"""

import copy
import warnings

import numpy as np
import torch

from src.components.mc import mc_cosine_seq
from models.components import AttentionLMMCWrapper


def _as_dtype(model, dtype):
    """Return `model` cast to `dtype`, copying iff dtypes differ.

    The copy is deep so we don't mutate the caller's model. For tiny
    research models this is cheap; for larger ones, callers who care
    should pass an already-cast model.
    """
    if next(model.parameters()).dtype == dtype:
        return model
    return copy.deepcopy(model).to(dtype=dtype)


def _forward_from_embeddings(model, x: torch.Tensor) -> torch.Tensor:
    """Run a model forward starting from residual-stream embeddings.

    Bypasses token embedding but still applies any post-embed norms,
    per-layer norms, final norm, and unembedding.
    """
    embed_norm = getattr(model, "embed_norm", None)
    if embed_norm is not None:
        x = embed_norm(x)

    layer_norms = getattr(model, "layer_norms", None)
    if layer_norms is not None:
        for norm, layer in zip(layer_norms, model.layers):
            x_normed = norm(x)
            out = layer(x_normed)
            x = x + (out - x_normed)
    else:
        for layer in model.layers:
            x = layer(x)

    final_norm = getattr(model, "final_norm", None)
    if final_norm is not None:
        x = final_norm(x)
    return model.unembed(x)


def _finalise_cosine(ip_AB: torch.Tensor, ip_AA: torch.Tensor, ip_BB: torch.Tensor) -> float:
    """Turn on-device fp-accumulated inner products into a cosine scalar.

    Returns `nan` (with a RuntimeWarning) if the denominator underflows to 0,
    rather than silently returning 0.0, so callers can distinguish genuine
    orthogonality (|ip_AB| ~ denom ~ small) from numerical failure.
    """
    denom = torch.sqrt(ip_AA.abs() * ip_BB.abs())
    if not torch.isfinite(denom) or denom.item() == 0.0:
        warnings.warn(
            f"MC cosine denominator underflowed or was non-finite "
            f"(ip_AA={ip_AA.item():.3e}, ip_BB={ip_BB.item():.3e}). "
            f"Returning nan. This usually indicates the model's outputs "
            f"blew up on the MC input distribution; prefer TN similarity.",
            RuntimeWarning,
            stacklevel=3,
        )
        return float("nan")
    return (ip_AB / denom).item()


@torch.no_grad()
def mc_similarity_gaussian_tokens(
    model_A,
    model_B,
    device,
    n_samples: int = 20000,
    batch_size: int = 256,
    dtype: torch.dtype = torch.float64,
) -> float:
    """MC cosine via ``src.components.mc.mc_cosine_seq`` on mel models
    wrapped to match main's "first Linear is the embedding" convention.
    """
    model_A = _as_dtype(model_A, dtype).eval()
    model_B = _as_dtype(model_B, dtype).eval()
    return mc_cosine_seq(
        AttentionLMMCWrapper(model_A),
        AttentionLMMCWrapper(model_B),
        d_input=model_A.vocab_size,
        n_ctx=model_A.n_ctx,
        n_samples=n_samples,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )


@torch.no_grad()
def mc_similarity(
    model_A,
    model_B,
    device,
    n_samples: int = 2000,
    batch_size: int = 256,
    dtype: torch.dtype = torch.float64,
) -> float:
    """MC cosine similarity using Gaussian residual-stream samples.

    Args:
        model_A/model_B: AttentionLM models on the same device.
        device: torch device for sampling (kept explicit to avoid mismatches).
        n_samples: number of sequence samples (each sample has length n_ctx).
        batch_size: number of sequences per MC batch.
        dtype: dtype used for sampling, forward, and accumulation. Defaults
            to float64 to control MC estimator variance on deep
            bilinear/quadratic stacks.
    """
    model_A = _as_dtype(model_A, dtype).eval()
    model_B = _as_dtype(model_B, dtype).eval()
    n_ctx = model_A.n_ctx
    d_model = model_A.d_model

    ip_AB = torch.zeros((), device=device, dtype=dtype)
    ip_AA = torch.zeros((), device=device, dtype=dtype)
    ip_BB = torch.zeros((), device=device, dtype=dtype)
    total = 0
    done = 0
    while done < n_samples:
        bs = min(batch_size, n_samples - done)
        x = torch.randn(bs, n_ctx, d_model, device=device, dtype=dtype)

        logits_A = _forward_from_embeddings(model_A, x)
        logits_B = _forward_from_embeddings(model_B, x)

        V = logits_A.shape[-1]
        flat_A = logits_A.reshape(-1, V)
        flat_B = logits_B.reshape(-1, V)

        ip_AB += (flat_A * flat_B).sum()
        ip_AA += (flat_A * flat_A).sum()
        ip_BB += (flat_B * flat_B).sum()
        total += flat_A.shape[0]
        done += bs

    ip_AB = ip_AB / max(1, total)
    ip_AA = ip_AA / max(1, total)
    ip_BB = ip_BB / max(1, total)
    return _finalise_cosine(ip_AB, ip_AA, ip_BB)


@torch.no_grad()
def random_sim(
    model_A,
    model_B,
    device,
    n_samples: int = 2000,
    batch_size: int = 256,
    dtype: torch.dtype = torch.float64,
) -> float:
    """Cosine similarity under discrete-uniform token sampling.

    The inputs are bounded by the embedding, so numerics are much tamer than
    the Gaussian variants; the dtype default still matches for consistency.
    """
    model_A = _as_dtype(model_A, dtype).eval()
    model_B = _as_dtype(model_B, dtype).eval()
    vocab_size = model_A.vocab_size
    n_ctx = model_A.n_ctx

    ip_AB = torch.zeros((), device=device, dtype=dtype)
    ip_AA = torch.zeros((), device=device, dtype=dtype)
    ip_BB = torch.zeros((), device=device, dtype=dtype)
    total = 0
    done = 0
    while done < n_samples:
        bs = min(batch_size, n_samples - done)
        input_ids = torch.randint(0, vocab_size, (bs, n_ctx), device=device)

        logits_A = model_A(input_ids)
        logits_B = model_B(input_ids)

        V = logits_A.shape[-1]
        flat_A = logits_A.reshape(-1, V)
        flat_B = logits_B.reshape(-1, V)

        ip_AB += (flat_A * flat_B).sum()
        ip_AA += (flat_A * flat_A).sum()
        ip_BB += (flat_B * flat_B).sum()
        total += flat_A.shape[0]
        done += bs

    ip_AB = ip_AB / max(1, total)
    ip_AA = ip_AA / max(1, total)
    ip_BB = ip_BB / max(1, total)
    return _finalise_cosine(ip_AB, ip_AA, ip_BB)
