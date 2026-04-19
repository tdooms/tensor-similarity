"""Empirical similarity measures for bilinear attention models.

MC similarity samples Gaussian residual-stream inputs (torch.randn) and
computes cosine similarity from model logits. This mirrors Logan's baseline:
the samples are applied post-embed (residual stream), then the attention stack
+ norms + unembedding are evaluated explicitly.

Random similarity instead samples discrete-uniform token IDs over the vocab
and uses the full model forward pass.

For cosine similarity:
  sim(A, B) = <T_A, T_B> / sqrt(<T_A, T_A> * <T_B, T_B>)
"""

import torch
import numpy as np


def _forward_from_embeddings(model, x: torch.Tensor) -> torch.Tensor:
    """Run a model forward starting from residual-stream embeddings.

    This bypasses token embedding but still applies any post-embed norms,
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


@torch.no_grad()
def mc_similarity_gaussian_tokens(
    model_A,
    model_B,
    device,
    n_samples: int = 20000,
    batch_size: int = 64,
) -> float:
    """MC cosine similarity matched to the TN algorithm's input distribution.

    The exact TN similarity assumes the first component's input is Gaussian:
    for AttentionLM that's a vector in ``R^{vocab_size + 1}`` with a leading
    constant-1 axis and the remaining entries ``~ N(0, I)``. This function
    samples at exactly that level and propagates through mel's actual forward
    (residual-add), so it's a fair empirical baseline for ``cosine_similarity``.

    In contrast, ``mc_similarity`` below samples Gaussians at the residual
    stream (bypassing the embedding), which is a different function and
    will not agree with the TN result for a non-trivial embedding.
    """
    model_A.eval(); model_B.eval()
    dtype = next(model_A.parameters()).dtype
    V, D = model_A.vocab_size, model_A.d_model
    n_ctx = model_A.n_ctx

    ip_AB = ip_AA = ip_BB = 0.0
    total_count = 0
    for start in range(0, n_samples, batch_size):
        bs = min(batch_size, n_samples - start)
        # z[..., 0] = 1 (constant axis); z[..., 1:] ~ N(0, I).
        z = torch.empty(bs, n_ctx, V + 1, device=device, dtype=dtype).normal_()
        z[..., 0] = 1.0

        # Apply the padded embedding manually: mel's nn.Embedding has no bias,
        # so only the z[..., 1:] slice contributes to the residual stream,
        # via x = z[..., 1:] @ E (where E is model.embed.weight, shape (V, D)).
        x_A = z[..., 1:] @ model_A.embed.weight
        x_B = z[..., 1:] @ model_B.embed.weight

        for layer in model_A.layers:
            x_A = layer(x_A)
        for layer in model_B.layers:
            x_B = layer(x_B)

        logits_A = model_A.unembed(x_A)
        logits_B = model_B.unembed(x_B)

        flat_A = logits_A.reshape(-1, logits_A.shape[-1])
        flat_B = logits_B.reshape(-1, logits_B.shape[-1])
        ip_AB += (flat_A * flat_B).sum().item()
        ip_AA += (flat_A * flat_A).sum().item()
        ip_BB += (flat_B * flat_B).sum().item()
        total_count += flat_A.shape[0]

    ip_AB /= total_count; ip_AA /= total_count; ip_BB /= total_count
    denom = (abs(ip_AA) * abs(ip_BB)) ** 0.5
    return 0.0 if denom < 1e-30 else ip_AB / denom


@torch.no_grad()
def mc_similarity(model_A, model_B, device, n_samples=2000, batch_size=64):
    """Compute MC cosine similarity using Gaussian residual-stream samples.

    Args:
        model_A/model_B: AttentionLM models on the same device.
        device: torch device for sampling (kept explicit to avoid mismatches).
        n_samples: number of sequence samples (each sample has length n_ctx).
        batch_size: number of sequences per MC batch.
    """
    model_A.eval()
    model_B.eval()

    dtype = next(model_A.parameters()).dtype
    n_ctx = model_A.n_ctx
    d_model = model_A.d_model

    ip_AB, ip_AA, ip_BB = 0.0, 0.0, 0.0
    total_count = 0
    total_samples = 0

    n_batches = (n_samples + batch_size - 1) // batch_size

    for _ in range(n_batches):
        bs = min(batch_size, n_samples - total_samples)
        if bs <= 0:
            break

        x = torch.randn(bs, n_ctx, d_model, device=device, dtype=dtype)

        logits_A = _forward_from_embeddings(model_A, x)
        logits_B = _forward_from_embeddings(model_B, x)

        B, T, V = logits_A.shape
        flat_A = logits_A.reshape(-1, V)
        flat_B = logits_B.reshape(-1, V)

        ip_AB += (flat_A * flat_B).sum().item()
        ip_AA += (flat_A * flat_A).sum().item()
        ip_BB += (flat_B * flat_B).sum().item()
        total_count += B * T
        total_samples += bs

    ip_AB /= max(1, total_count)
    ip_AA /= max(1, total_count)
    ip_BB /= max(1, total_count)

    denom = np.sqrt(abs(ip_AA) * abs(ip_BB))
    if denom < 1e-30:
        return 0.0
    return ip_AB / denom


@torch.no_grad()
def random_sim(model_A, model_B, device, n_samples=2000, batch_size=64):
    """Compute cosine similarity using discrete-uniform token sampling.

    Args:
        model_A/model_B: AttentionLM models on the same device.
        device: torch device for sampling (kept explicit to avoid mismatches).
        n_samples: number of sequence samples (each sample has length n_ctx).
        batch_size: number of sequences per batch.
    """
    model_A.eval()
    model_B.eval()

    vocab_size = model_A.vocab_size
    n_ctx = model_A.n_ctx

    ip_AB, ip_AA, ip_BB = 0.0, 0.0, 0.0
    total_count = 0
    total_samples = 0

    n_batches = (n_samples + batch_size - 1) // batch_size

    for _ in range(n_batches):
        bs = min(batch_size, n_samples - total_samples)
        if bs <= 0:
            break

        input_ids = torch.randint(0, vocab_size, (bs, n_ctx), device=device)

        logits_A = model_A(input_ids)
        logits_B = model_B(input_ids)

        B, T, V = logits_A.shape
        flat_A = logits_A.reshape(-1, V)
        flat_B = logits_B.reshape(-1, V)

        ip_AB += (flat_A * flat_B).sum().item()
        ip_AA += (flat_A * flat_A).sum().item()
        ip_BB += (flat_B * flat_B).sum().item()
        total_count += B * T
        total_samples += bs

    ip_AB /= max(1, total_count)
    ip_AA /= max(1, total_count)
    ip_BB /= max(1, total_count)

    denom = np.sqrt(abs(ip_AA) * abs(ip_BB))
    if denom < 1e-30:
        return 0.0
    return ip_AB / denom
