"""Monte Carlo baselines for functional similarity.

``mc_inner_product_seq`` mirrors the single-moment estimator in
``tests/similarity.py``. ``mc_cosine_seq`` is the three-moment cosine on
the same shared sample stream — preferred whenever you want
``cosine = E[f_a·f_b] / sqrt(E[f_a·f_a] · E[f_b·f_b])`` rather than three
independent calls.
"""
import warnings

import torch


@torch.no_grad()
def mc_inner_product_seq(
    model_a, model_b, d_input, n_ctx,
    n_samples=200_000, batch_size=256, **like,
) -> float:
    """MC estimate of ``E[Σ_s f_a(x)_s · f_b(x)_s]`` for sequential models.

    ``x ~ N(0, I)`` has shape ``(batch_size, n_ctx, d_input)`` and is
    streamed in batches.
    """
    total = torch.zeros((), **like)
    done = 0
    while done < n_samples:
        bs = min(batch_size, n_samples - done)
        x = torch.randn(bs, n_ctx, d_input, **like)
        total += (model_a(x) * model_b(x)).sum(dim=(-1, -2)).sum()
        done += bs
    return (total / n_samples).item()


@torch.no_grad()
def mc_cosine_seq(
    model_a, model_b, d_input, n_ctx,
    n_samples=200_000, batch_size=256, **like,
) -> float:
    """MC cosine ``E[f_a·f_b] / sqrt(E[f_a·f_a] · E[f_b·f_b])``, all three
    moments accumulated on the same sample stream in one pass.
    """
    aa = torch.zeros((), **like)
    bb = torch.zeros((), **like)
    ab = torch.zeros((), **like)
    done = 0
    while done < n_samples:
        bs = min(batch_size, n_samples - done)
        x = torch.randn(bs, n_ctx, d_input, **like)
        fa, fb = model_a(x), model_b(x)
        aa += (fa * fa).sum(dim=(-1, -2)).sum()
        bb += (fb * fb).sum(dim=(-1, -2)).sum()
        ab += (fa * fb).sum(dim=(-1, -2)).sum()
        done += bs
    denom = (aa.abs() * bb.abs()).sqrt()
    if not torch.isfinite(denom) or denom.item() == 0.0:
        warnings.warn(
            f"mc_cosine_seq: denominator underflowed or was non-finite "
            f"(aa={aa.item():.3e}, bb={bb.item():.3e}). Returning nan.",
            RuntimeWarning, stacklevel=2,
        )
        return float("nan")
    return (ab / denom).item()
