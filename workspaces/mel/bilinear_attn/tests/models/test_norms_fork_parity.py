"""Differential parity between production attention kernels and the
``experiments/norms/`` fork.

The fork (``BilinearAttentionNorm`` / ``QuadraticAttentionNorm`` /
``SoftmaxAttentionNorm``) exists to support the ``qk_norm_type`` ablation
study. With ``qk_norm_type='none'`` it **must** be functionally identical
to the production kernel — anything else means one side has drifted, which
is the class of bug that caused the original lerp regression.

If this test ever starts failing, the right fix is not to weaken the
assertion: decide whether the fork or production is correct, fix the
other side, and keep the test.
"""
from __future__ import annotations

import pytest
import torch

from models.attention_kernels.bilinear import BilinearAttention, QuadraticAttention
from models.attention_kernels.softmax import SoftmaxAttention
from experiments.norms.attention_kernels import (
    BilinearAttentionNorm,
    QuadraticAttentionNorm,
    SoftmaxAttentionNorm,
)


KERNEL_PAIRS = [
    (BilinearAttention, BilinearAttentionNorm, ("q1", "k1", "q2", "k2", "v", "o")),
    (QuadraticAttention, QuadraticAttentionNorm, ("q", "k", "v", "o")),
    (SoftmaxAttention, SoftmaxAttentionNorm, ("q", "k", "v", "o")),
]


@pytest.mark.parametrize(
    "prod_cls,fork_cls,weights",
    KERNEL_PAIRS,
    ids=lambda x: getattr(x, "__name__", str(x)),
)
@pytest.mark.parametrize("scale", [0.0, 0.2, 0.5, 1.0])
def test_fork_matches_production_when_qk_norm_none(prod_cls, fork_cls, weights, scale):
    torch.manual_seed(0)
    prod = prod_cls(
        d_model=16, n_head=4, n_ctx=8, scale=scale,
        use_rmsnorm_qk=False, use_bias_qk=True,
    ).double().eval()

    fork = fork_cls(
        d_model=16, n_head=4, n_ctx=8, scale=scale,
        qk_norm_type="none", use_bias_qk=True,
    ).double().eval()

    # Copy weights so the two kernels compute the exact same function.
    with torch.no_grad():
        for name in weights:
            getattr(fork, name).weight.copy_(getattr(prod, name).weight)
            if getattr(prod, name).bias is not None:
                getattr(fork, name).bias.copy_(getattr(prod, name).bias)

    torch.manual_seed(1)
    x = torch.randn(2, 8, 16, dtype=torch.float64)

    with torch.no_grad():
        out_prod = prod(x)
        out_fork = fork(x)

    torch.testing.assert_close(out_fork, out_prod, atol=1e-12, rtol=0.0)
