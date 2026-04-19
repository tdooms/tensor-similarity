"""Quadratic attention math matches a naive reference implementation."""
import torch

from models.attention_kernels.bilinear import QuadraticAttention
from tests.models.conftest import B, T, D_MODEL, N_HEAD, N_CTX, ATOL, RTOL


def _naive(q, k, v, d_head):
    """Explicit-loop reference for pattern and z."""
    B_, T_, H_, _ = q.shape
    pattern = torch.zeros(B_, H_, T_, T_)
    z = torch.zeros_like(v)
    for b in range(B_):
        for h in range(H_):
            for t in range(T_):
                for s in range(t + 1):
                    score = (q[b, t, h] * k[b, s, h]).sum() / d_head
                    pattern[b, h, t, s] = score ** 2
                for i in range(v.shape[-1]):
                    z[b, t, h, i] = (pattern[b, h, t, :] * v[b, :, h, i]).sum()
    return pattern, z


def test_quadratic_attention_matches_reference():
    """Pattern, z, and the scaling law all match the naive reference."""
    d_head = D_MODEL // N_HEAD
    attn = QuadraticAttention(
        d_model=D_MODEL, n_head=N_HEAD, n_ctx=N_CTX,
        scale=0.2, use_rmsnorm_qk=False,
    )

    torch.manual_seed(42)
    x = torch.randn(B, T, D_MODEL)
    _, debug = attn(x, return_debug=True)

    pattern_ref, z_ref = _naive(debug["q"], debug["k"], debug["v"], d_head)
    torch.testing.assert_close(debug["pattern"], pattern_ref, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(debug["z"], z_ref, atol=ATOL, rtol=RTOL)

    # Scaling law: pattern = (scores/d_head)^2 * causal_mask
    mask = torch.tril(torch.ones(T, T))[None, None]
    expected = (debug["scores"] / d_head).square() * mask
    torch.testing.assert_close(debug["pattern"], expected, atol=ATOL, rtol=RTOL)
