import torch

from bilinear_icl.models._kernels.attention_kernels import SoftmaxAttention


def test_softmax_attention_shapes_and_causal_mask():
    attn = SoftmaxAttention(d_model=32, n_head=4, n_ctx=17, scale=0.35, use_bias_qk=False)
    x = torch.randn(2, 17, 32)
    out, dbg = attn(x, return_debug=True)

    assert out.shape == x.shape
    pattern = dbg["pattern"]
    upper = torch.triu(torch.ones(17, 17, dtype=torch.bool), diagonal=1)
    assert torch.allclose(pattern[..., upper], torch.zeros_like(pattern[..., upper]))


def test_softmax_attention_backward():
    attn = SoftmaxAttention(d_model=32, n_head=4, n_ctx=17, scale=0.35, use_bias_qk=False)
    x = torch.randn(2, 17, 32, requires_grad=True)
    out = attn(x)
    loss = out.pow(2).mean()
    loss.backward()
    assert x.grad is not None
