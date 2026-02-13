"""All attention kernels: softmax, quadratic, bilinear, cubic."""
import torch
import torch.nn as nn
from einops import rearrange, einsum
from .rotary import Rotary


class SoftmaxAttention(nn.Module):
    """Standard softmax attention with causal masking."""

    def __init__(self, d_model, n_head, n_ctx, scale=1.0, use_rmsnorm_qk=False,
                 use_bias_qkv=True, use_bias_o=True, rope_base=10000):
        super().__init__()
        self.d_head = d_model // n_head
        self.n_head = n_head
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.scale = scale

        self.rotary = Rotary(self.d_head, n_ctx, base=rope_base)
        self.norm_qk = nn.RMSNorm(self.d_head) if use_rmsnorm_qk else nn.Identity()

        causal_mask = torch.triu(torch.full((n_ctx, n_ctx), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        self.q = nn.Linear(d_model, d_model, bias=use_bias_qkv)
        self.k = nn.Linear(d_model, d_model, bias=use_bias_qkv)
        self.v = nn.Linear(d_model, d_model, bias=use_bias_qkv)
        self.o = nn.Linear(d_model, d_model, bias=use_bias_o)

    def forward(self, x, return_debug=False):
        B, T, _ = x.shape
        q = rearrange(self.q(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        k = rearrange(self.k(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        v = rearrange(self.v(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)

        q = self.rotary(self.norm_qk(q))
        k = self.rotary(self.norm_qk(k))

        scores = einsum(q, k, "b sq nh dh, b sk nh dh -> b nh sq sk")
        scores = scores / (self.d_head ** 0.5)
        scores = scores + self.causal_mask[None, None, :T, :T]
        pattern = torch.softmax(scores, dim=-1)

        z = einsum(pattern, v, "b nh sq sk, b sk nh dh -> b sq nh dh")
        z_merge = rearrange(z, "b s nh dh -> b s (nh dh)")
        out = x + self.scale * self.o(z_merge)

        if return_debug:
            return out, {"q": q, "k": k, "v": v, "scores": scores, "pattern": pattern, "z": z}
        return out


class QuadraticAttention(nn.Module):
    """Quadratic attention: (q·k/d)^2 with causal masking."""

    def __init__(self, d_model, n_head, n_ctx, scale=1.0, use_rmsnorm_qk=False,
                 use_bias_qkv=True, use_bias_o=True, rope_base=10000):
        super().__init__()
        self.d_head = d_model // n_head
        self.n_head = n_head
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.scale = scale

        self.rotary = Rotary(self.d_head, n_ctx, base=rope_base)
        self.norm_qk = nn.RMSNorm(self.d_head) if use_rmsnorm_qk else nn.Identity()

        causal_mask = torch.tril(torch.ones(n_ctx, n_ctx))
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        self.q = nn.Linear(d_model, d_model, bias=use_bias_qkv)
        self.k = nn.Linear(d_model, d_model, bias=use_bias_qkv)
        self.v = nn.Linear(d_model, d_model, bias=use_bias_qkv)
        self.o = nn.Linear(d_model, d_model, bias=use_bias_o)

    def forward(self, x, return_debug=False):
        B, T, _ = x.shape
        q = rearrange(self.q(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        k = rearrange(self.k(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        v = rearrange(self.v(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)

        q = self.rotary(self.norm_qk(q))
        k = self.rotary(self.norm_qk(k))

        scores = einsum(q, k, "b sq nh dh, b sk nh dh -> b nh sq sk")
        pattern = (scores / self.d_head).square()
        pattern = pattern * self.causal_mask[None, None, :T, :T]

        z = einsum(pattern, v, "b nh sq sk, b sk nh dh -> b sq nh dh")
        z_merge = rearrange(z, "b s nh dh -> b s (nh dh)")
        out = x + self.scale * self.o(z_merge)

        if return_debug:
            return out, {"q": q, "k": k, "v": v, "scores": scores, "pattern": pattern, "z": z}
        return out


class BilinearAttention(nn.Module):
    """Bilinear attention: (q1·k1)*(q2·k2)/d^2 with causal masking."""

    def __init__(self, d_model, n_head, n_ctx, scale=1.0, use_rmsnorm_qk=False,
                 use_bias_qkv=True, use_bias_o=True, rope_base=10000):
        super().__init__()
        self.d_head = d_model // n_head
        self.n_head = n_head
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.scale = scale

        self.rotary = Rotary(self.d_head, n_ctx, base=rope_base)
        self.norm_qk = nn.RMSNorm(self.d_head) if use_rmsnorm_qk else nn.Identity()

        causal_mask = torch.tril(torch.ones(n_ctx, n_ctx))
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        self.q = nn.Linear(d_model, d_model, bias=use_bias_qkv)
        self.k = nn.Linear(d_model, d_model, bias=use_bias_qkv)
        self.q2 = nn.Linear(d_model, d_model, bias=use_bias_qkv)
        self.k2 = nn.Linear(d_model, d_model, bias=use_bias_qkv)
        self.v = nn.Linear(d_model, d_model, bias=use_bias_qkv)
        self.o = nn.Linear(d_model, d_model, bias=use_bias_o)

    def forward(self, x, return_debug=False):
        B, T, _ = x.shape
        q1 = rearrange(self.q(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        k1 = rearrange(self.k(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        q2 = rearrange(self.q2(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        k2 = rearrange(self.k2(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        v = rearrange(self.v(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)

        q1 = self.rotary(self.norm_qk(q1))
        k1 = self.rotary(self.norm_qk(k1))
        q2 = self.rotary(self.norm_qk(q2))
        k2 = self.rotary(self.norm_qk(k2))

        scores1 = einsum(q1, k1, "b sq nh dh, b sk nh dh -> b nh sq sk")
        scores2 = einsum(q2, k2, "b sq nh dh, b sk nh dh -> b nh sq sk")

        pattern = (scores1 * scores2) / (self.d_head ** 2)
        pattern = pattern * self.causal_mask[None, None, :T, :T]

        z = einsum(pattern, v, "b nh sq sk, b sk nh dh -> b sq nh dh")
        z_merge = rearrange(z, "b s nh dh -> b s (nh dh)")
        out = x + self.scale * self.o(z_merge)

        if return_debug:
            return out, {"q1": q1, "k1": k1, "q2": q2, "k2": k2, "v": v,
                         "scores1": scores1, "scores2": scores2, "pattern": pattern, "z": z}
        return out


class CubicAttention(nn.Module):
    """Cubic attention: (QK^T/sqrt(d_k))^3 / sqrt(N) with causal masking."""

    def __init__(self, d_model, n_head, n_ctx, scale=1.0, use_rmsnorm_qk=False,
                 use_bias_qkv=True, use_bias_o=True, rope_base=10000):
        super().__init__()
        self.d_head = d_model // n_head
        self.n_head = n_head
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.scale = scale

        self.rotary = Rotary(self.d_head, n_ctx, base=rope_base)
        self.norm_qk = nn.RMSNorm(self.d_head) if use_rmsnorm_qk else nn.Identity()

        causal_mask = torch.tril(torch.ones(n_ctx, n_ctx))
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        self.q = nn.Linear(d_model, d_model, bias=use_bias_qkv)
        self.k = nn.Linear(d_model, d_model, bias=use_bias_qkv)
        self.v = nn.Linear(d_model, d_model, bias=use_bias_qkv)
        self.o = nn.Linear(d_model, d_model, bias=use_bias_o)

    def forward(self, x, return_debug=False):
        B, T, _ = x.shape
        q = rearrange(self.q(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        k = rearrange(self.k(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        v = rearrange(self.v(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)

        q = self.rotary(self.norm_qk(q))
        k = self.rotary(self.norm_qk(k))

        scores = einsum(q, k, "b sq nh dh, b sk nh dh -> b nh sq sk")
        normalized = scores / (self.d_head ** 0.5)
        pattern = normalized.pow(3) / (T ** 0.5)
        pattern = pattern * self.causal_mask[None, None, :T, :T]

        z = einsum(pattern, v, "b nh sq sk, b sk nh dh -> b sq nh dh")
        z_merge = rearrange(z, "b s nh dh -> b s (nh dh)")
        out = x + self.scale * self.o(z_merge)

        if return_debug:
            return out, {"q": q, "k": k, "v": v, "scores": scores, "pattern": pattern, "z": z}
        return out
