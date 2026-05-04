import torch
import torch.nn as nn
from einops import einsum, rearrange

from .rotary import Rotary


class BilinearAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_ctx: int,
        scale: float = 1.0,
        use_bias_qk: bool = False,
        rope_base: int = 10000,
    ) -> None:
        super().__init__()
        self.d_head = d_model // n_head
        self.n_head = n_head
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.scale = scale

        self.rotary = Rotary(self.d_head, n_ctx, base=rope_base)

        causal_mask = torch.tril(torch.ones(n_ctx, n_ctx))
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        self.q1 = nn.Linear(d_model, d_model, bias=use_bias_qk)
        self.k1 = nn.Linear(d_model, d_model, bias=use_bias_qk)
        self.q2 = nn.Linear(d_model, d_model, bias=use_bias_qk)
        self.k2 = nn.Linear(d_model, d_model, bias=use_bias_qk)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        _, t, _ = x.shape

        q1 = rearrange(self.q1(x), "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)
        k1 = rearrange(self.k1(x), "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)
        q2 = rearrange(self.q2(x), "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)
        k2 = rearrange(self.k2(x), "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)
        v = rearrange(self.v(x), "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)

        q1 = self.rotary(q1)
        k1 = self.rotary(k1)
        q2 = self.rotary(q2)
        k2 = self.rotary(k2)

        scores1 = einsum(
            q1,
            k1,
            "b seq_q n_head d_head, b seq_k n_head d_head -> b n_head seq_q seq_k",
        )
        scores2 = einsum(
            q2,
            k2,
            "b seq_q n_head d_head, b seq_k n_head d_head -> b n_head seq_q seq_k",
        )

        pattern = (scores1 * scores2) / (self.d_head ** 2)
        pattern = pattern * self.causal_mask[None, None, :t, :t]

        z = einsum(
            pattern,
            v,
            "b n_head seq_q seq_k, b seq_k n_head d_head -> b seq_q n_head d_head",
        )

        z_merge = rearrange(z, "b seq n_head d_head -> b seq (n_head d_head)")
        out = torch.lerp(x, self.o(z_merge), self.scale)

        if return_debug:
            debug = {
                "q1": q1,
                "k1": k1,
                "q2": q2,
                "k2": k2,
                "v": v,
                "scores1": scores1,
                "scores2": scores2,
                "pattern": pattern,
                "A_signed": pattern,
                "z": z,
            }
            return out, debug
        return out
