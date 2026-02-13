"""Softmax attention baseline for comparison."""
import torch
import torch.nn as nn
from einops import rearrange, einsum
from .rotary import Rotary


class SoftmaxAttention(nn.Module):
    """Standard softmax attention with causal masking (baseline).
    
    Implements:
        scores = q @ k.T / sqrt(d_head)
        pattern = softmax(scores + causal_mask)
        z = pattern @ v
        out = x + scale * Wo(z)
    """
    
    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_ctx: int,
        scale: float = 1.0,
        use_rmsnorm_qk: bool = False,
        use_bias_qk: bool = True,
        rope_base: int = 10000,
    ) -> None:
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
        
        self.q = nn.Linear(d_model, d_model, bias=use_bias_qk)
        self.k = nn.Linear(d_model, d_model, bias=use_bias_qk)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x: torch.Tensor, return_debug: bool = False):
        """Forward pass with optional debug outputs."""
        B, T, _ = x.shape
        
        q = rearrange(self.q(x), "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)
        k = rearrange(self.k(x), "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)
        v = rearrange(self.v(x), "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)
        
        q = self.rotary(self.norm_qk(q))
        k = self.rotary(self.norm_qk(k))
        
        scores = einsum(
            q, k,
            "b seq_q n_head d_head, b seq_k n_head d_head -> b n_head seq_q seq_k",
        )
        
        scores = scores / (self.d_head ** 0.5)
        scores = scores + self.causal_mask[None, None, :T, :T]
        pattern = torch.softmax(scores, dim=-1)
        
        z = einsum(
            pattern, v,
            "b n_head seq_q seq_k, b seq_k n_head d_head -> b seq_q n_head d_head",
        )
        
        z_merge = rearrange(z, "b seq n_head d_head -> b seq (n_head d_head)")
        out = x + self.scale * self.o(z_merge)
        
        if return_debug:
            debug = {
                "q": q,
                "k": k,
                "v": v,
                "scores": scores,
                "pattern": pattern,
                "z": z,
            }
            return out, debug
        return out
