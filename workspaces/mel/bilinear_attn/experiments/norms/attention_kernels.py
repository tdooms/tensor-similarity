"""Modified attention kernels with configurable Q/K normalization."""
import torch
import torch.nn as nn
from einops import rearrange, einsum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.attention_kernels.rotary import Rotary
from experiments.norms.qk_norms import QKNormWrapper


class BilinearAttentionNorm(nn.Module):
    """Bilinear attention with configurable Q/K normalization.
    
    Implements:
        scores1 = q1 @ k1.T
        scores2 = q2 @ k2.T
        pattern = (scores1 * scores2) / d_head^2 * causal_mask
        z = pattern @ v
        out = x + scale * Wo(z)
    
    Args:
        qk_norm_type: Type of Q/K normalization ('none', 'rmsnorm', 'alpha_head')
        alpha_init: Initial value for alpha parameters (if using alpha_head)
    """
    
    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_ctx: int,
        scale: float = 1.0,
        qk_norm_type: str = "none",
        alpha_init: float = 1.0,
        use_bias_qk: bool = True,
        rope_base: int = 10000,
    ) -> None:
        super().__init__()
        self.d_head = d_model // n_head
        self.n_head = n_head
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.scale = scale
        self.qk_norm_type = qk_norm_type
        
        self.rotary = Rotary(self.d_head, n_ctx, base=rope_base)
        self.norm_qk = QKNormWrapper(
            norm_type=qk_norm_type,
            d_head=self.d_head,
            n_head=n_head,
            alpha_init=alpha_init,
        )
        
        causal_mask = torch.tril(torch.ones(n_ctx, n_ctx))
        self.register_buffer("causal_mask", causal_mask, persistent=False)
        
        self.q1 = nn.Linear(d_model, d_model, bias=use_bias_qk)
        self.k1 = nn.Linear(d_model, d_model, bias=use_bias_qk)
        self.q2 = nn.Linear(d_model, d_model, bias=use_bias_qk)
        self.k2 = nn.Linear(d_model, d_model, bias=use_bias_qk)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x: torch.Tensor, return_debug: bool = False):
        """Forward pass with optional debug outputs."""
        B, T, _ = x.shape
        
        q1 = rearrange(self.q1(x), "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)
        k1 = rearrange(self.k1(x), "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)
        q2 = rearrange(self.q2(x), "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)
        k2 = rearrange(self.k2(x), "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)
        v = rearrange(self.v(x), "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)
        
        q1 = self.rotary(self.norm_qk.forward_q(q1))
        k1 = self.rotary(self.norm_qk.forward_k(k1))
        q2 = self.rotary(self.norm_qk.forward_q(q2))
        k2 = self.rotary(self.norm_qk.forward_k(k2))
        
        scores1 = einsum(
            q1, k1,
            "b seq_q n_head d_head, b seq_k n_head d_head -> b n_head seq_q seq_k",
        )
        scores2 = einsum(
            q2, k2,
            "b seq_q n_head d_head, b seq_k n_head d_head -> b n_head seq_q seq_k",
        )
        
        pattern = (scores1 * scores2) / self.d_head ** 2
        pattern = pattern * self.causal_mask[None, None, :T, :T]
        
        z = einsum(
            pattern, v,
            "b n_head seq_q seq_k, b seq_k n_head d_head -> b seq_q n_head d_head",
        )
        
        z_merge = rearrange(z, "b seq n_head d_head -> b seq (n_head d_head)")
        out = x + self.scale * self.o(z_merge)
        
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
                "z": z,
            }
            return out, debug
        return out


class QuadraticAttentionNorm(nn.Module):
    """Quadratic attention with configurable Q/K normalization.
    
    Implements:
        scores = q @ k.T
        pattern = (scores / d_head)^2 * causal_mask
        z = pattern @ v
        out = x + scale * Wo(z)
    """
    
    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_ctx: int,
        scale: float = 1.0,
        qk_norm_type: str = "none",
        alpha_init: float = 1.0,
        use_bias_qk: bool = True,
        rope_base: int = 10000,
    ) -> None:
        super().__init__()
        self.d_head = d_model // n_head
        self.n_head = n_head
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.scale = scale
        self.qk_norm_type = qk_norm_type
        
        self.rotary = Rotary(self.d_head, n_ctx, base=rope_base)
        self.norm_qk = QKNormWrapper(
            norm_type=qk_norm_type,
            d_head=self.d_head,
            n_head=n_head,
            alpha_init=alpha_init,
        )
        
        causal_mask = torch.tril(torch.ones(n_ctx, n_ctx))
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
        
        q = self.rotary(self.norm_qk.forward_q(q))
        k = self.rotary(self.norm_qk.forward_k(k))
        
        scores = einsum(
            q, k,
            "b seq_q n_head d_head, b seq_k n_head d_head -> b n_head seq_q seq_k",
        )
        
        pattern = (scores / self.d_head).square()
        pattern = pattern * self.causal_mask[None, None, :T, :T]
        
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


class SoftmaxAttentionNorm(nn.Module):
    """Softmax attention with configurable Q/K normalization.
    
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
        qk_norm_type: str = "none",
        alpha_init: float = 1.0,
        use_bias_qk: bool = True,
        rope_base: int = 10000,
    ) -> None:
        super().__init__()
        self.d_head = d_model // n_head
        self.n_head = n_head
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.scale = scale
        self.qk_norm_type = qk_norm_type
        
        self.rotary = Rotary(self.d_head, n_ctx, base=rope_base)
        self.norm_qk = QKNormWrapper(
            norm_type=qk_norm_type,
            d_head=self.d_head,
            n_head=n_head,
            alpha_init=alpha_init,
        )
        
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
        
        q = self.rotary(self.norm_qk.forward_q(q))
        k = self.rotary(self.norm_qk.forward_k(k))
        
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
