import torch
import torch.nn as nn
from einops import rearrange, einsum
from .rotary import Rotary


class BilinearAttention(nn.Module):
    """Bilinear attention with two independent QK circuits and causal masking.
    
    Implements:
        scores1 = q1 @ k1.T
        scores2 = q2 @ k2.T
        pattern = (scores1 * scores2) / d_head^2 * causal_mask
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
        use_bias_qkv: bool = True,
        use_bias_o: bool = True,
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
        
        causal_mask = torch.tril(torch.ones(n_ctx, n_ctx))
        self.register_buffer("causal_mask", causal_mask, persistent=False)
        
        self.q1 = nn.Linear(d_model, d_model, bias=use_bias_qkv)
        self.k1 = nn.Linear(d_model, d_model, bias=use_bias_qkv)
        self.q2 = nn.Linear(d_model, d_model, bias=use_bias_qkv)
        self.k2 = nn.Linear(d_model, d_model, bias=use_bias_qkv)
        self.v = nn.Linear(d_model, d_model, bias=use_bias_qkv)
        self.o = nn.Linear(d_model, d_model, bias=use_bias_o)
    
    def forward(self, x: torch.Tensor, return_debug: bool = False):
        """Forward pass with optional debug outputs.
        
        Args:
            x: Input tensor of shape (B, T, d_model)
            return_debug: If True, return debug dict with intermediates
            
        Returns:
            out: Output tensor of shape (B, T, d_model)
            debug (optional): Dict with q1, k1, q2, k2, v, scores1, scores2, pattern, z
        """
        B, T, _ = x.shape
        
        q1 = rearrange(self.q1(x), "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)
        k1 = rearrange(self.k1(x), "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)
        q2 = rearrange(self.q2(x), "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)
        k2 = rearrange(self.k2(x), "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)
        v = rearrange(self.v(x), "b t (n_head d_head) -> b t n_head d_head", n_head=self.n_head)
        
        q1 = self.rotary(self.norm_qk(q1))
        k1 = self.rotary(self.norm_qk(k1))
        q2 = self.rotary(self.norm_qk(q2))
        k2 = self.rotary(self.norm_qk(k2))
        
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


class QuadraticAttention(nn.Module):
    """Quadratic (polynomial) attention with causal masking.
    
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
        use_rmsnorm_qk: bool = False,
        use_bias_qkv: bool = True,
        use_bias_o: bool = True,
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
        
        causal_mask = torch.tril(torch.ones(n_ctx, n_ctx))
        self.register_buffer("causal_mask", causal_mask, persistent=False)
        
        self.q = nn.Linear(d_model, d_model, bias=use_bias_qkv)
        self.k = nn.Linear(d_model, d_model, bias=use_bias_qkv)
        self.v = nn.Linear(d_model, d_model, bias=use_bias_qkv)
        self.o = nn.Linear(d_model, d_model, bias=use_bias_o)
    
    def forward(self, x: torch.Tensor, return_debug: bool = False):
        """Forward pass with optional debug outputs.
        
        Args:
            x: Input tensor of shape (B, T, d_model)
            return_debug: If True, return debug dict with intermediates
            
        Returns:
            out: Output tensor of shape (B, T, d_model)
            debug (optional): Dict with q, k, v, scores, pattern, z
        """
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
