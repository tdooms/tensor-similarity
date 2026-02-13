import torch
import torch.nn as nn


class Rotary(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, n_ctx: int, base: int = 10000) -> None:
        super().__init__()
        freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        ctx = torch.arange(n_ctx).float()
        freqs = torch.einsum("i,j->ij", ctx, freq)
        cos = freqs.cos()
        sin = freqs.sin()
        self.register_buffer("cos_cached", torch.cat([cos, cos], dim=-1)[None, :, None, :], persistent=False)
        self.register_buffer("sin_cached", torch.cat([sin, sin], dim=-1)[None, :, None, :], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        a, b = x.chunk(2, dim=-1)
        y = torch.cat((-b, a), dim=-1)
        return (x * self.cos_cached[:, :seq_len]) + (y * self.sin_cached[:, :seq_len])
