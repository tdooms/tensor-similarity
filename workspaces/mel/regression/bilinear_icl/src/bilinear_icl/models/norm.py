import torch
from torch import nn


class BOSScalarNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = x[:, 0, :].pow(2).mean(dim=-1, keepdim=True)
        s = (e0 + self.eps).sqrt().unsqueeze(1)
        return x / s
