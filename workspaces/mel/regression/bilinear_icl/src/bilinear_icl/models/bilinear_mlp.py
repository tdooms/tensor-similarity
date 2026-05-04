import torch
from torch import nn


class BilinearMLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, scale: float = 0.35):
        super().__init__()
        self.scale = scale
        self.l = nn.Linear(d_model, d_mlp, bias=False)
        self.r = nn.Linear(d_model, d_mlp, bias=False)
        self.d = nn.Linear(d_mlp, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.lerp(x, self.d(self.l(x) * self.r(x)), self.scale)
