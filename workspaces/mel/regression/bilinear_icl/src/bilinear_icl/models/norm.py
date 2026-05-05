import torch
from torch import nn


NORM_TYPES = ("none", "tok0")


class BOSScalarNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = x[:, 0, :].pow(2).mean(dim=-1, keepdim=True)
        s = (e0 + self.eps).sqrt().unsqueeze(1)
        return x / s


def make_norm(norm_type: str, d_model: int, eps: float = 1e-6) -> nn.Module:
    assert norm_type in NORM_TYPES, f"norm_type must be one of {NORM_TYPES}, got {norm_type!r}"
    if norm_type == "none":
        return nn.Identity()
    return BOSScalarNorm(eps=eps)
