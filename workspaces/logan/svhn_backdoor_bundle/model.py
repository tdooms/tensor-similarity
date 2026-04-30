"""SVHN bilinear model.

Architecture: Bilinear(d_in=1024 → d_hidden) → Linear(d_hidden → 10).

Reuses the Bilinear layer from grokking_summary_bundle/core so the
symmetric TN inner-product machinery (which only relies on w_l/w_r/w_p
properties) works without modification.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor, nn

# Make grokking_summary_bundle.core importable.
_GROKKING = Path(__file__).parent.parent / "grokking_summary_bundle"
if str(_GROKKING) not in sys.path:
    sys.path.insert(0, str(_GROKKING))
from core.models import Bilinear  # noqa: E402

D_IN = 1024
N_CLASSES = 10


class SVHNBilinear(nn.Module):
    """Bilinear bottleneck classifier for grayscale 32x32 SVHN."""

    def __init__(self, d_hidden: int, bias: bool = False) -> None:
        super().__init__()
        self.d_hidden = d_hidden
        self.bi_linear = Bilinear(d_in=D_IN, d_out=d_hidden, bias=bias)
        self.projection = nn.Linear(d_hidden, N_CLASSES, bias=bias)

    def forward(self, x: Float[Tensor, "n 1024"]) -> Float[Tensor, "n 10"]:
        return self.projection(self.bi_linear(x))

    @property
    def w_l(self) -> Tensor:
        return self.bi_linear.w_l

    @property
    def w_r(self) -> Tensor:
        return self.bi_linear.w_r

    @property
    def w_p(self) -> Tensor:
        return self.projection.weight


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
