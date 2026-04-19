import torch
from torch import nn
from quimb.tensor import Tensor, TensorNetwork

from src.components.base import Term


class Linear(nn.Linear):
    """A component wrapper around nn.Linear."""
    def __init__(self, d_in: int, d_out: int, bias: bool = False) -> None:
        super().__init__(d_in, d_out, bias=bias)

    def _like(self):
        return dict(device=self.weight.device, dtype=self.weight.dtype)

    def network(self):
        assert self.bias is None, "Linear network representation does not support bias (yet)"
        w = torch.block_diag(torch.eye(1, **self._like()), self.weight)
        return TensorNetwork([Tensor(w, inds=('out:d', 'in:d0'), tags=('E',))])

    def terms(self, n_ctx):
        return [Term(self.network(), {'in:d0': 'out:s'})]
