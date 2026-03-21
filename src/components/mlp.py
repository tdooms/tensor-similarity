
from torch import nn
from quimb.tensor import Tensor, TensorNetwork

from src.components.base import Component
from src.components.compose import pad

import torch


class MLP(Component):
    """The canonical residual + bilinear layer."""
    def __init__(self, d_model: int, d_hidden: int, bias: bool = False, scale: float = 1) -> None:
        super().__init__()
        
        self.scale = scale  # scale=1 is no residual, scale=0 is pure residual.
        self.d_model, self.d_hidden = d_model, d_hidden
        
        self.l = nn.Linear(d_model, d_hidden, bias=False)
        self.r = nn.Linear(d_model, d_hidden, bias=False)
        self.d = nn.Linear(d_hidden, d_model, bias=bias)

    def forward(self, x):
        return torch.lerp(x, self.d(self.l(x) * self.r(x)), self.scale)
    
    def _like(self):
        return dict(device=self.d.weight.device, dtype=self.d.weight.dtype)

    def network(self):
        dim, like = self.d_model, self._like()
        
        residual = torch.cat([torch.zeros(dim, 1, **like), torch.eye(dim, **like)], dim=1)
        constant = torch.cat([torch.ones(dim, 1, **like), torch.zeros(dim, dim, **like)], dim=1)
        
        l = torch.cat([pad(self.l.weight, self.l.bias), residual], dim=0)
        r = torch.cat([pad(self.r.weight, self.r.bias), constant], dim=0)

        res = torch.cat([torch.zeros(1, dim, **like), torch.eye(dim, **like)], dim=0)
        d = torch.cat([pad(self.d.weight, self.d.bias, scale=self.scale), (1 - self.scale) * res], dim=1)
        
        l = Tensor(l, inds=('h:b', 'in:d0'), tags=('L',))
        r = Tensor(r, inds=('h:b', 'in:d1'), tags=('R',))
        d = Tensor(d, inds=('out:d', 'h:b'), tags=('D',))
        
        return TensorNetwork([l, r, d])
    
    def permutations(self):
        return [
            [('in:d0', 'in:d0*'), ('in:d1', 'in:d1*')],
            [('in:d0', 'in:d1*'), ('in:d1', 'in:d0*')],
            [('in:d0', 'in:d1'), ('in:d0*', 'in:d1*')]
        ]