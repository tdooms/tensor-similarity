
from torch import nn
from quimb.tensor import Tensor, TensorNetwork

from src.components.base import Component
from src.components.compose import pad

import torch


class MLP(Component):
    """The canonical residual + bilinear layer."""
    def __init__(self, d_model: int, d_hidden: int, scale: float = 1, bias: bool = True) -> None:
        super().__init__()
        
        # Set scale to 0 to remove the residual.
        self.scale = scale
        self.d_model, self.d_hidden = d_model, d_hidden
        
        self.l = nn.Linear(d_model, d_hidden, bias=False)
        self.r = nn.Linear(d_model, d_hidden, bias=False)
        self.d = nn.Linear(d_hidden, d_model, bias=bias)

    def forward(self, x):
        return (x * (1 - self.scale)) + self.d(self.l(x) * self.r(x)) * self.scale
    
    def _like(self):
        return dict(device=self.d.weight.device, dtype=self.d.weight.dtype)
    
    def network(self):
        dim, like = self.d_model, self._like()
        
        residual = torch.cat([torch.zeros(dim, 1, **like), torch.eye(dim, **like)], dim=1)
        constant = torch.cat([torch.ones(dim, 1, **like), torch.zeros(dim, dim, **like)], dim=1)
        
        l = torch.cat([pad(self.l.weight, self.l.bias), residual], dim=0)
        r = torch.cat([pad(self.r.weight, self.r.bias), constant], dim=0)

        residual_d = torch.cat([torch.zeros(1, dim, **like), torch.eye(dim, **like)], dim=0)
        d = torch.cat([pad(self.d.weight, self.d.bias, scale=self.scale), (1 - self.scale) * residual_d], dim=1)
        
        u = [Tensor(torch.stack([l + r, l - r]), inds=[f"h:s{i}", "h:b", f"in:d{i}"], tags=['U']) for i in range(2)]
        s = Tensor(torch.tensor([[0.25, 0.0], [0.0, -0.25]], **like), inds=['h:s0', 'h:s1'], tags=['S'])
        d = Tensor(d, inds=['out:d', 'h:b'], tags=['D'])
        
        return TensorNetwork(u + [s] + [d])