
from torch import nn
from quimb.tensor import Tensor, TensorNetwork

from src.components.base import Component
from src.components.compose import sequential2

import torch


class MLP(Component):
    """MLP replacement using a (batteries-included) bilinear layer."""
    def __init__(self, d_model: int, d_hidden: int, scale: float = 1, normalize: bool = True, bias: bool = True) -> None:
        super().__init__()
        
        self.d_model, self.d_hidden = d_model, d_hidden
        self.scale = scale
        self.norm = nn.RMSNorm(d_model) if normalize else nn.Identity()
        
        self.l = nn.Linear(d_model, d_hidden, bias=False)
        self.r = nn.Linear(d_model, d_hidden, bias=False)
        self.d = nn.Linear(d_hidden, d_model, bias=bias)

    def forward(self, x):
        nx = self.norm(x)
        return x + self.d(self.l(nx) * self.r(nx) * self.scale)
    
    def _like(self):
        return dict(device=self.d.weight.device, dtype=self.d.weight.dtype)
    
    def normalisation(self):
        """Creates a CPD of normalisation + residual. See documentation TODO."""
        dim, like = self.d_model, self._like()
        
        e0 = torch.cat([torch.zeros(dim, 1, **like), torch.eye(dim, **like)], dim=1)
        e1 = torch.cat([torch.ones(dim, 1, **like), torch.eye(dim, **like)], dim=1)
        e2 = torch.cat([-torch.ones(dim, 1, **like), torch.eye(dim, **like)], dim=1)
        e = torch.cat([torch.tensor([[1] + [0] * dim], **like), e0, e1, e2], dim=0)

        m = torch.cat([torch.eye(dim, **like), -torch.eye(dim, **like)], dim=-1)
        m = torch.block_diag(torch.ones(1, 1, **like), torch.ones(1, dim, **like) / dim, 0.25 * m)
        
        t = [Tensor(e, inds=['h:n', f'in:d{i}'], tags=['N']) for i in range(2)] + [Tensor(m, inds=['out:d', 'h:n'], tags=['N'])]
        return TensorNetwork(t)
    
    def bilinear(self):
        """Creates a CPD of the bilinear + residual. See documentation TODO."""
        dim, hid, like = self.d_model, self.d_hidden, self._like()
        
        l = self.l.weight * self.norm.weight[None, :]
        r = self.r.weight * self.norm.weight[None, :]
        d = self.scale * self.d.weight
        db = torch.zeros_like(self.d.weight[:, 0]) if self.d.bias is None else self.d.bias
        
        l = torch.block_diag(torch.tensor([[0, 1]], **like), torch.cat([torch.eye(dim, **like), l], dim=0))
        r = torch.block_diag(torch.ones(1, 1, **like), torch.ones(dim, 1, **like), r)
        s = torch.cat([l + r, l - r], dim=0)
    
        q = torch.cat([db[:, None], torch.eye(dim, **like), d], dim=1)
        q = torch.cat([torch.tensor([[1] + [0]*(hid + dim)], **like), q], dim=0)
        q = 0.25 * torch.cat([q, -q], dim=-1)

        t = [Tensor(s, inds=['h:b', f'in:d{i}'], tags=['B']) for i in range(2)] + [Tensor(q, inds=['out:d', 'h:b'], tags=['B'])]
        return TensorNetwork(t)
    
    def network(self):
        """Creates a tensor network representation of the MLP (see documentation TODO)."""

        # The output of this function are two stacked bilinear layers.
        # The first simply computes the norm, the second one contains the actual MLP along with the residual scaling.
        return sequential2(self.normalisation(), self.bilinear(), tag='m')