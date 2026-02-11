from torch import nn
from quimb.tensor import Tensor, TensorNetwork
from src.components.compose import pad

import torch


class Bilinear(nn.Module):
    """A pure bilinear layer, only used in toy models."""
    def __init__(self, d_in, d_hid, d_out=None, bias: bool = True) -> None:
        super().__init__()
        d_out = d_out or d_in
        
        self.left = nn.Linear(d_in, d_hid, bias=bias)
        self.right = nn.Linear(d_in, d_hid, bias=bias)
        self.down = nn.Linear(d_hid, d_out, bias=bias)
        
        self.t = nn.Buffer(torch.tensor([[0.25, 0], [0, -0.25]]), persistent=False)
        
        if bias: torch.nn.init.zeros_(self.left.bias)
        if bias: torch.nn.init.zeros_(self.right.bias)
        if bias: torch.nn.init.zeros_(self.down.bias)
    
    def forward(self, x):
        return self.down(self.left(x) * self.right(x))

    def network(self):
        l, r = pad(self.left), pad(self.right)
        s = torch.stack([l + r, l - r], dim=0)
        
        d = Tensor(pad(self.down), inds=["out:d", "hid"], tags=['D'])
        s = [Tensor(s, inds=[f"s{i}", "hid", f"in:d{i}"], tags=['S']) for i in [0, 1]]
        t = Tensor(self.t, inds=["s0", "s1"], tags=['T'])
        
        return d & s[0] & s[1] & t
    