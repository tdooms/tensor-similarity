from torch import nn
import torch
from quimb.tensor import Tensor

class LinearAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, n_ctx: int):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.n_ctx = n_ctx

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        
        self.mask = nn.Buffer(torch.tril(torch.ones(n_ctx, n_ctx)), persistent=False)
    
    def network(self):
        q = Tensor(self.q, inds=("hid:qk", "in:d0"), tags=("Q",))
        k = Tensor(self.k, inds=("hid:qk", "in:d1"), tags=("K",))
        v = Tensor(self.v, inds=("out:d", "in:d2"), tags=("V",))
        m = Tensor(self.mask, inds=("out:s", "in:s"), tags=("M",))
        
        return q & k & v & m