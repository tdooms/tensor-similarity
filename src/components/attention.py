from torch import nn
from einops import rearrange, einsum
from quimb.tensor import Tensor, TensorNetwork
import torch

from src.components.base import Component


class Rotary(Component):
    """A modern implementation of the rotary position encoding."""
    def __init__(self, dim: int, n_ctx: int, base: int = 10_000) -> None:
        super().__init__()
        
        freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        ctx = torch.arange(n_ctx).type_as(freq)
        freqs = torch.einsum("i,j->ij", ctx, freq)
        
        self.cos = nn.Buffer(freqs.cos(), persistent=False)
        self.sin = nn.Buffer(freqs.sin(), persistent=False)
        
        self.cos_cached = nn.Buffer(torch.cat([self.cos, self.cos], dim=-1)[None, :, None, :], persistent=False)
        self.sin_cached = nn.Buffer(torch.cat([self.sin, self.sin], dim=-1)[None, :, None, :], persistent=False)

    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        y = torch.cat((-b, a), dim=-1)
        return (x * self.cos_cached[:, :x.size(-3)]) + (y * self.sin_cached[:, :x.size(-3)])
    
    def network(self, mod, **kwargs):
        data = [[[[1, 0], [0, 1]], [[0, -1], [1, 0]]], [[[0, 1], [-1, 0]], [[1, 0], [0, 1]]]]
        black = Tensor(torch.tensor(data, **kwargs), inds=[f'{mod}:iq', f'{mod}:ik', f'{mod}:2q', f'{mod}:2k'], tags=['#'])
        
        emb = torch.stack([self.cos, self.sin], dim=-1)
        q_rot = Tensor(emb, inds=['out:t', f'{mod}:h', f'{mod}:iq'], tags=['E'])
        k_rot = Tensor(emb, inds=['in:s', f'{mod}:h', f'{mod}:ik'], tags=['E'])
        
        return black & q_rot & k_rot

    
class Mask(Component):
    def __init__(self, n_ctx: int, kind: str) -> None:
        super().__init__()
        
        data = dict(
            causal=torch.tril(torch.ones(n_ctx, n_ctx)),
            none=torch.ones(n_ctx, n_ctx),
            diag=torch.eye(n_ctx, n_ctx),
        )
        
        self.mask = nn.Buffer(data[kind], persistent=False)
    
    def forward(self, x):
        return x * self.mask[None, None, :x.size(-2), :x.size(-1)]
    
    def network(self, inds=['out:t', 'in:s'], tag='M'):
        return Tensor(self.mask.data, inds=inds, tags=[tag])

class Attention(Component):
    """Attention replacement using a quadratic scoring function."""
    def __init__(self, d_model: int, n_head: int, n_ctx: int, mask: str, scale: int = 1) -> None:
        super().__init__()
        self.d_head = d_model // n_head
        self.n_head = n_head
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.scale = scale
        
        self.rotary = Rotary(self.d_head, n_ctx)
        self.mask = Mask(n_ctx, mask)
        
        self.q1 = nn.Linear(d_model, d_model, bias=False)
        self.k1 = nn.Linear(d_model, d_model, bias=False)
        self.q2 = nn.Linear(d_model, d_model, bias=False)
        self.k2 = nn.Linear(d_model, d_model, bias=False)
        
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x):
        q1, k1, q2, k2, v = [rearrange(op(x), '... (n_head d_head) -> ... n_head d_head', n_head=self.n_head) for op in [self.q1, self.k1, self.q2, self.k2, self.v]]
        q1, k1, q2, k2 = self.rotary(self.norm(q1)), self.rotary(self.norm(k1)), self.rotary(self.norm(q2)), self.rotary(self.norm(k2))
        
        scores1 = einsum(q1, k1, "... seq_q n_head d_head, ... seq_k n_head d_head -> ... n_head seq_q seq_k")
        scores2 = einsum(q2, k2, "... seq_q n_head d_head, ... seq_k n_head d_head -> ... n_head seq_q seq_k")
        pattern = self.mask((scores1 * scores2) / self.d_head**2)
        
        z = einsum(pattern, v, "... n_head seq_q seq_k, ... seq_k n_head d_head -> ... seq_q n_head d_head")
        z = rearrange(z, '... seq n_head d_head -> ... seq (n_head d_head)')
        return x + self.o(z) * self.scale

    def network(self):
        raise NotImplementedError("Attention is currently unimplemented.")

    def make_attn(self, q, k, mod):
        rot = self.rotary.network(mod, **self._like())
        # s = Tensor(torch.tensor([1.0 / self.d_head], **self._like()), inds=[f'{mod}:h'], tags=['S'])
        
        rename = {idx: f'{mod}:{idx}' for idx in ['2q', '2k', 'h', 'q', 'k']}
        network = rot & q.reindex(rename) & k.reindex(rename) & Tensor(self.norm.weight.pow(2), inds=[f'{mod}:h'], tags=['S'])
        
        network.add_at(mod.upper())
        return network

    def make_norm(self, g, mod):
        # s = Tensor(torch.tensor([1.0 / self.d_model], **self._like()), inds=[f'{mod}:h'], tags=['S'])
        return g.reindex({f'2{mod}': '2', mod: f'left:{mod}'}) & g.reindex({f'2{mod}': '2', mod: f'right:{mod}'})

    # def network(self):
    #     # Rename some key/query entries in the network and define the matrices
    #     rename = {'left:k': 'in:d1', 'right:k': 'in:d2', 'left:q': 'in:d3', 'right:q': 'in:d4'}

    #     o = make_padded(self.o.weight.view(self.d_model, self.n_head, self.d_head), dim=0, inds=['out:d', 'n', 'ov:h'], tags=['O']) * self.scale
        
    #     v = make_padded(self.v.weight.view(self.n_head, self.d_head, self.d_model), self.v.bias, dim=-1, inds=['n', 'ov:h', 'in:d0'], tags=['V'])
    #     q = make_padded(self.q.weight.view(self.n_head, 2, self.d_head // 2, self.d_model), self.q.bias, dim=-1, inds=['n', '2q', 'h', 'q'], tags=['Q'])
    #     k = make_padded(self.k.weight.view(self.n_head, 2, self.d_head // 2, self.d_model), self.k.bias, dim=-1, inds=['n', '2k', 'h', 'k'], tags=['K'])

    #     # Quadratic attention uses two identical key/query circuits
    #     left, right = [self.make_attn(q, k, mod=mod) for mod in ['left', 'right']]
    #     mask = self.mask.network()
    #     layer = TensorNetwork([o, v, mask, left, right], check_collisions=False).reindex(rename)

    #     # The residual connection consists two wires and a norm computation
    #     res = make_wire(self.d_model, inds=['out:d', 'in:d0'], tags=["I"], **self._like())
    #     connect = make_wire(self.n_ctx - 1, inds=['in:s', 'out:t'], tags=['M'], **self._like())
    #     norm = self.make_norm(k, mod='k') & self.make_norm(q, mod='q')
    #     residual = TensorNetwork([res, norm, connect], check_collisions=False).reindex(rename)
    
    #     return Diag(layer, residual, qkv=self.qkv)