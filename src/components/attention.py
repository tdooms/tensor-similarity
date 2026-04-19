from torch import nn
from einops import rearrange, einsum
from quimb.tensor import Tensor, TensorNetwork
import torch

from src.components.base import Component, Term
from src.components.compose import pad


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

        black = torch.tensor([[[[1, 0], [0, 1]], [[0, -1], [1, 0]]],
                              [[[0, 1], [-1, 0]], [[1, 0], [0, 1]]]], dtype=torch.float32)
        self.black = nn.Buffer(black, persistent=False)

    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        y = torch.cat((-b, a), dim=-1)
        return (x * self.cos_cached[:, :x.size(-3)]) + (y * self.sin_cached[:, :x.size(-3)])
    
    def network(self, mod, device, dtype):
        inds = (f'{mod}:iq', f'{mod}:ik', f'{mod}:2q', f'{mod}:2k')
        black = Tensor(self.black.to(dtype), inds=inds, tags=('#',))

        emb = torch.stack([self.cos, self.sin], dim=-1)
        q = Tensor(emb, inds=('out:s', f'{mod}:h', f'{mod}:iq'), tags=('E',))
        k = Tensor(emb, inds=('in:s', f'{mod}:h', f'{mod}:ik'), tags=('E',))

        return black & q & k

class Mask(Component):
    def __init__(self, n_ctx: int, kind: str) -> None:
        super().__init__()
        
        data = dict(
            causal=torch.tril(torch.ones(n_ctx, n_ctx)),
            none=torch.ones(n_ctx, n_ctx),
            diag=torch.eye(n_ctx, n_ctx),
        )
        
        self.mask = nn.Buffer(data[kind], persistent=False)
    
    def forward(self, x: Tensor):
        return x * self.mask[None, None, :x.size(-2), :x.size(-1)]
    
    def network(self, tag='M'):
        return Tensor(self.mask.data, inds=('out:s', 'in:s'), tags=(tag,))

class Attention(Component):
    """Attention replacement using a quadratic scoring function."""
    def __init__(self, d_model: int, n_head: int, n_ctx: int, mask: str, bias: bool = False, scale: int = 1) -> None:
        super().__init__()
        self.d_head = d_model // n_head
        self.n_head = n_head
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.scale = scale
        
        self.rotary = Rotary(self.d_head, n_ctx)
        self.mask = Mask(n_ctx, mask)
        
        self.q1 = nn.Linear(d_model, d_model, bias=bias)
        self.k1 = nn.Linear(d_model, d_model, bias=bias)
        self.q2 = nn.Linear(d_model, d_model, bias=bias)
        self.k2 = nn.Linear(d_model, d_model, bias=bias)
        
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x: Tensor):
        q1, k1, q2, k2, v = [rearrange(op(x), '... (n_head d_head) -> ... n_head d_head', n_head=self.n_head) for op in [self.q1, self.k1, self.q2, self.k2, self.v]]
        q1, k1, q2, k2 = [self.rotary(t) for t in [q1, k1, q2, k2]]
        
        scores1 = einsum(q1, k1, "... seq_q n_head d_head, ... seq_k n_head d_head -> ... n_head seq_q seq_k")
        scores2 = einsum(q2, k2, "... seq_q n_head d_head, ... seq_k n_head d_head -> ... n_head seq_q seq_k")
        pattern = self.mask((scores1 * scores2) / self.d_head**2)
        
        z = einsum(pattern, v, "... n_head seq_q seq_k, ... seq_k n_head d_head -> ... seq_q n_head d_head")
        z = rearrange(z, '... seq n_head d_head -> ... seq (n_head d_head)')
        return torch.lerp(x, self.o(z), self.scale)

    def _like(self):
        return dict(device=self.o.weight.device, dtype=self.o.weight.dtype)

    def attention(self, q: Tensor, k: Tensor, mod: str):
        """Creates the QK-circuit with rotary embeddings."""
        r = self.rotary.network(mod, **self._like())
        s = Tensor(torch.full((self.d_head // 2,), 1.0 / self.d_head, **self._like()), inds=(f'{mod}:h',), tags=('S',)) # what the fuck was this?

        rename = {idx: f'{mod}:{idx}' for idx in ('2q', '2k', 'h', 'q', 'k')}
        return r & q.reindex(rename) & k.reindex(rename) & s

    def network(self):
        """Attention-only TN (without residual). No constant output dim."""
        d, n, h = self.d_model, self.n_head, self.d_head

        o = Tensor(self.scale * self.o.weight.view(d, n, h), inds=('out:d', 'n', 'ov:h'), tags=('O',))
        v = Tensor(pad(self.v.weight, self.v.bias, constant=False).view(n, h, d + 1), inds=('n', 'ov:h', 'in:d0'), tags=('V',))

        q1 = Tensor(pad(self.q1.weight, self.q1.bias, constant=False).view(n, 2, h // 2, d + 1), inds=('n', '2q', 'h', 'q'), tags=('Q',))
        k1 = Tensor(pad(self.k1.weight, self.k1.bias, constant=False).view(n, 2, h // 2, d + 1), inds=('n', '2k', 'h', 'k'), tags=('K',))
        q2 = Tensor(pad(self.q2.weight, self.q2.bias, constant=False).view(n, 2, h // 2, d + 1), inds=('n', '2q', 'h', 'q'), tags=('Q',))
        k2 = Tensor(pad(self.k2.weight, self.k2.bias, constant=False).view(n, 2, h // 2, d + 1), inds=('n', '2k', 'h', 'k'), tags=('K',))

        left = self.attention(q1, k1, 'left')
        right = self.attention(q2, k2, 'right')
        mask = self.mask.network()

        rename = {'left:k': 'in:d1', 'right:k': 'in:d2', 'left:q': 'in:d3', 'right:q': 'in:d4'}
        return TensorNetwork([o, v, mask, left, right], check_collisions=False).reindex(rename)

    def terms(self, n_ctx):
        d, d1, like = self.d_model, self.d_model + 1, self._like()

        # Term 1: residual — constant dim preserved, data dims scaled by (1-s)
        scale = torch.cat([torch.ones(1, **like), (1 - self.scale) * torch.ones(d, **like)])
        identity = TensorNetwork([
            Tensor(torch.diag(scale), inds=('out:d', 'in:d0')),
        ])

        # Term 2: active attention, embedded from d → d+1 output via [0; I] tensor
        active = self.network().reindex({'out:d': 'mid:d'})
        embed = torch.zeros(d1, d, **like)
        embed[1:] = torch.eye(d, **like)
        active &= Tensor(embed, inds=('out:d', 'mid:d'))

        # Legs 0-2 (V, K1, K2) share 'in:s'; legs 3-4 (Q1, Q2) share 'out:s'.
        # These indices are already present in the active TN (mask/rotary/V),
        # so no delta tensors are needed — the bridges will reuse them.
        legs = {'in:d0': 'in:s', 'in:d1': 'in:s', 'in:d2': 'in:s',
                'in:d3': 'out:s', 'in:d4': 'out:s'}
        # (q1·k1)(q2·k2) is invariant under the simultaneous swap
        # (K1↔K2, Q1↔Q2), i.e. (in:d1↔in:d2, in:d3↔in:d4).
        swap = {'in:d1': 'in:d2', 'in:d2': 'in:d1',
                'in:d3': 'in:d4', 'in:d4': 'in:d3'}
        return [Term(identity, {'in:d0': 'out:s'}),
                Term(active, legs, symmetries=(swap,))]
