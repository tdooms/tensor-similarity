# %%
"""
Exact functional similarity for linear attention via Wick decomposition.

f(x)_{s1} = O · Σ_{s2} M[s1,s2] · (R_{s1} Q x_{s1})·(R_{s2} K x_{s2}) · (V x_{s2})

6 Gaussian input legs grouped by sequence position:
    s1: Q_a, Q_b          → 1 forced pairing
    s2: K_a, V_a           → 1 forced pairing  (when s2 ≠ s2')
    s2': K_b, V_b          → 1 forced pairing  (when s2 ≠ s2')

Leading (s2 ≠ s2'):  1 contraction (all pairings forced)
Collision (s2 = s2'): 2 cross-pairings added (exact, O(1/S) suppressed)
Total: 3 TN contractions.
"""
import torch
import torch.nn as nn
from quimb.tensor import Tensor, TensorNetwork
from einops import rearrange, einsum


# ── Components ──────────────────────────────────────────────────────────

class Rotary(nn.Module):
    def __init__(self, d_head, n_ctx, base=10_000):
        super().__init__()
        freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        ctx = torch.arange(n_ctx).float()
        freqs = torch.outer(ctx, freq)
        self.register_buffer('cos', freqs.cos(), persistent=False)
        self.register_buffer('sin', freqs.sin(), persistent=False)

    def forward(self, x):
        s = x.size(-3)
        a, b = x.chunk(2, dim=-1)
        y = torch.cat((-b, a), dim=-1)
        cos = torch.cat([self.cos[:s], self.cos[:s]], dim=-1)[None, :, None, :]
        sin = torch.cat([self.sin[:s], self.sin[:s]], dim=-1)[None, :, None, :]
        return x * cos + y * sin

    def tn(self, mod, device, dtype):
        data = [[[[1,0],[0,1]],[[0,-1],[1,0]]],[[[0,1],[-1,0]],[[1,0],[0,1]]]]
        black = Tensor(torch.tensor(data, device=device, dtype=dtype),
                       inds=[f'{mod}:iq', f'{mod}:ik', f'{mod}:2q', f'{mod}:2k'], tags=['#'])
        emb = torch.stack([self.cos, self.sin], dim=-1)
        eq = Tensor(emb.to(device=device, dtype=dtype),
                    inds=['out:s', f'{mod}:h', f'{mod}:iq'], tags=['E'])
        ek = Tensor(emb.to(device=device, dtype=dtype),
                    inds=['in:s', f'{mod}:h', f'{mod}:ik'], tags=['E'])
        return black & eq & ek


class LinearAttention(nn.Module):
    def __init__(self, d_model, n_head, n_ctx, mask='causal'):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.n_ctx = n_ctx

        self.rotary = Rotary(self.d_head, n_ctx)
        masks = dict(causal=torch.tril, none=torch.ones_like, diag=torch.eye)
        M = torch.tril(torch.ones(n_ctx, n_ctx)) if mask == 'causal' else torch.ones(n_ctx, n_ctx)
        self.register_buffer('mask', M, persistent=False)

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        q, k, v = [rearrange(op(x), '... s (n h) -> ... n s h', n=self.n_head)
                    for op in [self.q, self.k, self.v]]
        q = self.rotary(q.transpose(-2, -3)).transpose(-2, -3)
        k = self.rotary(k.transpose(-2, -3)).transpose(-2, -3)

        score = einsum(q, k, '... n s1 h, ... n s2 h -> ... n s1 s2') / self.d_head
        pattern = score * self.mask[None, :q.size(-2), :k.size(-2)]
        z = einsum(pattern, v, '... n s1 s2, ... n s2 h -> ... n s1 h')
        return rearrange(z, '... n s h -> ... s (n h)') @ self.o.weight.T

    def _like(self):
        return dict(device=self.o.weight.device, dtype=self.o.weight.dtype)

    def network(self):
        d, n, h = self.d_model, self.n_head, self.d_head
        like = self._like()

        o = Tensor(self.o.weight.view(d, n, h), inds=['out:d', 'n', 'ov:h'], tags=['O'])
        v = Tensor(self.v.weight.view(n, h, d), inds=['n', 'ov:h', 'in:v'], tags=['V'])

        q = Tensor(self.q.weight.view(n, 2, h // 2, d),
                   inds=['n', 'r:2q', 'r:h', 'in:q'], tags=['Q'])
        k = Tensor(self.k.weight.view(n, 2, h // 2, d),
                   inds=['n', 'r:2k', 'r:h', 'in:k'], tags=['K'])

        rot = self.rotary.tn('r', **like)
        scale = Tensor(torch.full((h // 2,), 1.0 / h, **like), inds=['r:h'], tags=['S'])
        mask = Tensor(self.mask.to(**like), inds=['out:s', 'in:s'], tags=['M'])

        rename = {'r:k': 'in:k', 'r:q': 'in:q'}  # not needed here but for clarity
        return TensorNetwork([o, v, q, k, scale, mask]) & rot


# ── Similarity ──────────────────────────────────────────────────────────

def _prefix(tn, p):
    return tn.reindex({ix: f'{p}:{ix}' for ix in tn.ind_map})


def _double(tn_a, tn_b, collision=False):
    a = _prefix(tn_a, 'a')
    b = _prefix(tn_b, 'b')
    remap = {'b:out:d': 'a:out:d', 'b:out:s': 'a:out:s'}
    if collision:
        remap['b:in:s'] = 'a:in:s'
    return a & b.reindex(remap)


def _bridge(tn, sigma, pairs):
    tn = tn.copy()
    for i, (l1, l2) in enumerate(pairs):
        tn &= Tensor(sigma, inds=[l1, l2], tags=[f'Σ{i}'])
    return tn


def similarity(tn_a, tn_b, sigma):
    """
    Exact E[f_a(x) · f_b(x)] for linear attention. 3 contractions.

    t1: leading (all seq positions). Q_a↔Q_b, K_a↔V_a, K_b↔V_b.
    t2: collision cross-pairing 1. K_a↔K_b, V_a↔V_b.
    t3: collision cross-pairing 2. K_a↔V_b, V_a↔K_b.
    """
    Q = ('a:in:q', 'b:in:q')

    C = dict(output_inds=())  # head index 'n' is hyper (4 tensors)

    sep = _double(tn_a, tn_b, collision=False)
    t1 = _bridge(sep, sigma, [Q, ('a:in:k', 'a:in:v'), ('b:in:k', 'b:in:v')]).contract(**C)

    col = _double(tn_a, tn_b, collision=True)
    t2 = _bridge(col, sigma, [Q, ('a:in:k', 'b:in:k'), ('a:in:v', 'b:in:v')]).contract(**C)
    t3 = _bridge(col, sigma, [Q, ('a:in:k', 'b:in:v'), ('a:in:v', 'b:in:k')]).contract(**C)

    return t1 + t2 + t3


def cosine(tn_a, tn_b, sigma):
    ab = similarity(tn_a, tn_b, sigma)
    aa = similarity(tn_a, tn_a, sigma)
    bb = similarity(tn_b, tn_b, sigma)
    return ab / (aa * bb) ** 0.5


# ── Validation ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    torch.manual_seed(42)

    d, n_head, n_ctx = 16, 4, 8
    dtype = torch.float64

    attn_a = LinearAttention(d, n_head, n_ctx).to(dtype=dtype)
    attn_b = LinearAttention(d, n_head, n_ctx).to(dtype=dtype)

    sigma = torch.eye(d, dtype=dtype)

    # TN computation
    tn_a = attn_a.network()
    tn_b = attn_b.network()
    tn_sim = similarity(tn_a, tn_b, sigma)
    tn_aa = similarity(tn_a, tn_a, sigma)
    tn_bb = similarity(tn_b, tn_b, sigma)
    tn_cos = tn_sim / (tn_aa * tn_bb) ** 0.5

    print(f"TN similarity:  {tn_sim:.6f}")
    print(f"TN norm a:      {tn_aa:.6f}")
    print(f"TN norm b:      {tn_bb:.6f}")
    print(f"TN cosine:      {tn_cos:.6f}")

    # Monte Carlo validation
    N = 50_000
    total_ab, total_aa, total_bb = 0.0, 0.0, 0.0

    with torch.no_grad():
        for _ in range(N):
            x = torch.randn(1, n_ctx, d, dtype=dtype)
            ya = attn_a(x).squeeze(0)  # [n_ctx, d]
            yb = attn_b(x).squeeze(0)

            total_ab += (ya * yb).sum().item()
            total_aa += (ya * ya).sum().item()
            total_bb += (yb * yb).sum().item()

    mc_ab = total_ab / N
    mc_aa = total_aa / N
    mc_bb = total_bb / N
    mc_cos = mc_ab / (mc_aa * mc_bb) ** 0.5

    print(f"\nMC similarity:  {mc_ab:.6f}  (N={N})")
    print(f"MC norm a:      {mc_aa:.6f}")
    print(f"MC norm b:      {mc_bb:.6f}")
    print(f"MC cosine:      {mc_cos:.6f}")

    print(f"\nRelative error: {abs(tn_sim - mc_ab) / abs(mc_ab):.4%}")
# %%