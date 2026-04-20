"""Mel checkpoint similarity: MelLM polynomial wrapper + loader + cosine helper.

MelAttention uses additive residual `y = x + scale·o(z)` (identity on the
residual, `scale` baked into `o.weight` in `network()`), not lerp. `cosine()`
clamps the self-traces before sqrt because fp32 noise near zero can flip the
sign of the normalized diagonal.
"""
from pathlib import Path

import torch
from torch import nn
from quimb.tensor import Tensor, TensorNetwork

from src.components.attention import Attention
from src.components.linear import Linear
from src.components.base import Term


CFG = dict(vocab=4096, n_ctx=4, d_model=256, n_head=8, n_layers=2, attn_scale=0.35)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32
CKPT_DIR = Path.home() / 'Downloads' / 'mel_ckpts'
PICKED_FILE = CKPT_DIR / 'picked.txt'


class MelAttention(Attention):
    """Additive-residual bilinear attention. Identity residual (no `(1-s)`
    lerp factor); `scale` multiplies `o.weight` inside `network()` already."""
    def terms(self, n_ctx):
        d, like = self.d_model, self._like()
        identity = TensorNetwork([
            Tensor(torch.eye(d + 1, **like), inds=('out:d', 'in:d0')),
        ])
        active = self.network().reindex({'out:d': 'mid:d'})
        embed = torch.zeros(d + 1, d, **like)
        embed[1:] = torch.eye(d, **like)
        active &= Tensor(embed, inds=('out:d', 'mid:d'))
        legs = {'in:d0': 'in:s', 'in:d1': 'in:s', 'in:d2': 'in:s',
                'in:d3': 'out:s', 'in:d4': 'out:s'}
        return [Term(identity, {'in:d0': 'out:s'}),
                Term(active, legs, symmetries=((0, 2, 1, 4, 3),))]


class MelLM(nn.Module):
    """Minimal 2-layer bilinear attention LM, shape-compatible with Mel's."""
    def __init__(self, vocab, n_ctx, d_model, n_head, n_layers, attn_scale):
        super().__init__()
        self.n_ctx = n_ctx
        self.embed = Linear(vocab, d_model, bias=False)
        self.layers = nn.ModuleList([
            MelAttention(d_model, n_head, n_ctx, mask='causal', bias=True, scale=attn_scale)
            for _ in range(n_layers)
        ])
        self.unembed = Linear(d_model, vocab, bias=False)

    def components(self):
        return [self.embed, *self.layers, self.unembed]

    def load_mel(self, sd):
        """Load Mel's state dict. We don't fold `final_norm.running_mean_energy`
        into unembed the way eval-mode forward does — cosine is invariant to
        per-model output rescaling (α cancels in the ratio), and the ~22× scale
        training produces overflows fp32 at vocab-scale contractions → NaN."""
        self.embed.weight.data.copy_(sd['embed.weight'].T)
        for i, layer in enumerate(self.layers):
            for name in ('q1', 'k1', 'q2', 'k2', 'v', 'o'):
                lin = getattr(layer, name)
                lin.weight.data.copy_(sd[f'layers.{i}.{name}.weight'])
                if lin.bias is not None and f'layers.{i}.{name}.bias' in sd:
                    lin.bias.data.copy_(sd[f'layers.{i}.{name}.bias'])
        self.unembed.weight.data.copy_(sd['unembed.weight'])


def cosine(aa, ab, bb, eps=1e-30):
    """Cosine similarity from the three Σ tensors directly (avoids the 4× stack
    alloc inside `similarity()`). `clamp_min(0)` guards fp32 noise near zero."""
    tr = lambda x: torch.einsum('ijij->', x[:, 1:, :, 1:])
    denom = (tr(aa).clamp_min(0) * tr(bb).clamp_min(0)).sqrt() + eps
    return (tr(ab) / denom).item()


def load_models(limit=None):
    """Load Mel checkpoints as frozen MelLM instances on DEVICE/DTYPE."""
    with open(PICKED_FILE) as f:
        ckpts = [line.strip().split('\t') for line in f]
    if limit: ckpts = ckpts[:limit]
    models = []
    for rel, path in ckpts:
        sd = torch.load(path, map_location='cpu', weights_only=False)['model_state_dict']
        m = MelLM(**CFG).to(device=DEVICE, dtype=DTYPE)
        m.load_mel(sd)
        m.requires_grad_(False)
        step = int(rel.split('_')[-1].split('.')[0])
        models.append((step, m))
    return models
