"""
1-layer bilinear attention-only model for SimpleStories.

Uses the Attention class from src/components/attention.py which implements
quadratic scoring: pattern = (Q1·K1) * (Q2·K2), i.e. the attention pattern
is the element-wise product of two independent attention patterns.

Config matches reference implementation (tn_4_interp/language/):
- d_model=256, n_head=4, d_head=64
- NO RMSNorm anywhere
- NO weight tying (separate lm_head, zeros-init)
- Zero-init output projection (model starts as identity)
- scale=1 (full residual, no damping)
- NO norm inside attention (Identity)
"""

import torch
from torch import nn

import sys
sys.path.insert(0, "/workspace/tensor-mars")

from src.components.attention import Attention


class AttnOnlyModel(nn.Module):
    """1-layer bilinear attention-only transformer for next-token prediction."""

    def __init__(self, vocab_size, d_model, n_head, n_ctx, mask="causal", scale=1):
        super().__init__()
        self.d_model = d_model
        self.n_ctx = n_ctx
        self.vocab_size = vocab_size
        self.n_head = n_head

        self.embed = nn.Embedding(vocab_size, d_model)
        self.attn = Attention(d_model, n_head, n_ctx, mask=mask, scale=scale)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

        # No normalization inside attention
        self.attn.norm = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        # Normal init for output projection and unembed (PyTorch defaults)

    def forward(self, x):
        """x: (batch, seq) token ids -> (batch, seq, vocab) logits"""
        h = self.embed(x)
        h = self.attn(h)
        return self.unembed(h)


def make_model(vocab_size=4096, d_model=256, n_head=4, n_ctx=256):
    """Create a 1-layer attn-only model matching reference config."""
    return AttnOnlyModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_head=n_head,
        n_ctx=n_ctx,
        mask="causal",
        scale=1,
    )
