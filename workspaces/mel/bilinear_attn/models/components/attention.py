"""Bilinear attention as a Component for TN similarity.

This module provides a Component-compatible wrapper around BilinearAttention
that implements the network() and terms() methods required by the main
codebase's TN similarity algorithm.

The implementation closely follows src/components/attention.py but adapts
to the mel workspace's BilinearAttention architecture.
"""

import torch
from torch import nn
from quimb.tensor import Tensor, TensorNetwork

from src.components.base import Component, Term
from src.components.compose import pad
from src.components.attention import Rotary, Mask


class BilinearAttentionComponent(Component):
    """Bilinear attention as a Component for TN similarity.
    
    This wraps the mel workspace's BilinearAttention to provide the Component
    interface required by the main codebase's TN similarity algorithm.
    
    Architecture:
        scores1 = q1 @ k1.T
        scores2 = q2 @ k2.T
        pattern = (scores1 * scores2) / d_head^2 * causal_mask
        z = pattern @ v
        out = x + scale * Wo(z)
    
    The TN representation decomposes this into:
        - Term 1 (residual): Identity with scale (1 - attn_scale)
        - Term 2 (active): Full attention TN with 5 input legs
    """
    
    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_ctx: int,
        scale: float = 1.0,
        bias: bool = True,
        rope_base: int = 10000,
    ) -> None:
        super().__init__()
        self.d_head = d_model // n_head
        self.n_head = n_head
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.scale = scale
        self.bias = bias
        
        # Use main codebase's Rotary and Mask components
        self.rotary = Rotary(self.d_head, n_ctx, base=rope_base)
        self.mask = Mask(n_ctx, 'causal')
        
        # Weight matrices (will be loaded from trained model)
        self.q1 = nn.Linear(d_model, d_model, bias=bias)
        self.k1 = nn.Linear(d_model, d_model, bias=bias)
        self.q2 = nn.Linear(d_model, d_model, bias=bias)
        self.k2 = nn.Linear(d_model, d_model, bias=bias)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
    
    def _like(self):
        return dict(device=self.o.weight.device, dtype=self.o.weight.dtype)
    
    def attention(self, q: Tensor, k: Tensor, mod: str):
        """Creates the QK-circuit with rotary embeddings.
        
        This follows the main codebase's Attention.attention() method exactly.
        """
        r = self.rotary.network(mod, **self._like())
        s = Tensor(
            torch.full((self.d_head // 2,), 1.0 / self.d_head, **self._like()),
            inds=(f'{mod}:h',),
            tags=('S',)
        )
        
        rename = {idx: f'{mod}:{idx}' for idx in ('2q', '2k', 'h', 'q', 'k')}
        return r & q.reindex(rename) & k.reindex(rename) & s
    
    def network(self):
        """Attention-only TN (without residual). No constant output dim.
        
        This follows the main codebase's Attention.network() method exactly.
        
        Returns:
            TensorNetwork with indices:
                - 'out:d': output dimension (d_model)
                - 'in:d0': V input (d_model+1, includes constant)
                - 'in:d1', 'in:d2': K1, K2 inputs (d_model+1 each)
                - 'in:d3', 'in:d4': Q1, Q2 inputs (d_model+1 each)
                - 'in:s', 'out:s': sequence positions (internal)
        """
        d, n, h = self.d_model, self.n_head, self.d_head
        
        # Output projection: scale * O.weight reshaped to (d_model, n_head, d_head)
        o = Tensor(
            self.scale * self.o.weight.view(d, n, h),
            inds=('out:d', 'n', 'ov:h'),
            tags=('O',)
        )
        
        # Value projection: padded weight (n_head, d_head, d_model+1)
        v = Tensor(
            pad(self.v.weight, self.v.bias, constant=False).view(n, h, d + 1),
            inds=('n', 'ov:h', 'in:d0'),
            tags=('V',)
        )
        
        # Q/K projections: padded weights (n_head, 2, d_head//2, d_model+1)
        # The '2' dimension is for the rotary embedding's cos/sin split
        q1 = Tensor(
            pad(self.q1.weight, self.q1.bias, constant=False).view(n, 2, h // 2, d + 1),
            inds=('n', '2q', 'h', 'q'),
            tags=('Q',)
        )
        k1 = Tensor(
            pad(self.k1.weight, self.k1.bias, constant=False).view(n, 2, h // 2, d + 1),
            inds=('n', '2k', 'h', 'k'),
            tags=('K',)
        )
        q2 = Tensor(
            pad(self.q2.weight, self.q2.bias, constant=False).view(n, 2, h // 2, d + 1),
            inds=('n', '2q', 'h', 'q'),
            tags=('Q',)
        )
        k2 = Tensor(
            pad(self.k2.weight, self.k2.bias, constant=False).view(n, 2, h // 2, d + 1),
            inds=('n', '2k', 'h', 'k'),
            tags=('K',)
        )
        
        # Build QK circuits with rotary embeddings
        left = self.attention(q1, k1, 'left')
        right = self.attention(q2, k2, 'right')
        
        # Causal mask
        mask = self.mask.network()
        
        # Rename internal indices to input leg indices
        rename = {
            'left:k': 'in:d1',   # K1 input
            'right:k': 'in:d2',  # K2 input
            'left:q': 'in:d3',   # Q1 input
            'right:q': 'in:d4',  # Q2 input
        }
        
        return TensorNetwork(
            [o, v, mask, left, right],
            check_collisions=False
        ).reindex(rename)
    
    def terms(self, n_ctx, **like):
        """Returns list of Terms for second-moment propagation.

        Mirrors the main codebase's ``Attention.terms()``: legs share ``in:s``
        (V, K1, K2) and ``out:s`` (Q1, Q2) implicitly via the TN itself, so no
        spider/delta tensors are needed. The bilinear term carries the
        (K1↔K2, Q1↔Q2) swap symmetry that Isserlis uses to dedupe matchings.
        """
        d, d1 = self.d_model, self.d_model + 1

        # Term 1: residual. mel's forward is ``lerp(x, o(z), scale)``, i.e.
        # ``(1 - scale) * x + scale * o(z)``, so the identity term is scaled
        # by ``(1 - scale)`` on data dims. The constant-1 axis is always
        # preserved unchanged.
        scale = torch.cat([torch.ones(1, **like), (1 - self.scale) * torch.ones(d, **like)])
        identity = TensorNetwork([Tensor(torch.diag(scale), inds=('out:d', 'in:d0'))])

        # Term 2: active attention, embedded from d → d+1 via [0; I].
        active = self.network().reindex({'out:d': 'mid:d'})
        embed = torch.zeros(d1, d, **like)
        embed[1:] = torch.eye(d, **like)
        active &= Tensor(embed, inds=('out:d', 'mid:d'))

        # Legs 0-2 (V, K1, K2) share 'in:s'; legs 3-4 (Q1, Q2) share 'out:s'.
        # Those indices are already present in the active TN via mask/rotary/V,
        # so bridges reuse them—no delta tensors required.
        legs = {'in:d0': 'in:s', 'in:d1': 'in:s', 'in:d2': 'in:s',
                'in:d3': 'out:s', 'in:d4': 'out:s'}
        # (q1·k1)(q2·k2) is invariant under the simultaneous swap
        # (K1↔K2, Q1↔Q2), i.e. (in:d1↔in:d2, in:d3↔in:d4).
        swap = {'in:d1': 'in:d2', 'in:d2': 'in:d1',
                'in:d3': 'in:d4', 'in:d4': 'in:d3'}
        return [Term(identity, {'in:d0': 'out:s'}),
                Term(active, legs, symmetries=(swap,))]

    @classmethod
    def from_bilinear_attention(cls, layer, rope_base: int = 10000) -> "BilinearAttentionComponent":
        """Create from a trained BilinearAttention layer.
        
        Args:
            layer: Trained BilinearAttention from mel workspace
            rope_base: RoPE base frequency (default 10000)
            
        Returns:
            BilinearAttentionComponent with weights copied from the layer
            
        Raises:
            ValueError: If layer has RMSNorm on Q/K (not supported for TN similarity)
        """
        # Check for unsupported features
        if hasattr(layer, 'norm_qk') and not isinstance(layer.norm_qk, nn.Identity):
            raise ValueError(
                "BilinearAttention with use_rmsnorm_qk=True is not supported for TN similarity. "
                "RMSNorm is a non-polynomial operation that cannot be represented as a tensor network."
            )
        
        component = cls(
            d_model=layer.d_model,
            n_head=layer.n_head,
            n_ctx=layer.n_ctx,
            scale=layer.scale,
            bias=layer.q1.bias is not None,
            rope_base=rope_base,
        )
        
        # Copy weights
        component.q1.weight.data.copy_(layer.q1.weight.data)
        component.k1.weight.data.copy_(layer.k1.weight.data)
        component.q2.weight.data.copy_(layer.q2.weight.data)
        component.k2.weight.data.copy_(layer.k2.weight.data)
        component.v.weight.data.copy_(layer.v.weight.data)
        component.o.weight.data.copy_(layer.o.weight.data)
        
        if layer.q1.bias is not None:
            component.q1.bias.data.copy_(layer.q1.bias.data)
            component.k1.bias.data.copy_(layer.k1.bias.data)
            component.q2.bias.data.copy_(layer.q2.bias.data)
            component.k2.bias.data.copy_(layer.k2.bias.data)
        
        return component


class QuadraticAttentionComponent(Component):
    """Quadratic attention as a Component for TN similarity.
    
    This is a simplified version with a single QK circuit:
        scores = q @ k.T
        pattern = (scores / d_head)^2 * causal_mask
        z = pattern @ v
        out = x + scale * Wo(z)
    
    For TN similarity, we can represent this as a special case of bilinear
    attention where Q1=Q2 and K1=K2.
    """
    
    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_ctx: int,
        scale: float = 1.0,
        bias: bool = True,
        rope_base: int = 10000,
    ) -> None:
        super().__init__()
        self.d_head = d_model // n_head
        self.n_head = n_head
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.scale = scale
        self.bias = bias
        
        self.rotary = Rotary(self.d_head, n_ctx, base=rope_base)
        self.mask = Mask(n_ctx, 'causal')
        
        # Single Q/K circuit (unlike bilinear which has two)
        self.q = nn.Linear(d_model, d_model, bias=bias)
        self.k = nn.Linear(d_model, d_model, bias=bias)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
    
    def _like(self):
        return dict(device=self.o.weight.device, dtype=self.o.weight.dtype)
    
    def attention(self, q: Tensor, k: Tensor, mod: str):
        """Creates the QK-circuit with rotary embeddings."""
        r = self.rotary.network(mod, **self._like())
        s = Tensor(
            torch.full((self.d_head // 2,), 1.0 / self.d_head, **self._like()),
            inds=(f'{mod}:h',),
            tags=('S',)
        )
        
        rename = {idx: f'{mod}:{idx}' for idx in ('2q', '2k', 'h', 'q', 'k')}
        return r & q.reindex(rename) & k.reindex(rename) & s
    
    def network(self):
        """Attention-only TN (without residual).
        
        For quadratic attention, we use the same Q/K weights for both circuits.
        """
        d, n, h = self.d_model, self.n_head, self.d_head
        
        o = Tensor(
            self.scale * self.o.weight.view(d, n, h),
            inds=('out:d', 'n', 'ov:h'),
            tags=('O',)
        )
        
        v = Tensor(
            pad(self.v.weight, self.v.bias, constant=False).view(n, h, d + 1),
            inds=('n', 'ov:h', 'in:d0'),
            tags=('V',)
        )
        
        # Same Q/K for both circuits (quadratic = bilinear with shared weights)
        q_tensor = Tensor(
            pad(self.q.weight, self.q.bias, constant=False).view(n, 2, h // 2, d + 1),
            inds=('n', '2q', 'h', 'q'),
            tags=('Q',)
        )
        k_tensor = Tensor(
            pad(self.k.weight, self.k.bias, constant=False).view(n, 2, h // 2, d + 1),
            inds=('n', '2k', 'h', 'k'),
            tags=('K',)
        )
        
        left = self.attention(q_tensor, k_tensor, 'left')
        right = self.attention(q_tensor, k_tensor, 'right')
        mask = self.mask.network()
        
        rename = {
            'left:k': 'in:d1',
            'right:k': 'in:d2',
            'left:q': 'in:d3',
            'right:q': 'in:d4',
        }
        
        return TensorNetwork(
            [o, v, mask, left, right],
            check_collisions=False
        ).reindex(rename)
    
    def terms(self, n_ctx, **like):
        """Returns list of Terms for second-moment propagation.

        Quadratic attention shares Q and K weights across the two score
        factors, so the swap symmetry (K1↔K2, Q1↔Q2) still holds and we use
        the same schema as the bilinear case.
        """
        d, d1 = self.d_model, self.d_model + 1

        scale = torch.cat([torch.ones(1, **like), (1 - self.scale) * torch.ones(d, **like)])
        identity = TensorNetwork([Tensor(torch.diag(scale), inds=('out:d', 'in:d0'))])

        active = self.network().reindex({'out:d': 'mid:d'})
        embed = torch.zeros(d1, d, **like)
        embed[1:] = torch.eye(d, **like)
        active &= Tensor(embed, inds=('out:d', 'mid:d'))

        legs = {'in:d0': 'in:s', 'in:d1': 'in:s', 'in:d2': 'in:s',
                'in:d3': 'out:s', 'in:d4': 'out:s'}
        swap = {'in:d1': 'in:d2', 'in:d2': 'in:d1',
                'in:d3': 'in:d4', 'in:d4': 'in:d3'}
        return [Term(identity, {'in:d0': 'out:s'}),
                Term(active, legs, symmetries=(swap,))]
    
    @classmethod
    def from_quadratic_attention(cls, layer, rope_base: int = 10000) -> "QuadraticAttentionComponent":
        """Create from a trained QuadraticAttention layer.
        
        Args:
            layer: Trained QuadraticAttention from mel workspace
            rope_base: RoPE base frequency (default 10000)
            
        Returns:
            QuadraticAttentionComponent with weights copied from the layer
        """
        if hasattr(layer, 'norm_qk') and not isinstance(layer.norm_qk, nn.Identity):
            raise ValueError(
                "QuadraticAttention with use_rmsnorm_qk=True is not supported for TN similarity."
            )
        
        component = cls(
            d_model=layer.d_model,
            n_head=layer.n_head,
            n_ctx=layer.n_ctx,
            scale=layer.scale,
            bias=layer.q.bias is not None,
            rope_base=rope_base,
        )
        
        component.q.weight.data.copy_(layer.q.weight.data)
        component.k.weight.data.copy_(layer.k.weight.data)
        component.v.weight.data.copy_(layer.v.weight.data)
        component.o.weight.data.copy_(layer.o.weight.data)
        
        if layer.q.bias is not None:
            component.q.bias.data.copy_(layer.q.bias.data)
            component.k.bias.data.copy_(layer.k.bias.data)
        
        return component
