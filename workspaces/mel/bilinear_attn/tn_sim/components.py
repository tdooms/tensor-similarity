"""Component wrappers for mel's attention kernels.

Adapts BilinearAttention and QuadraticAttention to the Component interface
required by src/components/similarity.py.
"""

import torch
from torch import nn
from quimb.tensor import Tensor, TensorNetwork

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.components.base import Component, Term, spider
from src.components.compose import pad


class BilinearAttentionComponent(Component):
    """Component wrapper for BilinearAttention.
    
    Converts a trained BilinearAttention module into a Component that can be
    used with the TN similarity computation.
    
    Note: This assumes use_rmsnorm_qk=False (TN-clean mode). If Q/K normalization
    is enabled, the TN representation is approximate.
    """
    
    def __init__(self, attn_module):
        """Wrap a BilinearAttention module.
        
        Args:
            attn_module: Instance of models.attention_kernels.bilinear.BilinearAttention
        """
        super().__init__()
        self.attn = attn_module
        self.d_model = attn_module.d_model
        self.n_head = attn_module.n_head
        self.d_head = attn_module.d_head
        self.n_ctx = attn_module.n_ctx
        self.scale = attn_module.scale
        
        # Check if TN-clean
        if not isinstance(attn_module.norm_qk, nn.Identity):
            import warnings
            warnings.warn(
                "BilinearAttention has Q/K normalization enabled. "
                "TN representation will be approximate."
            )
    
    def _like(self):
        return dict(device=self.attn.o.weight.device, dtype=self.attn.o.weight.dtype)
    
    def _rotary_network(self, mod: str):
        """Create rotary embedding TN for one QK circuit."""
        device, dtype = self.attn.o.weight.device, self.attn.o.weight.dtype
        
        # Black box tensor (rotation structure)
        data = [
            [[[1, 0], [0, 1]], [[0, -1], [1, 0]]],
            [[[0, 1], [-1, 0]], [[1, 0], [0, 1]]],
        ]
        black = Tensor(
            torch.tensor(data, device=device, dtype=dtype),
            inds=(f'{mod}:iq', f'{mod}:ik', f'{mod}:2q', f'{mod}:2k'),
            tags=('#',)
        )
        
        # Positional embeddings
        emb = torch.stack([self.attn.rotary.cos, self.attn.rotary.sin], dim=-1)
        q_rot = Tensor(emb, inds=('out:s', f'{mod}:h', f'{mod}:iq'), tags=('E',))
        k_rot = Tensor(emb, inds=('in:s', f'{mod}:h', f'{mod}:ik'), tags=('E',))
        
        return black & q_rot & k_rot
    
    def _qk_circuit(self, q_proj: nn.Linear, k_proj: nn.Linear, mod: str):
        """Create QK circuit TN with rotary embeddings."""
        d, n, h = self.d_model, self.n_head, self.d_head
        
        # Pad and reshape Q/K projections
        q_padded = pad(q_proj.weight, q_proj.bias, constant=False)
        k_padded = pad(k_proj.weight, k_proj.bias, constant=False)
        
        q = Tensor(
            q_padded.view(n, 2, h // 2, d + 1),
            inds=('n', '2q', 'h', 'q'),
            tags=('Q',)
        )
        k = Tensor(
            k_padded.view(n, 2, h // 2, d + 1),
            inds=('n', '2k', 'h', 'k'),
            tags=('K',)
        )
        
        # Rotary embeddings
        rotary = self._rotary_network(mod)
        
        # Scaling factor
        scale_tensor = Tensor(
            torch.full((h // 2,), 1.0 / self.d_head, **self._like()),
            inds=(f'{mod}:h',),
            tags=('S',)
        )
        
        # Combine and rename indices
        rename = {idx: f'{mod}:{idx}' for idx in ('2q', '2k', 'h', 'q', 'k')}
        return rotary & q.reindex(rename) & k.reindex(rename) & scale_tensor
    
    def network(self):
        """Build the active attention TN (without residual).
        
        Returns TensorNetwork with:
        - 5 input indices: in:d0 (V), in:d1 (K1), in:d2 (K2), in:d3 (Q1), in:d4 (Q2)
        - 1 output index: out:d
        - Internal sequence indices: in:s (for K/V), out:s (for Q)
        """
        d, n, h = self.d_model, self.n_head, self.d_head
        
        # Output projection (scaled)
        o = Tensor(
            self.scale * self.attn.o.weight.view(d, n, h),
            inds=('out:d', 'n', 'ov:h'),
            tags=('O',)
        )
        
        # Value projection (padded, no constant output)
        v_padded = pad(self.attn.v.weight, self.attn.v.bias, constant=False)
        v = Tensor(
            v_padded.view(n, h, d + 1),
            inds=('n', 'ov:h', 'in:d0'),
            tags=('V',)
        )
        
        # Two QK circuits
        left = self._qk_circuit(self.attn.q1, self.attn.k1, 'left')
        right = self._qk_circuit(self.attn.q2, self.attn.k2, 'right')
        
        # Causal mask
        mask = Tensor(
            self.attn.causal_mask.data,
            inds=('out:s', 'in:s'),
            tags=('M',)
        )
        
        # Combine and rename to standard input indices
        tn = TensorNetwork([o, v, mask, left, right], check_collisions=False)
        rename = {
            'left:k': 'in:d1',
            'right:k': 'in:d2',
            'left:q': 'in:d3',
            'right:q': 'in:d4',
        }
        return tn.reindex(rename)
    
    def terms(self, n_ctx, **like):
        """Decompose into residual + active terms.
        
        Returns:
            List of 2 Terms:
            - Term 1: Residual (identity scaled by 1-scale)
            - Term 2: Active attention circuit
        """
        d, d1 = self.d_model, self.d_model + 1
        
        # Term 1: Residual - constant dim preserved, data dims scaled by (1-scale)
        scale_vec = torch.cat([
            torch.ones(1, **like),
            (1 - self.scale) * torch.ones(d, **like)
        ])
        identity = TensorNetwork([
            Tensor(torch.diag(scale_vec), inds=('out:d', 'in:d0')),
            Tensor(spider(1, n_ctx, **like), inds=('in:s0', 'out:s')),
        ])
        
        # Term 2: Active attention, embedded from d → d+1 output via [0; I] tensor
        active = self.network().reindex({'out:d': 'mid:d'})
        embed = torch.zeros(d1, d, **like)
        embed[1:] = torch.eye(d, **like)
        active &= Tensor(embed, inds=('out:d', 'mid:d'))
        
        # Tie input legs to sequence positions via spider tensors
        # 3-way spider: V, K1, K2 all read from same input position
        active &= Tensor(
            spider(3, n_ctx, **like),
            inds=('in:s0', 'in:s1', 'in:s2', 'in:s')
        )
        # 2-way spider: Q1, Q2 read from same output position
        active &= Tensor(
            spider(2, n_ctx, **like),
            inds=('in:s3', 'in:s4', 'out:s')
        )
        
        legs = {f'in:d{i}': f'in:s{i}' for i in range(5)}
        return [
            Term(identity, {'in:d0': 'in:s0'}),
            Term(active, legs)
        ]


class QuadraticAttentionComponent(Component):
    """Component wrapper for QuadraticAttention.
    
    Similar to BilinearAttentionComponent but for single QK circuit with squared scores.
    """
    
    def __init__(self, attn_module):
        """Wrap a QuadraticAttention module.
        
        Args:
            attn_module: Instance of models.attention_kernels.bilinear.QuadraticAttention
        """
        super().__init__()
        self.attn = attn_module
        self.d_model = attn_module.d_model
        self.n_head = attn_module.n_head
        self.d_head = attn_module.d_head
        self.n_ctx = attn_module.n_ctx
        self.scale = attn_module.scale
        
        if not isinstance(attn_module.norm_qk, nn.Identity):
            import warnings
            warnings.warn(
                "QuadraticAttention has Q/K normalization enabled. "
                "TN representation will be approximate."
            )
    
    def _like(self):
        return dict(device=self.attn.o.weight.device, dtype=self.attn.o.weight.dtype)
    
    def _rotary_network(self, mod: str):
        """Create rotary embedding TN."""
        device, dtype = self.attn.o.weight.device, self.attn.o.weight.dtype
        
        data = [
            [[[1, 0], [0, 1]], [[0, -1], [1, 0]]],
            [[[0, 1], [-1, 0]], [[1, 0], [0, 1]]],
        ]
        black = Tensor(
            torch.tensor(data, device=device, dtype=dtype),
            inds=(f'{mod}:iq', f'{mod}:ik', f'{mod}:2q', f'{mod}:2k'),
            tags=('#',)
        )
        
        emb = torch.stack([self.attn.rotary.cos, self.attn.rotary.sin], dim=-1)
        q_rot = Tensor(emb, inds=('out:s', f'{mod}:h', f'{mod}:iq'), tags=('E',))
        k_rot = Tensor(emb, inds=('in:s', f'{mod}:h', f'{mod}:ik'), tags=('E',))
        
        return black & q_rot & k_rot
    
    def network(self):
        """Build the active attention TN (without residual).
        
        For quadratic attention: pattern = (Q@K^T / d_head)^2
        This requires TWO copies of the same QK circuit.
        
        Returns TensorNetwork with:
        - 3 input indices: in:d0 (V), in:d1 (K), in:d2 (Q)
        - 1 output index: out:d
        """
        d, n, h = self.d_model, self.n_head, self.d_head
        
        # Output projection
        o = Tensor(
            self.scale * self.attn.o.weight.view(d, n, h),
            inds=('out:d', 'n', 'ov:h'),
            tags=('O',)
        )
        
        # Value projection
        v_padded = pad(self.attn.v.weight, self.attn.v.bias, constant=False)
        v = Tensor(
            v_padded.view(n, h, d + 1),
            inds=('n', 'ov:h', 'in:d0'),
            tags=('V',)
        )
        
        # Q and K projections (padded)
        q_padded = pad(self.attn.q.weight, self.attn.q.bias, constant=False)
        k_padded = pad(self.attn.k.weight, self.attn.k.bias, constant=False)
        
        # For squared scores, we need the SAME Q and K to appear twice
        # We'll create two copies with different index names
        q1 = Tensor(
            q_padded.view(n, 2, h // 2, d + 1),
            inds=('n', '2q', 'h', 'q'),
            tags=('Q',)
        )
        k1 = Tensor(
            k_padded.view(n, 2, h // 2, d + 1),
            inds=('n', '2k', 'h', 'k'),
            tags=('K',)
        )
        
        # Rotary embeddings (same for both copies)
        rotary1 = self._rotary_network('left')
        rotary2 = self._rotary_network('right')
        
        # Scaling
        scale_tensor = Tensor(
            torch.full((h // 2,), 1.0 / self.d_head, **self._like()),
            inds=('left:h',),
            tags=('S',)
        )
        
        # Build two QK circuits that share the same input
        rename1 = {idx: f'left:{idx}' for idx in ('2q', '2k', 'h', 'q', 'k')}
        rename2 = {idx: f'right:{idx}' for idx in ('2q', '2k', 'h', 'q', 'k')}
        
        left = rotary1 & q1.reindex(rename1) & k1.reindex(rename1) & scale_tensor
        right = rotary2 & q1.reindex(rename2) & k1.reindex(rename2)
        
        # Causal mask
        mask = Tensor(
            self.attn.causal_mask.data,
            inds=('out:s', 'in:s'),
            tags=('M',)
        )
        
        # Combine
        tn = TensorNetwork([o, v, mask, left, right], check_collisions=False)
        
        # Rename to standard indices
        # Both left and right circuits read from the SAME Q and K inputs
        rename = {
            'left:k': 'in:d1',
            'right:k': 'in:d1',  # Same K input
            'left:q': 'in:d2',
            'right:q': 'in:d2',  # Same Q input
        }
        return tn.reindex(rename)
    
    def terms(self, n_ctx, **like):
        """Decompose into residual + active terms.
        
        Returns:
            List of 2 Terms:
            - Term 1: Residual
            - Term 2: Active quadratic attention
        """
        d, d1 = self.d_model, self.d_model + 1
        
        # Term 1: Residual
        scale_vec = torch.cat([
            torch.ones(1, **like),
            (1 - self.scale) * torch.ones(d, **like)
        ])
        identity = TensorNetwork([
            Tensor(torch.diag(scale_vec), inds=('out:d', 'in:d0')),
            Tensor(spider(1, n_ctx, **like), inds=('in:s0', 'out:s')),
        ])
        
        # Term 2: Active attention
        active = self.network().reindex({'out:d': 'mid:d'})
        embed = torch.zeros(d1, d, **like)
        embed[1:] = torch.eye(d, **like)
        active &= Tensor(embed, inds=('out:d', 'mid:d'))
        
        # Spider tensors for sequence positions
        # V and K read from input position
        active &= Tensor(
            spider(2, n_ctx, **like),
            inds=('in:s0', 'in:s1', 'in:s')
        )
        # Q reads from output position
        active &= Tensor(
            spider(1, n_ctx, **like),
            inds=('in:s2', 'out:s')
        )
        
        legs = {f'in:d{i}': f'in:s{i}' for i in range(3)}
        return [
            Term(identity, {'in:d0': 'in:s0'}),
            Term(active, legs)
        ]
