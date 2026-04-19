"""Linear component wrapper for embeddings and unembeddings.

Adapts standard PyTorch Linear layers to the Component interface.
"""

import torch
from torch import nn
from quimb.tensor import Tensor, TensorNetwork

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.components.base import Component, Term, spider


class LinearComponent(Component):
    """Component wrapper for nn.Linear or nn.Embedding layers.
    
    Handles both:
    - nn.Linear: Standard linear transformation
    - nn.Embedding: Lookup table (treated as linear with one-hot input)
    """
    
    def __init__(self, module, is_embedding=False):
        """Wrap a Linear or Embedding module.
        
        Args:
            module: nn.Linear or nn.Embedding instance
            is_embedding: True if module is nn.Embedding
        """
        super().__init__()
        self.module = module
        self.is_embedding = is_embedding
        
        if is_embedding:
            assert isinstance(module, nn.Embedding), "is_embedding=True requires nn.Embedding"
            self.weight = module.weight.T  # Transpose: (vocab, d) -> (d, vocab)
            self.bias = None
        else:
            assert isinstance(module, nn.Linear), "is_embedding=False requires nn.Linear"
            self.weight = module.weight
            self.bias = module.bias
    
    def _like(self):
        return dict(device=self.weight.device, dtype=self.weight.dtype)
    
    def network(self):
        """Build TN representation.
        
        For Linear/Embedding without bias, the TN is just a weight matrix
        with padded constant dimension:
        
        W_padded = [[1, 0, 0, ...],
                    [0, w_00, w_01, ...],
                    [0, w_10, w_11, ...],
                    ...]
        
        This allows the constant dimension to pass through unchanged.
        """
        if self.bias is not None:
            # Pad with bias in the constant dimension
            # W_padded[0, :] = 0, W_padded[1:, 0] = bias, W_padded[1:, 1:] = weight
            d_out, d_in = self.weight.shape
            w_padded = torch.zeros(d_out + 1, d_in + 1, **self._like())
            w_padded[0, 0] = 1.0  # Constant passes through
            w_padded[1:, 0] = self.bias  # Bias in constant column
            w_padded[1:, 1:] = self.weight  # Actual weights
        else:
            # No bias: just block diagonal [1; W]
            w_padded = torch.block_diag(
                torch.eye(1, **self._like()),
                self.weight
            )
        
        return TensorNetwork([
            Tensor(w_padded, inds=('out:d', 'in:d0'), tags=('L',))
        ])
    
    def terms(self, n_ctx, **like):
        """Decompose into terms with sequence position tracking.
        
        Linear layers have only one term (no residual connection).
        """
        tn = self.network()
        tn &= Tensor(spider(1, n_ctx, **like), inds=('in:s0', 'out:s'))
        return [Term(tn, {'in:d0': 'in:s0'})]
