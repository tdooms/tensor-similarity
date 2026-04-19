"""Embedding layer as a Component for TN similarity.

For TN similarity, we treat token embeddings as a Linear layer from vocab_size
to d_model. The input distribution is assumed to be uniform over the vocabulary,
which gives an initial gram matrix of (1/vocab_size) * E @ E^T.
"""

import torch
from torch import nn
from quimb.tensor import Tensor, TensorNetwork

from src.components.base import Component


class EmbeddingComponent(Component):
    """Token embedding as a Component (Linear from vocab_size to d_model).
    
    This wraps an nn.Embedding to provide the Component interface required
    by the main codebase's TN similarity algorithm.
    
    For TN similarity purposes, we treat the embedding as a linear map from
    one-hot token vectors to d_model dimensional embeddings.
    """
    
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Store weight as a parameter (will be loaded from trained model)
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
    
    def _like(self):
        return dict(device=self.weight.device, dtype=self.weight.dtype)
    
    def network(self):
        """Returns TN representation of embedding layer.
        
        The embedding is represented as a (d_model+1, vocab_size+1) matrix
        with the bias/constant dimension prepended:
        
            [[1,    0,    0,   ..., 0   ],
             [0,    E[0,0], E[1,0], ..., E[V-1,0]],
             [0,    E[0,1], E[1,1], ..., E[V-1,1]],
             ...
             [0,    E[0,D-1], ...        E[V-1,D-1]]]
        
        This preserves the constant dimension through the embedding.
        """
        # Pure on-device construction (avoids CPU->CUDA scalar copies
        # that break CUDA graph capture): block-diag [[1], E^T].
        w = torch.block_diag(torch.eye(1, **self._like()), self.weight.T)
        return TensorNetwork([Tensor(w, inds=('out:d', 'in:d0'), tags=('E',))])

    # terms() inherits from Component: single leg 'in:d0' tied to 'out:s'.

    @classmethod
    def from_embedding(cls, embedding: nn.Embedding) -> "EmbeddingComponent":
        """Create from a trained nn.Embedding layer.
        
        Args:
            embedding: Trained nn.Embedding with shape (vocab_size, d_model)
            
        Returns:
            EmbeddingComponent with weights copied from the embedding
        """
        vocab_size, d_model = embedding.weight.shape
        component = cls(vocab_size, d_model)
        component.weight.data.copy_(embedding.weight.data)
        return component


class UnembeddingComponent(Component):
    """Unembedding (output projection) as a Component.
    
    This wraps an nn.Linear to provide the Component interface.
    The unembedding projects from d_model to vocab_size.
    """
    
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Store weight as a parameter (will be loaded from trained model)
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
    
    def _like(self):
        return dict(device=self.weight.device, dtype=self.weight.dtype)
    
    def network(self):
        """Returns TN representation of unembedding layer.
        
        The unembedding is represented as a (vocab_size+1, d_model+1) matrix
        with the bias/constant dimension prepended.
        """
        # Pure on-device construction (avoids CPU->CUDA scalar copies
        # that break CUDA graph capture): block-diag [[1], W].
        w = torch.block_diag(torch.eye(1, **self._like()), self.weight)
        return TensorNetwork([Tensor(w, inds=('out:d', 'in:d0'), tags=('U',))])

    # terms() inherits from Component: single leg 'in:d0' tied to 'out:s'.

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "UnembeddingComponent":
        """Create from a trained nn.Linear layer.
        
        Args:
            linear: Trained nn.Linear with shape (vocab_size, d_model)
            
        Returns:
            UnembeddingComponent with weights copied from the linear layer
        """
        vocab_size, d_model = linear.weight.shape
        component = cls(d_model, vocab_size)
        component.weight.data.copy_(linear.weight.data)
        return component
