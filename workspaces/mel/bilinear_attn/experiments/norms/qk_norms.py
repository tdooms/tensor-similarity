"""Q/K normalization modules for attention experiments."""
import torch
import torch.nn as nn


class AlphaHeadNorm(nn.Module):
    """Learned per-head temperature scaling for Q and K vectors.
    
    Implements:
        q' = α_q · q
        k' = α_k · k
    
    Where α_q, α_k are learned scalars per head (not per token/batch/position).
    
    Equivalent to scaling attention scores:
        attn = (q k^T) · (α_q α_k)
    
    This is a learned temperature, not dynamic normalization.
    
    Args:
        n_head: Number of attention heads
        init_value: Initial value for alpha parameters (default: 1.0)
    """
    
    def __init__(self, n_head: int, init_value: float = 1.0) -> None:
        super().__init__()
        self.n_head = n_head
        # Learnable per-head scalars: shape (n_head,)
        self.alpha_q = nn.Parameter(torch.full((n_head,), init_value))
        self.alpha_k = nn.Parameter(torch.full((n_head,), init_value))
    
    def forward_q(self, q: torch.Tensor) -> torch.Tensor:
        """Apply alpha_q scaling to query vectors.
        
        Args:
            q: Query tensor of shape (B, T, n_head, d_head)
            
        Returns:
            Scaled query tensor of same shape
        """
        # alpha_q: (n_head,) -> (1, 1, n_head, 1)
        alpha = self.alpha_q.view(1, 1, self.n_head, 1)
        return q * alpha
    
    def forward_k(self, k: torch.Tensor) -> torch.Tensor:
        """Apply alpha_k scaling to key vectors.
        
        Args:
            k: Key tensor of shape (B, T, n_head, d_head)
            
        Returns:
            Scaled key tensor of same shape
        """
        # alpha_k: (n_head,) -> (1, 1, n_head, 1)
        alpha = self.alpha_k.view(1, 1, self.n_head, 1)
        return k * alpha


class QKNormWrapper(nn.Module):
    """Wrapper for different Q/K normalization strategies.
    
    Supports:
        - 'none': No normalization (Identity)
        - 'rmsnorm': Standard RMSNorm per token/head
        - 'alpha_head': Learned per-head temperature scaling
    
    Args:
        norm_type: Type of normalization ('none', 'rmsnorm', 'alpha_head')
        d_head: Dimension of each head (required for rmsnorm)
        n_head: Number of heads (required for alpha_head)
        alpha_init: Initial value for alpha parameters (default: 1.0)
    """
    
    def __init__(
        self,
        norm_type: str,
        d_head: int | None = None,
        n_head: int | None = None,
        alpha_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.norm_type = norm_type
        
        if norm_type == "none":
            self.norm = nn.Identity()
        elif norm_type == "rmsnorm":
            assert d_head is not None, "d_head required for rmsnorm"
            self.norm = nn.RMSNorm(d_head)
        elif norm_type == "alpha_head":
            assert n_head is not None, "n_head required for alpha_head"
            self.norm = AlphaHeadNorm(n_head, init_value=alpha_init)
        else:
            raise ValueError(
                f"Unknown norm_type: {norm_type!r}. "
                f"Must be one of: 'none', 'rmsnorm', 'alpha_head'"
            )
    
    def forward_q(self, q: torch.Tensor) -> torch.Tensor:
        """Apply normalization to query vectors."""
        if self.norm_type == "alpha_head":
            return self.norm.forward_q(q)
        else:
            return self.norm(q)
    
    def forward_k(self, k: torch.Tensor) -> torch.Tensor:
        """Apply normalization to key vectors."""
        if self.norm_type == "alpha_head":
            return self.norm.forward_k(k)
        else:
            return self.norm(k)
