"""BilinearMLP model for modular arithmetic."""

import torch
import torch.nn as nn


class BilinearMLP(nn.Module):
    """
    Bilinear model for modular arithmetic.
    Takes two separate inputs (one-hot for a and b).
    h = (W_l @ x_a) * (W_r @ x_b)
    output = W_p @ h
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.W_l = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.01)
        self.W_r = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.01)
        self.W_p = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.01)

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x_a: (batch, input_dim) one-hot vectors for first operand
            x_b: (batch, input_dim) one-hot vectors for second operand

        Returns:
            (batch, output_dim) logits
        """
        left = x_a @ self.W_l.T   # (batch, hidden)
        right = x_b @ self.W_r.T  # (batch, hidden)
        h = left * right          # (batch, hidden) element-wise product
        return h @ self.W_p.T     # (batch, output)
