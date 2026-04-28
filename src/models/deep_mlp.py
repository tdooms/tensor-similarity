from torch import nn

from src.components.linear import Linear
from src.components.mlp import MLP


class DeepMLP(nn.Module):
    """Embed → N × bilinear MLP block → Unembed. `n_ctx = 1` (no sequence dim)."""
    n_ctx = 1

    def __init__(self, d_input, d_model, d_hidden, d_output, n_layers=1, bias=False, scale=1.0):
        super().__init__()
        self.flatten = nn.Flatten()
        self.embed = Linear(d_input, d_model, bias=False)
        self.body = nn.ModuleList([MLP(d_model, d_hidden, bias=bias, scale=scale) for _ in range(n_layers)])
        self.head = Linear(d_model, d_output, bias=False)

    def components(self):
        return [self.embed, *self.body, self.head]

    def forward(self, x):
        x = self.embed(self.flatten(x))
        for layer in self.body:
            x = layer(x)
        return self.head(x)
