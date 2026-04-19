from torch import nn

from src.models.base import Model
from src.components.linear import Linear
from src.components.attention import Attention
from src.components.mlp import MLP


class Layer(nn.Module):
    def __init__(self, d_model, n_head, n_ctx, d_hidden, mask='causal', bias=False, scale=1):
        super().__init__()
        self.attn = Attention(d_model, n_head, n_ctx, mask=mask, bias=bias, scale=scale)
        self.mlp = MLP(d_model, d_hidden, bias=bias, scale=scale)

    def forward(self, x):
        return self.mlp(self.attn(x))


class Transformer(Model):
    def __init__(self, d_input, d_model, n_head, n_ctx, d_hidden, d_output, n_layer=2, mask='causal', bias=False, scale=1):
        super().__init__(None)
        self.n_ctx = n_ctx
        self.embed = Linear(d_input, d_model, bias=False)
        self.body = nn.ModuleList([
            Layer(d_model, n_head, n_ctx, d_hidden, mask=mask, bias=bias, scale=scale)
            for _ in range(n_layer)
        ])
        self.head = Linear(d_model, d_output, bias=False)

    def components(self):
        return [self.embed] + sum(([l.attn, l.mlp] for l in self.body), []) + [self.head]

    def forward(self, x):
        x = self.embed(x)
        for layer in self.body:
            x = layer(x)
        return self.head(x)
