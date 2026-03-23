# %%
%load_ext autoreload
%autoreload 2

from components import MLP, Component
from quimb.tensor import Tensor, TensorNetwork

import plotly.express as px
import torch

torch.set_grad_enabled(False)
color = dict(color_continuous_midpoint=0, color_continuous_scale='RdBu')
# %% single layer
alpha = 0.5

# RMSNorm is usually initialised at 1, change that
mlp = MLP(5, 20, bias=False).double()
mlp.norm.weight.data = torch.randn_like(mlp.norm.weight.data)

x = torch.randn(10, 5).double() * 5
xn = alpha * torch.nn.functional.pad(x, (1, 0), value=1)

y0 = Component.evaluate(mlp, xn)[:, 1:]
y1 = mlp(x)

# The relative error should be alpha**4 * norm(x)
torch.testing.assert_close((y0 / y1), alpha**4 * x.pow(2).mean(dim=1, keepdim=True).repeat(1, 5))
px.imshow((y0 / y1).cpu(), **color).show()

# %% two layers
alpha = 0.5

# RMSNorm is usually initialised at 1, change that
mlp = MLP(5, 20, bias=True).double()
mlp.norm.weight.data = torch.randn_like(mlp.norm.weight.data)

x = torch.randn(10, 5).double() * 2
xn = alpha * torch.nn.functional.pad(x, (1, 0), value=1)
xn = Component.evaluate(mlp, xn)

y0 = Component.evaluate(mlp, xn)[:, 1:]
yh = mlp(x)
y1 = mlp(yh)

# The relative error should be alpha**16 * norm(x)**4 * norm(yh)
torch.testing.assert_close((y0 / y1), alpha**16 * (x.pow(2).mean(dim=1, keepdim=True).pow(4) * yh.pow(2).mean(dim=1, keepdim=True)).repeat(1, 5))
px.imshow((y0 / y1).cpu(), **color).show()

# %%