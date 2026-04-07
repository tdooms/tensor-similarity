# %%
%load_ext autoreload
%autoreload 2

from components import Linear, Component
import plotly.express as px
import torch

torch.set_grad_enabled(False)
color = dict(color_continuous_midpoint=0, color_continuous_scale='RdBu')
# %%
linear = Linear(5, 5)

x = torch.randn(10, 5)
xn = torch.nn.functional.pad(x, (1, 0), value=1) * 0.8

y0 = Component.evaluate(linear, xn)[:, 1:]
y1 = linear(x)

px.imshow((y0 / y1).cpu(), **color).show()
# %%