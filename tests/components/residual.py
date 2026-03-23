# %%
%load_ext autoreload
%autoreload 2

from components import Residual, Component
import plotly.express as px
import torch

torch.set_grad_enabled(False)
color = dict(color_continuous_midpoint=0, color_continuous_scale='RdBu')
# %%
alpha = 0.8
residual = Residual(5)

x = torch.randn(10, 5)
xn = torch.nn.functional.pad(x, (1, 0), value=1.0) * alpha

y0 = Component.evaluate(residual, xn)[:, 1:]
y1 = residual(x)

px.imshow((y0 / y1).cpu(), **color).show()
# torch.testing.assert_close(y0, y1 * alpha**4)
# px.imshow(torch.stack([y0, y1]).cpu(), **color, facet_col=0).show()
# %%