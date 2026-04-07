# %%
import sys
import types

# Stub missing bilinear module so the __init__ import doesn't fail
sys.modules["src.components.bilinear"] = types.ModuleType("src.components.bilinear")
sys.modules["src.components.bilinear"].Bilinear = None

import torch
from src.components.attention import Attention
from src.components.mlp import MLP
from src.components.compose import sequential

# %%
# Small 2-layer transformer
d_model = 16
n_head = 2
n_ctx = 8
d_hidden = 32

attn0 = Attention(d_model, n_head, n_ctx, mask="causal")
mlp0 = MLP(d_model, d_hidden)
attn1 = Attention(d_model, n_head, n_ctx, mask="causal")
mlp1 = MLP(d_model, d_hidden)

# %%
# Build the full 2-layer tensor network: attn -> mlp -> attn -> mlp
tn = sequential(
    attn0.network(),
    mlp0.network(),
    attn1.network(),
    mlp1.network(),
)

print(f"Tensors: {tn.num_tensors}, Indices: {tn.num_indices}")

# %%
tn.draw(
    layout="kamada_kawai",
    iterations=500,
    k=0.02,
    figsize=(20, 12),
    color=["O", "V", "Q", "K", "U", "S", "D", "E", "#", "M"],
)

# %%
