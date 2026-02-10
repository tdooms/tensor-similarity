from dataclasses import dataclass
from functools import reduce
from typing import List
from torch import nn
from safetensors.torch import save_model, load_model

import json
import torch
import os

from src.components import MLP, Linear


@dataclass
class Config:
    n_layers: int                       # Layer to hook the model at
    d_input: int                        # Input dimension
    d_output: int                       # Output dimension
    d_model: int                        # Model dimension at the hook point
    bias: bool = True                   # Whether to use bias in the mlp
    expansion: int = 4                  # Expansion factor for the autoencoder
    tags: List[str] | None = None       # Tags to identify the model
    
    @property
    def d_hidden(self):
        return self.d_model * self.expansion
    
    @property
    def name(self):
        tags = self.tags or []
        return '-'.join(['toy', f"l{self.n_layers}", f"d{self.d_model}", tags])


class ToyModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        
        self.embed = Linear(config.d_input, config.d_model, bias=False)
        self.body = MLP(config.d_model, config.d_hidden)
        self.head = Linear(config.d_model, config.d_output, bias=False)
    
    @staticmethod
    def from_config(**kwargs):
        return ToyModel(Config(**kwargs))
    
    def _like(self):
        return dict(device=next(self.parameters()).device, dtype=next(self.parameters()).dtype)

    def forward(self, inputs):
        x = self.flatten(inputs)
        x = self.embed(x)
        x = self.body(x)
        return self.head(x)
