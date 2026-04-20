import sys
import types

# Stub missing bilinear module so src.components.__init__ doesn't fail
_bilinear = types.ModuleType("src.components.bilinear")
_bilinear.Bilinear = None
sys.modules["src.components.bilinear"] = _bilinear

# Patch Component into src.components namespace so `from src.components import Component` works
import src.components.base as _base
import src.components
src.components.Component = _base.Component

import pytest
import torch
from src.components.attention import Attention
from src.components.mlp import MLP
from src.components.similarity import _precompile_mode
from src.models.transformer import Transformer


@pytest.fixture(autouse=True)
def _allow_compile():
    """Tests build fresh models on every run; let `similarity()` compile on
    demand inside tests. Production callers stay fail-loud."""
    with _precompile_mode():
        yield


@pytest.fixture
def seed():
    torch.manual_seed(42)


@pytest.fixture
def small_mlp(seed):
    return MLP(d_model=4, d_hidden=6, bias=False, scale=1)


@pytest.fixture
def small_attention(seed):
    return Attention(d_model=4, n_head=2, n_ctx=3, mask='causal', bias=False, scale=1)


@pytest.fixture
def small_transformer(seed):
    return Transformer(d_model=4, n_head=2, n_ctx=3, d_hidden=6, mask='causal', bias=False, scale=1)
