import pytest
import torch

B = 2
T = 6
V = 97
D_MODEL = 32
N_HEAD = 4
D_HEAD = D_MODEL // N_HEAD
N_LAYERS = 2
N_CTX = 16

ATOL = 1e-6
RTOL = 1e-5


@pytest.fixture
def test_config():
    """Tiny test config."""
    return {
        "name": "test",
        "seed": 42,
        "model": {
            "vocab_size": V,
            "n_ctx": N_CTX,
            "d_model": D_MODEL,
            "n_head": N_HEAD,
            "n_layers": N_LAYERS,
            "attn_scale": 0.2,
            "rope_base": 10000,
            "use_rmsnorm_qk": False,
            "use_bias_qk": True,
        },
        "init": {
            "std_embed": 0.02,
            "std_qkv": 0.02,
            "std_o": 0.01,
        },
    }


@pytest.fixture
def random_input_ids():
    """Random input IDs for testing."""
    return torch.randint(0, V, (B, T))
