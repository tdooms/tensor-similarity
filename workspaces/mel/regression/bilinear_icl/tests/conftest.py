import pytest
import torch


@pytest.fixture
def small_cfg():
    return {
        "D": 4,
        "K": 8,
        "d_model": 32,
        "n_head": 4,
        "n_layers": 2,
        "d_mlp": 32,
        "attn_scale": 0.35,
        "mlp_scale": 0.35,
        "bos_norm_eps": 1e-6,
    }


@pytest.fixture
def cpu_device():
    return torch.device("cpu")
