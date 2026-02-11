import pytest
import torch

V = 100
D_MODEL = 32
N_HEAD = 4
N_LAYERS = 2
N_CTX = 32
B = 2
T = 16


@pytest.fixture
def tiny_config():
    """Minimal config for analysis tests."""
    return {
        "name": "analysis_test",
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
            "use_bias_qkv": True,
            "use_bias_o": True,
        },
    }


@pytest.fixture
def dummy_dataloader():
    """Create a simple dummy dataloader for testing."""
    from torch.utils.data import DataLoader, TensorDataset
    
    n_samples = 50
    input_ids = torch.randint(0, V, (n_samples, T))
    dataset = TensorDataset(input_ids)
    
    def collate_fn(batch):
        return {"input_ids": torch.stack([b[0] for b in batch])}
    
    return DataLoader(dataset, batch_size=B, shuffle=False, collate_fn=collate_fn)


@pytest.fixture
def model(tiny_config):
    """Create a small model for testing."""
    from models import AttentionLM
    torch.manual_seed(42)
    return AttentionLM.from_config(tiny_config)
