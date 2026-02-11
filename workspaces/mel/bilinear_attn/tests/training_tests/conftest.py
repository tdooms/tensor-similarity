import pytest
import torch

B = 2
T = 16
V = 97
D_MODEL = 32
N_HEAD = 4
N_LAYERS = 2
N_CTX = 32

ATOL = 1e-5
RTOL = 1e-4


@pytest.fixture
def tiny_config():
    """Minimal config for fast training tests."""
    return {
        "name": "tiny_test",
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
        "init": {
            "std_embed": 0.02,
            "std_qkv": 0.02,
            "std_o": 0.01,
        },
        "train": {
            "batch_size": B,
            "lr": 1e-3,
            "weight_decay": 0.1,
            "max_steps": 10,
            "warmup_steps": 2,
            "grad_clip": 1.0,
            "amp": False,
        },
        "loss": {
            "type": "next_token_ce",
            "label_smoothing": 0.0,
        },
    }


@pytest.fixture
def dummy_dataloader():
    """Create a simple dummy dataloader for testing."""
    from torch.utils.data import DataLoader, TensorDataset
    
    n_samples = 20
    input_ids = torch.randint(0, V, (n_samples, T))
    dataset = TensorDataset(input_ids)
    
    def collate_fn(batch):
        return {"input_ids": torch.stack([b[0] for b in batch])}
    
    return DataLoader(dataset, batch_size=B, shuffle=True, collate_fn=collate_fn)


@pytest.fixture
def device():
    """Get available device."""
    return "cpu"
