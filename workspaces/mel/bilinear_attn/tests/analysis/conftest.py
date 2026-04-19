import pytest
import torch
from torch.utils.data import Dataset, DataLoader

from data.tokenization import get_tokenizer

_tokenizer = get_tokenizer()
PAD_TOKEN = _tokenizer.pad_token_id

V = 100
D_MODEL = 32
N_HEAD = 4
N_LAYERS = 2
N_CTX = 32
B = 2
T = 16


class DictDataset(Dataset):
    """Simple dataset that returns dicts, usable by both DataLoader and direct indexing."""
    
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
    
    def __len__(self):
        return self.input_ids.shape[0]
    
    def __getitem__(self, idx):
        item = {"input_ids": self.input_ids[idx]}
        if self.attention_mask is not None:
            item["attention_mask"] = self.attention_mask[idx]
        return item


def _collate_fn(batch):
    """Collate dicts into batched dicts."""
    result = {"input_ids": torch.stack([b["input_ids"] for b in batch])}
    if "attention_mask" in batch[0]:
        result["attention_mask"] = torch.stack([b["attention_mask"] for b in batch])
    return result


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
            "use_bias_qk": True,
        },
    }


@pytest.fixture
def dummy_dataloader():
    """Create a simple dummy dataloader for testing (no padding)."""
    input_ids = torch.randint(0, V, (50, T))
    dataset = DictDataset(input_ids)
    return DataLoader(dataset, batch_size=B, shuffle=False, collate_fn=_collate_fn)


@pytest.fixture
def padded_dataloader():
    """Create a dataloader with padding and attention_mask for testing."""
    real_len = T // 2  # 8 real tokens per sequence
    input_ids = torch.full((50, T), PAD_TOKEN, dtype=torch.long)
    attention_mask = torch.zeros(50, T, dtype=torch.long)
    
    # Real tokens: pick from range [2, V) to guarantee they differ from PAD_TOKEN
    input_ids[:, :real_len] = torch.randint(2, V, (50, real_len))
    attention_mask[:, :real_len] = 1
    attention_mask[:, real_len:] = 0
    
    dataset = DictDataset(input_ids, attention_mask)
    return DataLoader(dataset, batch_size=B, shuffle=False, collate_fn=_collate_fn)


@pytest.fixture
def model(tiny_config):
    """Create a small model for testing."""
    from models import AttentionLM
    torch.manual_seed(42)
    return AttentionLM.from_config(tiny_config)


@pytest.fixture
def make_tracker(model, dummy_dataloader, tmp_path):
    """Factory that builds a BehaviourTracker with sensible defaults.

    Call as ``make_tracker()`` or ``make_tracker(config=..., run_dir=...)``.
    """
    from analysis.behaviour.tracker import BehaviourTracker

    _UNSET = object()

    def _factory(config=None, run_dir=_UNSET, cache_dir=None):
        if run_dir is _UNSET:
            run_dir = tmp_path  # default to a fresh tmp dir
        return BehaviourTracker(
            model=model,
            train_dataloader=dummy_dataloader,
            val_dataloader=dummy_dataloader,
            vocab_size=V,
            run_dir=str(run_dir) if run_dir is not None else None,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            config=config,
        )

    return _factory


def save_fake_checkpoint(model, path, step):
    """Save a checkpoint in the same format as ``Trainer.save_checkpoint``."""
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
    }, path)
