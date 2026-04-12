"""Data loading for SimpleStories using cached per-story tokens."""
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "cached_tokens"


class _SimpleStoriesDict(Dataset):
    def __init__(self, split, n_ctx, max_samples=None):
        path = CACHE_DIR / f"{split}_perstory.pt"
        data = torch.load(path, weights_only=True).to(torch.long)
        self.data = data[:, :n_ctx]
        if max_samples is not None:
            self.data = self.data[:max_samples]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return {"input_ids": self.data[idx]}


def create_dataloaders(n_ctx, batch_size=64, max_train_samples=None, max_val_samples=1000):
    train_ds = _SimpleStoriesDict("train", n_ctx, max_samples=max_train_samples)
    val_ds = _SimpleStoriesDict("test", n_ctx, max_samples=max_val_samples)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_dl, val_dl


def get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-V2-1.25M")
