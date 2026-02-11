"""
SimpleStories dataset loading for next-token prediction.

Uses pre-tokenized cached data. On first run (or if cache is missing), downloads
SimpleStories from HuggingFace and tokenizes with SimpleStories/SimpleStories-V2-1.25M.

Cached data: ~2.37M train chunks, ~24k test chunks, each (seq_len+1) tokens as int16.
Vocab: 4019 tokens (indices 0-4018).
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "cached_tokens"
VOCAB_SIZE = 4019
DEFAULT_SEQ_LEN = 256


def get_tokenizer():
    """Load the SimpleStories V2 tokenizer (vocab=4019)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-V2-1.25M")


def _tokenize_and_cache(seq_len=DEFAULT_SEQ_LEN):
    """Download SimpleStories, tokenize, and cache to disk. Run once."""
    from datasets import load_dataset

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = get_tokenizer()

    def tokenize_and_chunk(dataset, sl):
        all_tokens = []
        for example in dataset:
            tokens = tokenizer.encode(example["story"], add_special_tokens=False)
            tokens.append(tokenizer.eos_token_id or 1)
            all_tokens.extend(tokens)
        all_tokens = torch.tensor(all_tokens, dtype=torch.long)
        n_chunks = len(all_tokens) // (sl + 1)
        all_tokens = all_tokens[: n_chunks * (sl + 1)]
        return all_tokens.view(n_chunks, sl + 1)

    for split in ["train", "test"]:
        print(f"Tokenizing {split} split...")
        data = tokenize_and_chunk(
            load_dataset("SimpleStories/SimpleStories", split=split), seq_len
        )
        path = CACHE_DIR / f"{split}.pt"
        torch.save(data.to(torch.int16), path)
        mb = os.path.getsize(path) / 1e6
        print(f"  {split}: {data.shape[0]} chunks, {mb:.0f}MB -> {path}")

    print("Caching complete.")


def _ensure_cache():
    """Generate cached tokens if they don't exist yet."""
    train_path = CACHE_DIR / "train.pt"
    test_path = CACHE_DIR / "test.pt"

    # Also check the external tn_4_interp cache as a fallback
    ext_cache = Path("/workspace/tn_4_interp/language/cached_tokens")
    ext_train = ext_cache / "train.pt"
    ext_test = ext_cache / "test.pt"

    if train_path.exists() and test_path.exists():
        return  # local cache exists

    if ext_train.exists() and ext_test.exists():
        # Symlink from external cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Symlinking cached tokens from {ext_cache}/")
        os.symlink(ext_train, train_path)
        os.symlink(ext_test, test_path)
        return

    # No cache anywhere — generate from scratch
    print("No cached tokens found. Downloading and tokenizing SimpleStories...")
    _tokenize_and_cache()


class SimpleStoriesDataset(Dataset):
    """Pre-tokenized SimpleStories dataset for next-token prediction."""

    def __init__(self, split="train", n_ctx=DEFAULT_SEQ_LEN):
        _ensure_cache()

        self.n_ctx = n_ctx
        path = CACHE_DIR / f"{split}.pt"
        data = torch.load(path, weights_only=True).to(torch.long)
        cached_ctx = data.shape[1] - 1

        if n_ctx <= cached_ctx:
            self.data = data[:, : n_ctx + 1]
        else:
            raise ValueError(f"n_ctx={n_ctx} > cached seq_len={cached_ctx}")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return chunk[:-1], chunk[1:]  # (input, target)


def get_dataloader(split="train", n_ctx=DEFAULT_SEQ_LEN, batch_size=64):
    ds = SimpleStoriesDataset(split=split, n_ctx=n_ctx)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), drop_last=True)
