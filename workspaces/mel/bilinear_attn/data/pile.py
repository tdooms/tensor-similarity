"""DSIR-filtered Pile dataloaders (streaming train + cached val)."""

import hashlib
import re
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoTokenizer

PILE_DATASET_REPO = "stanford-crfm/DSIR-filtered-pile-50M"
PILE_TEXT_FIELD = "contents"
PILE_TOKENIZER = "gpt2"

_CACHE_DIR = Path(__file__).resolve().parent.parent / "cached_tokens"


def _tokenizer_tag(tokenizer_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", tokenizer_name)


def _split_tag(split_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", split_name)


def _compute_holdout_bucket(text: str, holdout_seed: int, holdout_modulus: int) -> int:
    payload = f"{holdout_seed}:{text}".encode("utf-8", errors="replace")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % holdout_modulus


def _load_tokenizer(tokenizer_name: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _assert_vocab_compat(tokenizer, expected_vocab_size: int) -> None:
    actual_vocab_size = int(tokenizer.vocab_size)
    if actual_vocab_size != expected_vocab_size:
        raise ValueError(
            f"Tokenizer vocab mismatch: tokenizer.vocab_size={actual_vocab_size}, "
            f"expected model vocab_size={expected_vocab_size}. "
            "Train/load a tokenizer with matching vocab size."
        )


class DSIRPileStreaming(IterableDataset):
    """Stream DSIR-filtered Pile, tokenize with GPT-2, and emit fixed-size windows."""

    def __init__(
        self,
        n_ctx: int,
        vocab_size: int,
        tokenizer_name: str = PILE_TOKENIZER,
        dataset_repo: str = PILE_DATASET_REPO,
        text_field: str = PILE_TEXT_FIELD,
        split: str = "train",
        dataset_revision: str | None = None,
        shuffle: bool = False,
        shuffle_seed: int = 0,
        shuffle_buffer_size: int = 10_000,
        insert_eos_between_documents: bool = True,
        holdout_modulus: int = 0,
        holdout_remainder: int = 0,
        include_holdout: bool = False,
        exclude_holdout: bool = False,
    ):
        self.n_ctx = n_ctx
        self.vocab_size = vocab_size
        self.tokenizer_name = tokenizer_name
        self.dataset_repo = dataset_repo
        self.text_field = text_field
        self.split = split
        self.dataset_revision = dataset_revision
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self.insert_eos_between_documents = insert_eos_between_documents
        self.holdout_modulus = holdout_modulus
        self.holdout_remainder = holdout_remainder
        self.include_holdout = include_holdout
        self.exclude_holdout = exclude_holdout

    def __iter__(self):
        from datasets import load_dataset

        tokenizer = _load_tokenizer(self.tokenizer_name)
        _assert_vocab_compat(tokenizer, self.vocab_size)
        ds = load_dataset(
            self.dataset_repo,
            split=self.split,
            streaming=True,
            revision=self.dataset_revision,
        )
        if self.shuffle:
            ds = ds.shuffle(seed=self.shuffle_seed, buffer_size=self.shuffle_buffer_size)

        eos_token_id = tokenizer.eos_token_id
        if self.insert_eos_between_documents and eos_token_id is None:
            raise ValueError("insert_eos_between_documents=True requires tokenizer.eos_token_id")
        if self.include_holdout and self.exclude_holdout:
            raise ValueError("Cannot set both include_holdout and exclude_holdout")
        if (self.include_holdout or self.exclude_holdout) and self.holdout_modulus <= 0:
            raise ValueError("holdout_modulus must be > 0 when include/exclude holdout is enabled")

        token_buffer: list[int] = []
        for example in ds:
            text = example[self.text_field]
            if self.include_holdout or self.exclude_holdout:
                bucket = _compute_holdout_bucket(text, self.shuffle_seed, self.holdout_modulus)
                is_holdout_doc = bucket == self.holdout_remainder
                if self.include_holdout and not is_holdout_doc:
                    continue
                if self.exclude_holdout and is_holdout_doc:
                    continue

            tokens = tokenizer.encode(text, add_special_tokens=False)
            if self.insert_eos_between_documents:
                tokens.append(eos_token_id)
            token_buffer.extend(tokens)

            while len(token_buffer) >= self.n_ctx:
                chunk = token_buffer[: self.n_ctx]
                token_buffer = token_buffer[self.n_ctx :]
                yield {"input_ids": torch.tensor(chunk, dtype=torch.long)}


class CachedTokenWindows(Dataset):
    """Simple cached token windows for eval."""

    def __init__(self, path: str | Path, n_ctx: int, max_samples: Optional[int] = None):
        data = torch.load(path, weights_only=True).to(torch.long)[:, :n_ctx]
        if max_samples is not None:
            data = data[:max_samples]
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return {"input_ids": self.data[idx]}


def cache_pile_val(
    n_ctx: int,
    vocab_size: int,
    n_val: int = 500,
    cache_dir: Optional[str | Path] = None,
    tokenizer_name: str = PILE_TOKENIZER,
    dataset_repo: str = PILE_DATASET_REPO,
    text_field: str = PILE_TEXT_FIELD,
    val_split: str = "validation",
    dataset_revision: str | None = None,
    insert_eos_between_documents: bool = True,
    holdout_fraction: float = 0.01,
    holdout_seed: int = 0,
) -> Path:
    """Cache Pile validation windows once and reuse them across runs."""
    if cache_dir is None:
        cache_dir = _CACHE_DIR

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tok_tag = _tokenizer_tag(tokenizer_name)
    split_tag = _split_tag(val_split)
    rev_tag = "floating" if dataset_revision is None else dataset_revision[:12]
    eos_tag = "eos1" if insert_eos_between_documents else "eos0"
    holdout_tag = ""
    if val_split == "heldout":
        holdout_pct = int(round(holdout_fraction * 10_000))
        holdout_tag = f"_holdout{holdout_pct:04d}_seed{holdout_seed}"
    val_path = cache_dir / f"dsir_pile_val_{split_tag}{holdout_tag}_ctx{n_ctx}_vocab{vocab_size}_{tok_tag}_rev{rev_tag}_{eos_tag}.pt"

    if val_path.exists():
        print(f"Pile val cache exists at {val_path}")
        return val_path

    from datasets import load_dataset

    print(f"Caching {n_val} Pile val windows (n_ctx={n_ctx}, vocab={vocab_size})...")

    tokenizer = _load_tokenizer(tokenizer_name)
    _assert_vocab_compat(tokenizer, vocab_size)
    if val_split == "heldout":
        ds = load_dataset(
            dataset_repo,
            split="train",
            streaming=True,
            revision=dataset_revision,
        )
    else:
        ds = load_dataset(
            dataset_repo,
            split=val_split,
            streaming=True,
            revision=dataset_revision,
        )

    eos_token_id = tokenizer.eos_token_id
    if insert_eos_between_documents and eos_token_id is None:
        raise ValueError("insert_eos_between_documents=True requires tokenizer.eos_token_id")

    all_chunks: list[list[int]] = []
    token_buffer: list[int] = []
    holdout_modulus = max(1, int(round(1.0 / max(1e-6, holdout_fraction))))

    for example in ds:
        text = example[text_field]
        if val_split == "heldout":
            bucket = _compute_holdout_bucket(text, holdout_seed, holdout_modulus)
            if bucket != 0:
                continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        if insert_eos_between_documents:
            tokens.append(eos_token_id)
        token_buffer.extend(tokens)

        while len(token_buffer) >= n_ctx:
            chunk = token_buffer[:n_ctx]
            token_buffer = token_buffer[n_ctx:]
            all_chunks.append(chunk)
            if len(all_chunks) >= n_val:
                break

        if len(all_chunks) >= n_val:
            break

    val_data = torch.tensor(all_chunks, dtype=torch.int16)
    torch.save(val_data, val_path)
    print(f"Saved: {val_path} ({val_data.shape})")
    return val_path


def create_pile_dataloaders(
    n_ctx: int,
    batch_size: int,
    vocab_size: int,
    max_val_samples: int = 500,
    val_cache_size: int = 500,
    cache_dir: Optional[str | Path] = None,
    tokenizer_name: str = PILE_TOKENIZER,
    dataset_repo: str = PILE_DATASET_REPO,
    text_field: str = PILE_TEXT_FIELD,
    train_split: str = "train",
    val_split: str = "validation",
    dataset_revision: str | None = None,
    train_shuffle: bool = True,
    train_shuffle_seed: int = 0,
    train_shuffle_buffer_size: int = 10_000,
    insert_eos_between_documents: bool = True,
    holdout_fraction: float = 0.01,
    holdout_seed: int = 0,
):
    """Create train/val dataloaders for DSIR-filtered Pile."""
    val_path = cache_pile_val(
        n_ctx=n_ctx,
        vocab_size=vocab_size,
        n_val=val_cache_size,
        cache_dir=cache_dir,
        tokenizer_name=tokenizer_name,
        dataset_repo=dataset_repo,
        text_field=text_field,
        val_split=val_split,
        dataset_revision=dataset_revision,
        insert_eos_between_documents=insert_eos_between_documents,
        holdout_fraction=holdout_fraction,
        holdout_seed=holdout_seed,
    )

    holdout_modulus = max(1, int(round(1.0 / max(1e-6, holdout_fraction))))
    use_heldout_partition = val_split == "heldout"

    train_ds = DSIRPileStreaming(
        n_ctx=n_ctx,
        vocab_size=vocab_size,
        tokenizer_name=tokenizer_name,
        dataset_repo=dataset_repo,
        text_field=text_field,
        split=train_split,
        dataset_revision=dataset_revision,
        shuffle=train_shuffle,
        shuffle_seed=train_shuffle_seed,
        shuffle_buffer_size=train_shuffle_buffer_size,
        insert_eos_between_documents=insert_eos_between_documents,
        holdout_modulus=holdout_modulus,
        holdout_remainder=0,
        include_holdout=False,
        exclude_holdout=use_heldout_partition,
    )
    val_ds = CachedTokenWindows(val_path, n_ctx=n_ctx, max_samples=max_val_samples)

    train_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_dl, val_dl
