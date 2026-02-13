"""Repeated-token data generation for induction head experiments."""

import torch
from torch.utils.data import Dataset, DataLoader


class RepeatedTokenDataset(Dataset):
    """Dataset that generates sequences of the form [random | random].

    Each sample is a sequence of length ``2 * seq_len`` where the second half
    is an exact copy of the first half.  A model with induction heads should
    learn to predict the repeated tokens with high accuracy.

    The tokens are drawn uniformly from ``[0, vocab_size)``.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        n_samples: int = 10_000,
        seed: int = 42,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_samples = n_samples

        gen = torch.Generator().manual_seed(seed)
        half = torch.randint(0, vocab_size, (n_samples, seq_len), generator=gen)
        self.data = torch.cat([half, half], dim=-1)  # (n_samples, 2*seq_len)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {"input_ids": self.data[idx]}


def create_repeated_token_dataloaders(
    vocab_size: int,
    seq_len: int,
    batch_size: int = 64,
    n_train: int = 50_000,
    n_val: int = 2_000,
    seed: int = 42,
):
    """Create train and validation dataloaders of repeated-token sequences.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        seq_len: Length of each half (full sequence is ``2 * seq_len``).
        batch_size: Batch size.
        n_train: Number of training samples.
        n_val: Number of validation samples.
        seed: Random seed.

    Returns:
        (train_dataloader, val_dataloader)
    """
    train_ds = RepeatedTokenDataset(vocab_size, seq_len, n_samples=n_train, seed=seed)
    val_ds = RepeatedTokenDataset(vocab_size, seq_len, n_samples=n_val, seed=seed + 1)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl
