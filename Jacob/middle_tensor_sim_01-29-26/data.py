"""Data utilities for modular addition task."""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List


class ModularAdditionDataset(Dataset):
    """
    Dataset for modular addition: (a, b) -> (a + b) mod p.
    Each sample consists of two one-hot encoded inputs and the target class.
    """
    def __init__(self, p: int, indices: Optional[List[int]] = None):
        """
        Args:
            p: The modulus
            indices: Optional list of indices into all p² pairs. If None, uses all pairs.
        """
        self.p = p
        # All p² pairs
        all_pairs = [(a, b) for a in range(p) for b in range(p)]
        if indices is not None:
            all_pairs = [all_pairs[i] for i in indices]
        self.pairs = all_pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        a, b = self.pairs[idx]
        # One-hot encode each input separately
        x_a = F.one_hot(torch.tensor(a), self.p).float()
        x_b = F.one_hot(torch.tensor(b), self.p).float()
        y = (a + b) % self.p
        return x_a, x_b, y


def get_dataloaders(
    p: int,
    train_fraction: float,
    batch_size: int,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test data loaders with a random split.

    Args:
        p: The modulus
        train_fraction: Fraction of data for training
        batch_size: Batch size
        seed: Random seed for reproducible split

    Returns:
        (train_loader, test_loader)
    """
    n_total = p * p

    # Use fixed seed for reproducible data split
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator).tolist()

    n_train = int(n_total * train_fraction)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    train_ds = ModularAdditionDataset(p, train_indices)
    test_ds = ModularAdditionDataset(p, test_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_full_test_loader(p: int, batch_size: int) -> DataLoader:
    """
    Get a data loader for all p² pairs (for comprehensive evaluation).

    Args:
        p: The modulus
        batch_size: Batch size

    Returns:
        DataLoader for all pairs
    """
    ds = ModularAdditionDataset(p)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)
