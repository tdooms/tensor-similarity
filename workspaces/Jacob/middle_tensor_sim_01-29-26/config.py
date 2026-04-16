"""Configuration for the generalized tensor similarity experiment."""

import torch

CONFIG = {
    "p": 113,                    # Modulus for modular addition
    "num_seeds": 100,            # Number of training runs
    "hidden_dim": 128,           # Hidden layer size
    "num_epochs": 500,           # Models grok by ~100 epochs
    "batch_size": 512,           # Large batch for grokking
    "learning_rate": 1e-3,
    "weight_decay": 0.01,        # Helps with grokking
    "train_fraction": 0.5,       # 50% train, 50% test
    "checkpoint_epochs": [100, 250, 500],
    "device": "mps" if torch.backends.mps.is_available() else "cpu",
    "num_workers": 4,            # Parallel training workers
}
