"""Training utilities for BilinearMLP models."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any

from model import BilinearMLP


def compute_accuracy(model: BilinearMLP, loader, device: str) -> float:
    """
    Compute classification accuracy.

    Args:
        model: BilinearMLP model
        loader: DataLoader
        device: Device string

    Returns:
        Accuracy in [0, 1]
    """
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x_a, x_b, y in loader:
            x_a, x_b, y = x_a.to(device), x_b.to(device), y.to(device)
            pred = model(x_a, x_b).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total


def train_model(
    seed: int,
    train_loader,
    test_loader,
    config: Dict[str, Any]
) -> Tuple[BilinearMLP, List[Dict[str, Any]]]:
    """
    Train a single BilinearMLP model.

    Args:
        seed: Random seed for this training run
        train_loader: Training data loader
        test_loader: Test data loader
        config: Configuration dictionary

    Returns:
        (trained_model, list_of_checkpoint_info)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = config["device"]
    p = config["p"]

    model = BilinearMLP(
        input_dim=p,
        hidden_dim=config["hidden_dim"],
        output_dim=p
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    criterion = nn.CrossEntropyLoss()

    checkpoints = []

    for epoch in range(1, config["num_epochs"] + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for x_a, x_b, y in train_loader:
            x_a, x_b, y = x_a.to(device), x_b.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x_a, x_b)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # Record checkpoint info at specified epochs
        if epoch in config["checkpoint_epochs"]:
            train_acc = compute_accuracy(model, train_loader, device)
            test_acc = compute_accuracy(model, test_loader, device)
            avg_loss = total_loss / num_batches

            print(f"Seed {seed}, Epoch {epoch}: "
                  f"loss={avg_loss:.4f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

            checkpoints.append({
                "epoch": epoch,
                "loss": avg_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
            })

    return model, checkpoints
