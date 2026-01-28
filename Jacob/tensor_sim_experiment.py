"""
Tensor Similarity Experiment for Bilinear MLPs on MNIST

This script trains bilinear MLPs and tests whether "tensor similarity"
(a weight-space metric) correlates with functional similarity (output agreement).

Tensor similarity is defined as symmetric bilinear model cosine similarity:
- For models with weights (W_l, W_r, W_p), compute symmetry-aware inner product
- Handles the W_l <-> W_r swap symmetry inherent in bilinear layers

Author: Jacob (with Claude Code assistance)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from itertools import combinations
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "num_seeds": 5,
    "num_checkpoints": 10,
    "hidden_dim": 128,
    "input_dim": 784,  # 28x28 flattened
    "output_dim": 10,  # MNIST classes
    "num_epochs": 20,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "device": "mps" if torch.backends.mps.is_available() else "cpu",
    "checkpoint_dir": Path("checkpoints"),
    "results_dir": Path("results"),
}

print(f"Using device: {CONFIG['device']}")


# =============================================================================
# MODEL DEFINITION
# =============================================================================

class BilinearMLP(nn.Module):
    """
    A simple MLP with one bilinear hidden layer.

    The bilinear computation is:
        h = (W_l @ x) ⊙ (W_r @ x)    # element-wise product of two linear projections
        output = W_p @ h              # project to output classes

    This architecture has a natural W_l <-> W_r swap symmetry since multiplication
    is commutative.

    Attributes:
        W_l: Left projection matrix, shape (hidden_dim, input_dim)
        W_r: Right projection matrix, shape (hidden_dim, input_dim)
        W_p: Output projection matrix, shape (output_dim, hidden_dim)
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        # Store dimensions for easy access
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Define the three weight matrices explicitly (no bias for simplicity)
        # Using nn.Parameter so they're registered as model parameters
        self.W_l = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.01)
        self.W_r = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.01)
        self.W_p = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the bilinear MLP.

        Args:
            x: Input tensor, shape (batch_size, input_dim)

        Returns:
            Logits tensor, shape (batch_size, output_dim)
        """
        # x: (batch, input_dim)
        # W_l @ x.T -> (hidden, batch), then transpose -> (batch, hidden)
        left = x @ self.W_l.T   # (batch, hidden)
        right = x @ self.W_r.T  # (batch, hidden)

        # Bilinear combination: element-wise product
        h = left * right  # (batch, hidden)

        # Output projection
        output = h @ self.W_p.T  # (batch, output)

        return output


# =============================================================================
# TENSOR SIMILARITY
# =============================================================================

def tensor_inner_product(model1: BilinearMLP, model2: BilinearMLP) -> float:
    """
    Compute the symmetric bilinear inner product between two models.

    This is the core of tensor similarity. It computes a symmetry-aware
    inner product that handles the W_l <-> W_r swap symmetry.

    Algorithm:
        1. Compute four gram matrices:
           ll = W_l1.T @ W_l2  (left-left alignment)
           rr = W_r1.T @ W_r2  (right-right alignment)
           lr = W_l1.T @ W_r2  (left-right cross)
           rl = W_r1.T @ W_l2  (right-left cross)

        2. Core = 0.5 * ((ll ⊙ rr) + (lr ⊙ rl))
           This averages the "aligned" term (ll⊙rr) with the "swapped" term (lr⊙rl)
           to achieve symmetry under W_l <-> W_r exchange

        3. Combine with output projections: dd = W_p1 @ W_p2.T

        4. Inner product = Tr(core @ dd)

    Args:
        model1: First bilinear MLP
        model2: Second bilinear MLP

    Returns:
        The symmetric inner product (a scalar)
    """
    # Extract weights and move to CPU for numerical stability
    W_l1 = model1.W_l.detach().cpu()
    W_r1 = model1.W_r.detach().cpu()
    W_p1 = model1.W_p.detach().cpu()

    W_l2 = model2.W_l.detach().cpu()
    W_r2 = model2.W_r.detach().cpu()
    W_p2 = model2.W_p.detach().cpu()

    # Step 1: Compute gram matrices
    # W_l is (hidden, input), so W_l.T @ W_l2 is (input, input)
    # Wait, let me reconsider the dimensions...
    # Actually, we want to compare how the hidden dimensions align
    # ll should be (hidden, hidden): W_l1 @ W_l2.T
    # Let me re-read the formula...

    # The formula says: ll = W_l1^T @ W_l2
    # If W_l is (hidden, input), then W_l1^T is (input, hidden)
    # W_l1^T @ W_l2 would be (input, hidden) @ (hidden, input) = (input, input)
    # That doesn't seem right for the trace computation later...

    # Let me interpret it differently:
    # Perhaps the convention is W_l is (input, hidden)?
    # Or perhaps it's: ll = W_l1 @ W_l2.T = (hidden, input) @ (input, hidden) = (hidden, hidden)
    # This makes more sense for computing Tr(core @ dd) where dd is (hidden, hidden)

    # I'll use: gram matrix = W1 @ W2.T giving (hidden, hidden) matrices
    ll = W_l1 @ W_l2.T  # (hidden, hidden)
    rr = W_r1 @ W_r2.T  # (hidden, hidden)
    lr = W_l1 @ W_r2.T  # (hidden, hidden)
    rl = W_r1 @ W_l2.T  # (hidden, hidden)

    # Step 2: Compute symmetrized core
    # Element-wise products, then average aligned and swapped terms
    aligned = ll * rr   # (hidden, hidden)
    swapped = lr * rl   # (hidden, hidden)
    core = 0.5 * (aligned + swapped)  # (hidden, hidden)

    # Step 3: Output projection gram matrix
    # W_p is (output, hidden), so W_p1 @ W_p2.T is (output, output)
    # But we need (hidden, hidden) to match core...
    # So it should be W_p1.T @ W_p2 = (hidden, output) @ (output, hidden) = (hidden, hidden)
    dd = W_p1.T @ W_p2  # (hidden, hidden)

    # Step 4: Inner product via trace
    inner = torch.trace(core @ dd).item()

    return inner


def tensor_similarity(model1: BilinearMLP, model2: BilinearMLP) -> float:
    """
    Compute tensor similarity (cosine similarity) between two bilinear models.

    This normalizes the inner product to get a cosine-like similarity in [−1, 1].

    Formula:
        tensor_sim(M1, M2) = inner(M1, M2) / sqrt(inner(M1, M1) * inner(M2, M2))

    Args:
        model1: First bilinear MLP
        model2: Second bilinear MLP

    Returns:
        Tensor similarity (cosine similarity), a value typically in [-1, 1]
    """
    inner_12 = tensor_inner_product(model1, model2)
    inner_11 = tensor_inner_product(model1, model1)
    inner_22 = tensor_inner_product(model2, model2)

    # Avoid division by zero
    denominator = np.sqrt(inner_11 * inner_22)
    if denominator < 1e-10:
        return 0.0

    return inner_12 / denominator


# =============================================================================
# FUNCTIONAL SIMILARITY
# =============================================================================

def compute_output_agreement(
    model1: BilinearMLP,
    model2: BilinearMLP,
    dataloader: DataLoader,
    device: str
) -> float:
    """
    Compute output agreement rate between two models.

    Agreement rate = fraction of inputs where both models predict the same class.

    Args:
        model1: First model
        model2: Second model
        dataloader: DataLoader for evaluation (typically test set)
        device: Device to run evaluation on

    Returns:
        Agreement rate in [0, 1]
    """
    model1.eval()
    model2.eval()

    total = 0
    agreements = 0

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.view(images.size(0), -1).to(device)  # Flatten

            logits1 = model1(images)
            logits2 = model2(images)

            pred1 = logits1.argmax(dim=1)
            pred2 = logits2.argmax(dim=1)

            agreements += (pred1 == pred2).sum().item()
            total += images.size(0)

    return agreements / total


def compute_accuracy(model: BilinearMLP, dataloader: DataLoader, device: str) -> float:
    """
    Compute classification accuracy for a model.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation
        device: Device to run on

    Returns:
        Accuracy in [0, 1]
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)

            logits = model(images)
            pred = logits.argmax(dim=1)

            correct += (pred == labels).sum().item()
            total += labels.size(0)

    return correct / total


# =============================================================================
# TRAINING
# =============================================================================

def train_model(
    seed: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: dict
) -> tuple[BilinearMLP, list[dict]]:
    """
    Train a bilinear MLP with a given seed, saving checkpoints along the way.

    Args:
        seed: Random seed for reproducibility
        train_loader: Training data
        test_loader: Test data (for checkpoint evaluation)
        config: Configuration dictionary

    Returns:
        Tuple of (final_model, list_of_checkpoint_info)
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = config["device"]

    # Initialize model
    model = BilinearMLP(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Determine checkpoint epochs (evenly spaced)
    num_epochs = config["num_epochs"]
    num_checkpoints = config["num_checkpoints"]
    checkpoint_epochs = np.linspace(1, num_epochs, num_checkpoints, dtype=int)
    checkpoint_epochs = sorted(set(checkpoint_epochs))  # Remove duplicates

    # Create checkpoint directory for this seed
    ckpt_dir = config["checkpoint_dir"] / f"seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = []

    print(f"\n--- Training seed {seed} ---")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Save checkpoint if this is a checkpoint epoch
        if epoch in checkpoint_epochs:
            accuracy = compute_accuracy(model, test_loader, device)

            ckpt_path = ckpt_dir / f"epoch_{epoch:02d}.pt"
            torch.save({
                "epoch": epoch,
                "seed": seed,
                "model_state_dict": model.state_dict(),
                "accuracy": accuracy,
            }, ckpt_path)

            checkpoints.append({
                "epoch": epoch,
                "path": str(ckpt_path),
                "accuracy": accuracy,
                "loss": avg_loss,
            })

            print(f"  Epoch {epoch:2d}: loss={avg_loss:.4f}, acc={accuracy:.4f} [checkpoint saved]")
        else:
            print(f"  Epoch {epoch:2d}: loss={avg_loss:.4f}")

    return model, checkpoints


# =============================================================================
# DATA LOADING
# =============================================================================

def get_data_loaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    """
    Get MNIST train and test data loaders.

    Args:
        batch_size: Batch size for data loaders

    Returns:
        Tuple of (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_experiment(config: dict):
    """
    Run the full tensor similarity experiment.

    Steps:
        1. Load MNIST data
        2. Train models with different seeds, saving checkpoints
        3. Compute pairwise tensor similarity and functional similarity
        4. Analyze correlation and create visualizations
    """
    print("=" * 60)
    print("TENSOR SIMILARITY EXPERIMENT")
    print("=" * 60)

    # Setup directories
    config["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    config["results_dir"].mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading MNIST data...")
    train_loader, test_loader = get_data_loaders(config["batch_size"])
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")

    # Train models
    print("\n" + "=" * 60)
    print("PHASE 1: TRAINING MODELS")
    print("=" * 60)

    models = {}
    all_checkpoints = {}

    for seed in range(config["num_seeds"]):
        model, checkpoints = train_model(seed, train_loader, test_loader, config)
        models[seed] = model
        all_checkpoints[seed] = checkpoints

    # Save checkpoint metadata
    with open(config["results_dir"] / "checkpoint_metadata.json", "w") as f:
        # Convert to serializable format
        serializable = {str(k): v for k, v in all_checkpoints.items()}
        json.dump(serializable, f, indent=2)

    # Compute pairwise metrics
    print("\n" + "=" * 60)
    print("PHASE 2: COMPUTING PAIRWISE METRICS")
    print("=" * 60)

    pairs = list(combinations(range(config["num_seeds"]), 2))
    print(f"\nComputing metrics for {len(pairs)} pairs...")

    results = []

    for i, j in pairs:
        model_i = models[i]
        model_j = models[j]

        # Tensor similarity
        t_sim = tensor_similarity(model_i, model_j)

        # Functional similarity (output agreement)
        agreement = compute_output_agreement(model_i, model_j, test_loader, config["device"])

        # Get individual accuracies
        acc_i = compute_accuracy(model_i, test_loader, config["device"])
        acc_j = compute_accuracy(model_j, test_loader, config["device"])

        results.append({
            "seed_i": i,
            "seed_j": j,
            "tensor_sim": t_sim,
            "agreement": agreement,
            "accuracy_i": acc_i,
            "accuracy_j": acc_j,
        })

        print(f"  Pair ({i}, {j}): tensor_sim={t_sim:.4f}, agreement={agreement:.4f}")

    # Analysis
    print("\n" + "=" * 60)
    print("PHASE 3: ANALYSIS")
    print("=" * 60)

    tensor_sims = [r["tensor_sim"] for r in results]
    agreements = [r["agreement"] for r in results]

    # Compute correlations
    pearson_r, pearson_p = stats.pearsonr(tensor_sims, agreements)
    spearman_r, spearman_p = stats.spearmanr(tensor_sims, agreements)

    print(f"\nCorrelation between tensor similarity and output agreement:")
    print(f"  Pearson:  r = {pearson_r:.4f}, p = {pearson_p:.4f}")
    print(f"  Spearman: r = {spearman_r:.4f}, p = {spearman_p:.4f}")

    # Save results
    results_summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
        "pairwise_results": results,
        "correlations": {
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
        },
        "model_accuracies": {
            seed: compute_accuracy(models[seed], test_loader, config["device"])
            for seed in range(config["num_seeds"])
        }
    }

    with open(config["results_dir"] / "results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    # Create scatter plot
    print("\nGenerating scatter plot...")

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(tensor_sims, agreements, alpha=0.7, s=100)

    # Add labels for each point
    for r in results:
        ax.annotate(
            f"({r['seed_i']},{r['seed_j']})",
            (r["tensor_sim"], r["agreement"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8
        )

    ax.set_xlabel("Tensor Similarity", fontsize=12)
    ax.set_ylabel("Output Agreement Rate", fontsize=12)
    ax.set_title(
        f"Tensor Similarity vs Functional Similarity\n"
        f"Pearson r={pearson_r:.3f}, Spearman r={spearman_r:.3f}",
        fontsize=14
    )

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(config["results_dir"] / "tensor_sim_vs_agreement.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {config['results_dir']}/")
    print("  - results.json: Full results data")
    print("  - tensor_sim_vs_agreement.png: Scatter plot")
    print("  - checkpoint_metadata.json: Checkpoint info")

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

    return results_summary


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    results = run_experiment(CONFIG)
