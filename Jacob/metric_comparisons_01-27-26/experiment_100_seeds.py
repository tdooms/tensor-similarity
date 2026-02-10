"""
100-Seed Tensor Similarity Experiment

Trains 100 MNIST bilinear networks and computes the full 2D correlation matrix
of tensor similarity metrics vs functional similarity metrics.

With 100 seeds, we have 100*99/2 = 4950 pairwise comparisons.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from itertools import combinations
import json
import time

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "num_seeds": 100,
    "hidden_dim": 128,
    "input_dim": 784,
    "output_dim": 10,
    "num_epochs": 20,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
}

RESULTS_DIR = Path("results_100_seeds")
RESULTS_DIR.mkdir(exist_ok=True)

print(f"Device: {CONFIG['device']}")
print(f"Will train {CONFIG['num_seeds']} models")
print(f"Will compute {CONFIG['num_seeds'] * (CONFIG['num_seeds'] - 1) // 2} pairwise comparisons")


# =============================================================================
# MODEL DEFINITION
# =============================================================================

class BilinearMLP(nn.Module):
    """Bilinear MLP: h = (W_l @ x) * (W_r @ x), output = W_p @ h"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.W_l = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.01)
        self.W_r = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.01)
        self.W_p = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = x @ self.W_l.T
        right = x @ self.W_r.T
        h = left * right
        return h @ self.W_p.T


# =============================================================================
# DATA LOADING
# =============================================================================

def get_data_loaders(batch_size: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# =============================================================================
# TRAINING
# =============================================================================

def train_model(seed: int, train_loader: DataLoader, device: str) -> BilinearMLP:
    """Train a single model with the given seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = BilinearMLP(
        input_dim=CONFIG["input_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        output_dim=CONFIG["output_dim"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(CONFIG["num_epochs"]):
        for images, labels in train_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    return model


def compute_accuracy(model: BilinearMLP, test_loader: DataLoader, device: str) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            pred = model(images).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total


# =============================================================================
# TENSOR METRICS
# =============================================================================

def get_weights(model):
    return (
        model.W_l.detach().cpu(),
        model.W_r.detach().cpu(),
        model.W_p.detach().cpu()
    )


def flatten_weights(model):
    W_l, W_r, W_p = get_weights(model)
    return torch.cat([W_l.flatten(), W_r.flatten(), W_p.flatten()])


# 1. Tensor Sim (corrected)
def metric_tensor_sim(model1, model2):
    W_l1, W_r1, W_p1 = get_weights(model1)
    W_l2, W_r2, W_p2 = get_weights(model2)

    def inner(Wl1, Wr1, Wp1, Wl2, Wr2, Wp2):
        ll = Wl1 @ Wl2.T
        rr = Wr1 @ Wr2.T
        lr = Wl1 @ Wr2.T
        rl = Wr1 @ Wl2.T
        core = 0.5 * (ll * rr + lr * rl)
        dd = Wp1.T @ Wp2
        return torch.sum(core * dd).item()

    i12 = inner(W_l1, W_r1, W_p1, W_l2, W_r2, W_p2)
    i11 = inner(W_l1, W_r1, W_p1, W_l1, W_r1, W_p1)
    i22 = inner(W_l2, W_r2, W_p2, W_l2, W_r2, W_p2)

    return i12 / np.sqrt(i11 * i22)


# 2. Flattened Cosine
def metric_flat_cosine(model1, model2):
    v1 = flatten_weights(model1)
    v2 = flatten_weights(model2)
    return (v1 @ v2 / (v1.norm() * v2.norm())).item() # thomas thinks I should add a sqrt here?? discuss with claude


# 3. Flattened L2 (negated)
def metric_flat_l2_neg(model1, model2):
    v1 = flatten_weights(model1)
    v2 = flatten_weights(model2)
    return -(v1 - v2).norm().item()


# 4. Weight Correlation
def metric_weight_corr(model1, model2):
    v1 = flatten_weights(model1).numpy()
    v2 = flatten_weights(model2).numpy()
    return stats.pearsonr(v1, v2)[0]


# 5. Layerwise Cosine
def metric_layerwise_cos(model1, model2):
    W_l1, W_r1, W_p1 = get_weights(model1)
    W_l2, W_r2, W_p2 = get_weights(model2)

    def cosine(a, b):
        a_flat, b_flat = a.flatten(), b.flatten()
        return (a_flat @ b_flat / (a_flat.norm() * b_flat.norm())).item()

    return (cosine(W_l1, W_l2) + cosine(W_r1, W_r2) + cosine(W_p1, W_p2)) / 3


# 6. Frobenius Diff (negated)
def metric_frob_neg(model1, model2):
    W_l1, W_r1, W_p1 = get_weights(model1)
    W_l2, W_r2, W_p2 = get_weights(model2)
    diff = (W_l1 - W_l2).norm() + (W_r1 - W_r2).norm() + (W_p1 - W_p2).norm()
    return -diff.item()


# 7. Spectral Similarity
def metric_spectral(model1, model2):
    W_l1, W_r1, W_p1 = get_weights(model1)
    W_l2, W_r2, W_p2 = get_weights(model2)

    def spectral_cosine(A, B):
        s1 = torch.linalg.svdvals(A)
        s2 = torch.linalg.svdvals(B)
        return (s1 @ s2 / (s1.norm() * s2.norm())).item()

    return (spectral_cosine(W_l1, W_l2) + spectral_cosine(W_r1, W_r2) + spectral_cosine(W_p1, W_p2)) / 3


# 8. Interaction Cosine
def metric_interact_cos(model1, model2):
    W_l1, W_r1, _ = get_weights(model1)
    W_l2, W_r2, _ = get_weights(model2)

    M1 = (W_l1[:, :, None] * W_r1[:, None, :]).flatten()
    M2 = (W_l2[:, :, None] * W_r2[:, None, :]).flatten()

    return (M1 @ M2 / (M1.norm() * M2.norm())).item()


# 9. Symmetrized Interaction
def metric_interact_sym(model1, model2):
    W_l1, W_r1, _ = get_weights(model1)
    W_l2, W_r2, _ = get_weights(model2)

    M1 = W_l1[:, :, None] * W_r1[:, None, :]
    M1 = ((M1 + M1.transpose(1, 2)) / 2).flatten()

    M2 = W_l2[:, :, None] * W_r2[:, None, :]
    M2 = ((M2 + M2.transpose(1, 2)) / 2).flatten()

    return (M1 @ M2 / (M1.norm() * M2.norm())).item()


# 10. CKA on Weights
def metric_cka(model1, model2):
    W_l1, W_r1, W_p1 = get_weights(model1)
    W_l2, W_r2, W_p2 = get_weights(model2)

    def cka(X, Y):
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)
        K = X @ X.T
        L = Y @ Y.T

        def hsic(K, L):
            n = K.shape[0]
            H = torch.eye(n) - torch.ones(n, n) / n
            return torch.trace(K @ H @ L @ H) / ((n - 1) ** 2)

        return (hsic(K, L) / torch.sqrt(hsic(K, K) * hsic(L, L))).item()

    return (cka(W_l1, W_l2) + cka(W_r1, W_r2) + cka(W_p1, W_p2)) / 3


# 11. Output Cosine
def metric_output_cos(model1, model2):
    _, _, W_p1 = get_weights(model1)
    _, _, W_p2 = get_weights(model2)
    v1, v2 = W_p1.flatten(), W_p2.flatten()
    return (v1 @ v2 / (v1.norm() * v2.norm())).item()


# 12. Input Cosine
def metric_input_cos(model1, model2):
    W_l1, W_r1, _ = get_weights(model1)
    W_l2, W_r2, _ = get_weights(model2)
    v1 = torch.cat([W_l1.flatten(), W_r1.flatten()])
    v2 = torch.cat([W_l2.flatten(), W_r2.flatten()])
    return (v1 @ v2 / (v1.norm() * v2.norm())).item()


TENSOR_METRICS = {
    "T: Tensor Sim": metric_tensor_sim,
    "T: Flat Cosine": metric_flat_cosine,
    "T: L2 Dist (neg)": metric_flat_l2_neg,
    "T: Weight Corr": metric_weight_corr,
    "T: Layerwise Cos": metric_layerwise_cos,
    "T: Frob Diff (neg)": metric_frob_neg,
    "T: Spectral Sim": metric_spectral,
    "T: Interact Cos": metric_interact_cos,
    "T: Interact Sym": metric_interact_sym,
    "T: CKA Weights": metric_cka,
    "T: Output Cos": metric_output_cos,
    "T: Input Cos": metric_input_cos,
}


# =============================================================================
# FUNCTIONAL METRICS
# =============================================================================

def compute_functional_metrics(model1, model2, test_loader, device):
    """Compute all functional similarity metrics."""
    model1.eval()
    model2.eval()

    all_logits1, all_logits2 = [], []
    agreements, total = 0, 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.view(images.size(0), -1).to(device)
            logits1 = model1(images)
            logits2 = model2(images)
            all_logits1.append(logits1.cpu())
            all_logits2.append(logits2.cpu())
            agreements += (logits1.argmax(1) == logits2.argmax(1)).sum().item()
            total += images.size(0)

    all_logits1 = torch.cat(all_logits1, dim=0)
    all_logits2 = torch.cat(all_logits2, dim=0)
    probs1 = F.softmax(all_logits1, dim=1)
    probs2 = F.softmax(all_logits2, dim=1)

    # Agreement
    agreement = agreements / total

    # Probability cosine
    dot_p = (probs1 * probs2).sum(dim=1)
    prob_cosine = (dot_p / (probs1.norm(dim=1) * probs2.norm(dim=1) + 1e-8)).mean().item()

    # Logit cosine
    dot_l = (all_logits1 * all_logits2).sum(dim=1)
    logit_cosine = (dot_l / (all_logits1.norm(dim=1) * all_logits2.norm(dim=1) + 1e-8)).mean().item()

    # Negative KL divergence
    kl = (probs1 * (torch.log(probs1 + 1e-8) - torch.log(probs2 + 1e-8))).sum(dim=1)
    neg_kl = -kl.mean().item()

    return {
        "F: Agreement": agreement,
        "F: Prob Cosine": prob_cosine,
        "F: Logit Cosine": logit_cosine,
        "F: Neg KL Div": neg_kl,
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment():
    start_time = time.time()

    print("=" * 70)
    print("100-SEED TENSOR SIMILARITY EXPERIMENT")
    print("=" * 70)

    device = CONFIG["device"]
    train_loader, test_loader = get_data_loaders(CONFIG["batch_size"])

    # Phase 1: Train all models
    print("\n" + "=" * 70)
    print("PHASE 1: TRAINING 100 MODELS")
    print("=" * 70)

    models = {}
    accuracies = {}

    for seed in range(CONFIG["num_seeds"]):
        model = train_model(seed, train_loader, device)
        accuracy = compute_accuracy(model, test_loader, device)
        models[seed] = model
        accuracies[seed] = accuracy

        if (seed + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Trained {seed + 1}/{CONFIG['num_seeds']} models "
                  f"(latest acc: {accuracy:.4f}, elapsed: {elapsed:.1f}s)")

    print(f"\nMean accuracy: {np.mean(list(accuracies.values())):.4f}")
    print(f"Std accuracy: {np.std(list(accuracies.values())):.4f}")

    # Phase 2: Compute pairwise metrics
    print("\n" + "=" * 70)
    print("PHASE 2: COMPUTING PAIRWISE METRICS")
    print("=" * 70)

    pairs = list(combinations(range(CONFIG["num_seeds"]), 2))
    n_pairs = len(pairs)
    print(f"\nComputing {n_pairs} pairwise comparisons...")

    # Initialize storage
    all_data = {name: [] for name in TENSOR_METRICS.keys()}
    all_data.update({"F: Agreement": [], "F: Prob Cosine": [], "F: Logit Cosine": [], "F: Neg KL Div": []})

    for idx, (i, j) in enumerate(pairs):
        # Tensor metrics
        for name, func in TENSOR_METRICS.items():
            try:
                val = func(models[i], models[j])
                all_data[name].append(val)
            except Exception as e:
                all_data[name].append(np.nan)

        # Functional metrics
        func_metrics = compute_functional_metrics(models[i], models[j], test_loader, device)
        for name, val in func_metrics.items():
            all_data[name].append(val)

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - start_time
            print(f"  Computed {idx + 1}/{n_pairs} pairs (elapsed: {elapsed:.1f}s)")

    # Phase 3: Build correlation matrix
    print("\n" + "=" * 70)
    print("PHASE 3: BUILDING CORRELATION MATRIX")
    print("=" * 70)

    metric_names = list(all_data.keys())
    n_metrics = len(metric_names)

    corr_matrix = np.zeros((n_metrics, n_metrics))
    for i, name_i in enumerate(metric_names):
        for j, name_j in enumerate(metric_names):
            vals_i = all_data[name_i]
            vals_j = all_data[name_j]
            if not any(np.isnan(vals_i)) and not any(np.isnan(vals_j)):
                corr_matrix[i, j] = stats.spearmanr(vals_i, vals_j)[0]
            else:
                corr_matrix[i, j] = np.nan

    # Phase 4: Create visualizations
    print("\n" + "=" * 70)
    print("PHASE 4: CREATING VISUALIZATIONS")
    print("=" * 70)

    # Full correlation matrix
    print("\nCreating full correlation matrix heatmap...")
    fig, ax = plt.subplots(figsize=(14, 12))

    cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light", as_cmap=True)

    sns.heatmap(corr_matrix,
                xticklabels=metric_names,
                yticklabels=metric_names,
                annot=True,
                fmt='.2f',
                cmap=cmap,
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                linewidths=0.5,
                annot_kws={"size": 8},
                ax=ax)

    # Color labels
    tensor_color = '#d4e6f1'
    func_color = '#d5f5e3'

    for label in ax.get_xticklabels():
        if label.get_text().startswith("T:"):
            label.set_backgroundcolor(tensor_color)
        else:
            label.set_backgroundcolor(func_color)
        label.set_fontsize(9)

    for label in ax.get_yticklabels():
        if label.get_text().startswith("T:"):
            label.set_backgroundcolor(tensor_color)
        else:
            label.set_backgroundcolor(func_color)
        label.set_fontsize(9)

    plt.title(f"Metric Correlation Matrix (Spearman ρ)\n"
              f"100 seeds, {n_pairs} pairs\n"
              f"T: Tensor metrics (blue) | F: Functional metrics (green)",
              fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "metric_correlation_matrix_100seeds.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Focused tensor vs functional matrix
    print("Creating tensor vs functional focused view...")

    tensor_names = [n for n in metric_names if n.startswith("T:")]
    func_names = [n for n in metric_names if n.startswith("F:")]

    tensor_idx = [metric_names.index(n) for n in tensor_names]
    func_idx = [metric_names.index(n) for n in func_names]

    submatrix = corr_matrix[np.ix_(tensor_idx, func_idx)]

    fig, ax = plt.subplots(figsize=(8, 10))

    sns.heatmap(submatrix,
                xticklabels=[n.replace("F: ", "") for n in func_names],
                yticklabels=[n.replace("T: ", "") for n in tensor_names],
                annot=True,
                fmt='.2f',
                cmap=cmap,
                center=0,
                vmin=-1,
                vmax=1,
                linewidths=0.5,
                annot_kws={"size": 10},
                ax=ax)

    ax.set_xlabel("Functional Similarity Metrics", fontsize=12, fontweight='bold')
    ax.set_ylabel("Tensor Similarity Metrics", fontsize=12, fontweight='bold')
    plt.title(f"Tensor vs Functional Metrics (100 seeds, {n_pairs} pairs)\n(Spearman ρ)",
              fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "tensor_vs_functional_100seeds.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    print("\nSaving results...")

    def to_native(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_native(v) for v in obj]
        return obj

    results = {
        "config": CONFIG,
        "num_pairs": n_pairs,
        "model_accuracies": to_native(accuracies),
        "correlation_matrix": to_native(corr_matrix),
        "metric_names": metric_names,
        "summary_stats": {
            "mean_accuracy": float(np.mean(list(accuracies.values()))),
            "std_accuracy": float(np.std(list(accuracies.values()))),
        }
    }

    with open(RESULTS_DIR / "results_100seeds.json", "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"\nResults saved to {RESULTS_DIR}/")
    print("  - metric_correlation_matrix_100seeds.png")
    print("  - tensor_vs_functional_100seeds.png")
    print("  - results_100seeds.json")

    return results


if __name__ == "__main__":
    run_experiment()
