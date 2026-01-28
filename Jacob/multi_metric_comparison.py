"""
Multi-Metric Tensor Comparison

Computes many different metrics for comparing bilinear models,
then checks correlation with functional similarity (output agreement).

Metrics implemented:
1. Tensor sim (corrected) - our symmetric bilinear cosine similarity
2. Flattened cosine - flatten all weights, compute cosine sim
3. Flattened L2 distance - Euclidean distance between flattened weights
4. Weight correlation - Pearson correlation of flattened weights
5. Layer-wise cosine (averaged) - cosine sim per layer, then average
6. Frobenius difference - sum of Frobenius norms of weight differences
7. Spectral similarity - compare singular value spectra
8. Interaction matrix cosine - compare full interaction tensors
9. Symmetrized interaction cosine - interaction matrices with symmetrization
10. CKA on weights - Centered Kernel Alignment adapted for weights
11. Output projection cosine - just compare W_p
12. Input projection cosine - compare (W_l, W_r) combined
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from itertools import combinations
import json

from tensor_sim_experiment import BilinearMLP, get_data_loaders

RESULTS_DIR = Path("results_multi_metric")
RESULTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# METRIC IMPLEMENTATIONS
# =============================================================================

def get_all_weights(model):
    """Extract all weights as a tuple."""
    return (
        model.W_l.detach().cpu(),
        model.W_r.detach().cpu(),
        model.W_p.detach().cpu()
    )


def flatten_weights(model):
    """Flatten all weights into a single vector."""
    W_l, W_r, W_p = get_all_weights(model)
    return torch.cat([W_l.flatten(), W_r.flatten(), W_p.flatten()])


# --- Metric 1: Tensor Sim (corrected) ---
def metric_tensor_sim_corrected(model1, model2):
    """Corrected symmetric bilinear cosine similarity."""
    W_l1, W_r1, W_p1 = get_all_weights(model1)
    W_l2, W_r2, W_p2 = get_all_weights(model2)

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


# --- Metric 2: Flattened Cosine Similarity ---
def metric_flattened_cosine(model1, model2):
    """Flatten all weights and compute cosine similarity."""
    v1 = flatten_weights(model1)
    v2 = flatten_weights(model2)
    return (v1 @ v2 / (v1.norm() * v2.norm())).item()


# --- Metric 3: Flattened L2 Distance (negated for correlation direction) ---
def metric_flattened_l2(model1, model2):
    """Euclidean distance between flattened weights. Returns negative so higher = more similar."""
    v1 = flatten_weights(model1)
    v2 = flatten_weights(model2)
    return -(v1 - v2).norm().item()


# --- Metric 4: Weight Correlation ---
def metric_weight_correlation(model1, model2):
    """Pearson correlation between flattened weights."""
    v1 = flatten_weights(model1).numpy()
    v2 = flatten_weights(model2).numpy()
    return stats.pearsonr(v1, v2)[0]


# --- Metric 5: Layer-wise Cosine (averaged) ---
def metric_layerwise_cosine(model1, model2):
    """Cosine similarity per layer, then average."""
    W_l1, W_r1, W_p1 = get_all_weights(model1)
    W_l2, W_r2, W_p2 = get_all_weights(model2)

    def cosine(a, b):
        a_flat, b_flat = a.flatten(), b.flatten()
        return (a_flat @ b_flat / (a_flat.norm() * b_flat.norm())).item()

    cos_l = cosine(W_l1, W_l2)
    cos_r = cosine(W_r1, W_r2)
    cos_p = cosine(W_p1, W_p2)

    return (cos_l + cos_r + cos_p) / 3


# --- Metric 6: Frobenius Difference (negated) ---
def metric_frobenius_diff(model1, model2):
    """Sum of Frobenius norms of weight differences. Negated so higher = more similar."""
    W_l1, W_r1, W_p1 = get_all_weights(model1)
    W_l2, W_r2, W_p2 = get_all_weights(model2)

    diff = (W_l1 - W_l2).norm() + (W_r1 - W_r2).norm() + (W_p1 - W_p2).norm()
    return -diff.item()


# --- Metric 7: Spectral Similarity ---
def metric_spectral_similarity(model1, model2):
    """Compare singular value spectra of weight matrices."""
    W_l1, W_r1, W_p1 = get_all_weights(model1)
    W_l2, W_r2, W_p2 = get_all_weights(model2)

    def spectral_cosine(A, B):
        # Get singular values, pad to same length, compute cosine
        s1 = torch.linalg.svdvals(A)
        s2 = torch.linalg.svdvals(B)
        # They should be same length for same-shaped matrices
        return (s1 @ s2 / (s1.norm() * s2.norm())).item()

    spec_l = spectral_cosine(W_l1, W_l2)
    spec_r = spectral_cosine(W_r1, W_r2)
    spec_p = spectral_cosine(W_p1, W_p2)

    return (spec_l + spec_r + spec_p) / 3


# --- Metric 8: Interaction Matrix Cosine (no symmetrization) ---
def metric_interaction_cosine(model1, model2):
    """Compare full interaction tensors directly."""
    W_l1, W_r1, W_p1 = get_all_weights(model1)
    W_l2, W_r2, W_p2 = get_all_weights(model2)

    # M[i,j,k] = W_l[i,j] * W_r[i,k]
    M1 = W_l1[:, :, None] * W_r1[:, None, :]
    M2 = W_l2[:, :, None] * W_r2[:, None, :]

    # Weight by W_p: full tensor T[o,j,k] = sum_i W_p[o,i] * M[i,j,k]
    # Just compare M directly (output-weighted comparison is tensor sim)
    M1_flat = M1.flatten()
    M2_flat = M2.flatten()

    return (M1_flat @ M2_flat / (M1_flat.norm() * M2_flat.norm())).item()


# --- Metric 9: Symmetrized Interaction Cosine ---
def metric_interaction_symmetrized(model1, model2):
    """Interaction matrices with symmetrization (removes antisymmetric part)."""
    W_l1, W_r1, W_p1 = get_all_weights(model1)
    W_l2, W_r2, W_p2 = get_all_weights(model2)

    # M[i,j,k] = W_l[i,j] * W_r[i,k], then symmetrize in j,k
    M1 = W_l1[:, :, None] * W_r1[:, None, :]
    M1 = (M1 + M1.transpose(1, 2)) / 2

    M2 = W_l2[:, :, None] * W_r2[:, None, :]
    M2 = (M2 + M2.transpose(1, 2)) / 2

    M1_flat = M1.flatten()
    M2_flat = M2.flatten()

    return (M1_flat @ M2_flat / (M1_flat.norm() * M2_flat.norm())).item()


# --- Metric 10: CKA on Weights ---
def metric_cka_weights(model1, model2):
    """
    Centered Kernel Alignment adapted for weight matrices.
    Treats each hidden unit as a "sample" and input dims as "features".
    """
    W_l1, W_r1, W_p1 = get_all_weights(model1)
    W_l2, W_r2, W_p2 = get_all_weights(model2)

    def cka(X, Y):
        """CKA between matrices X and Y (samples x features)."""
        # Center the matrices
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)

        # Compute kernels (linear)
        K = X @ X.T
        L = Y @ Y.T

        # HSIC
        def hsic(K, L):
            n = K.shape[0]
            H = torch.eye(n) - torch.ones(n, n) / n
            return torch.trace(K @ H @ L @ H) / ((n - 1) ** 2)

        hsic_kl = hsic(K, L)
        hsic_kk = hsic(K, K)
        hsic_ll = hsic(L, L)

        return (hsic_kl / torch.sqrt(hsic_kk * hsic_ll)).item()

    # CKA for each layer
    cka_l = cka(W_l1, W_l2)
    cka_r = cka(W_r1, W_r2)
    cka_p = cka(W_p1, W_p2)

    return (cka_l + cka_r + cka_p) / 3


# --- Metric 11: Output Projection Cosine ---
def metric_output_cosine(model1, model2):
    """Cosine similarity of just W_p."""
    _, _, W_p1 = get_all_weights(model1)
    _, _, W_p2 = get_all_weights(model2)

    v1, v2 = W_p1.flatten(), W_p2.flatten()
    return (v1 @ v2 / (v1.norm() * v2.norm())).item()


# --- Metric 12: Input Projection Cosine ---
def metric_input_cosine(model1, model2):
    """Cosine similarity of concatenated (W_l, W_r)."""
    W_l1, W_r1, _ = get_all_weights(model1)
    W_l2, W_r2, _ = get_all_weights(model2)

    v1 = torch.cat([W_l1.flatten(), W_r1.flatten()])
    v2 = torch.cat([W_l2.flatten(), W_r2.flatten()])

    return (v1 @ v2 / (v1.norm() * v2.norm())).item()


# --- Collect all metrics ---
METRICS = {
    "tensor_sim_corrected": metric_tensor_sim_corrected,
    "flattened_cosine": metric_flattened_cosine,
    "flattened_l2_neg": metric_flattened_l2,
    "weight_correlation": metric_weight_correlation,
    "layerwise_cosine": metric_layerwise_cosine,
    "frobenius_diff_neg": metric_frobenius_diff,
    "spectral_similarity": metric_spectral_similarity,
    "interaction_cosine": metric_interaction_cosine,
    "interaction_symmetrized": metric_interaction_symmetrized,
    "cka_weights": metric_cka_weights,
    "output_cosine": metric_output_cosine,
    "input_cosine": metric_input_cosine,
}


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def compute_output_agreement(model1, model2, dataloader, device):
    """Compute output agreement rate."""
    model1.eval()
    model2.eval()
    total, agreements = 0, 0

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.view(images.size(0), -1).to(device)
            pred1 = model1(images).argmax(dim=1)
            pred2 = model2(images).argmax(dim=1)
            agreements += (pred1 == pred2).sum().item()
            total += images.size(0)

    return agreements / total


def load_model(seed, epoch, device):
    """Load model from checkpoint."""
    path = f"checkpoints/seed_{seed}/epoch_{epoch:02d}.pt"
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model = BilinearMLP(input_dim=784, hidden_dim=128, output_dim=10)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    return model


def run_multi_metric_experiment():
    print("="*70)
    print("MULTI-METRIC TENSOR COMPARISON")
    print("="*70)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    _, test_loader = get_data_loaders(128)

    # Load models
    print("\nLoading models...")
    models = {seed: load_model(seed, 20, device) for seed in range(5)}

    # Compute all metrics for all pairs
    print("\nComputing metrics for all pairs...")
    pairs = list(combinations(range(5), 2))

    results = []
    for i, j in pairs:
        row = {"seed_i": i, "seed_j": j}

        # Functional similarity
        row["agreement"] = compute_output_agreement(models[i], models[j], test_loader, device)

        # All tensor metrics
        for name, func in METRICS.items():
            try:
                row[name] = func(models[i], models[j])
            except Exception as e:
                print(f"  Error computing {name} for ({i},{j}): {e}")
                row[name] = np.nan

        results.append(row)
        print(f"  Pair ({i},{j}) done")

    # Compute correlations
    print("\n" + "="*70)
    print("CORRELATION WITH OUTPUT AGREEMENT")
    print("="*70)

    agreements = [r["agreement"] for r in results]

    correlations = {}
    for name in METRICS.keys():
        values = [r[name] for r in results]
        if not any(np.isnan(values)):
            pearson_r, pearson_p = stats.pearsonr(values, agreements)
            spearman_r, spearman_p = stats.spearmanr(values, agreements)
            correlations[name] = {
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
            }
            print(f"{name:30s}: Pearson r={pearson_r:+.4f} (p={pearson_p:.3f}), Spearman r={spearman_r:+.4f}")
        else:
            print(f"{name:30s}: FAILED (contains NaN)")

    # Save results (convert numpy types to native Python)
    def to_native(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_native(v) for v in obj]
        return obj

    output = {
        "pairwise_results": to_native(results),
        "correlations": to_native(correlations),
    }
    with open(RESULTS_DIR / "multi_metric_results.json", "w") as f:
        json.dump(output, f, indent=2)

    # Create visualization
    print("\nCreating plots...")

    # Bar chart of correlations
    fig, ax = plt.subplots(figsize=(12, 6))

    names = list(correlations.keys())
    pearson_vals = [correlations[n]["pearson_r"] for n in names]
    spearman_vals = [correlations[n]["spearman_r"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, pearson_vals, width, label='Pearson r', color='steelblue')
    bars2 = ax.bar(x + width/2, spearman_vals, width, label='Spearman r', color='darkorange')

    ax.set_ylabel('Correlation with Output Agreement')
    ax.set_title('Metric Comparison: Correlation with Functional Similarity')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "metric_correlations_bar.png", dpi=150)
    plt.close()

    # Scatter plot grid
    n_metrics = len(METRICS)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for idx, name in enumerate(METRICS.keys()):
        ax = axes[idx]
        values = [r[name] for r in results]
        ax.scatter(values, agreements, alpha=0.7, s=60)
        ax.set_xlabel(name, fontsize=8)
        ax.set_ylabel('Agreement', fontsize=8)

        pr = correlations[name]["pearson_r"]
        ax.set_title(f"r={pr:.3f}", fontsize=10)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for idx in range(len(METRICS), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("All Metrics vs Output Agreement", fontsize=14)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "metric_scatter_grid.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {RESULTS_DIR}/")
    print("  - multi_metric_results.json")
    print("  - metric_correlations_bar.png")
    print("  - metric_scatter_grid.png")

    return output


if __name__ == "__main__":
    run_multi_metric_experiment()
