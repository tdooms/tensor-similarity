"""Evaluate pairwise tensor similarity metrics and correlations with functional similarity."""

import json
import torch
from pathlib import Path
from itertools import combinations
from scipy import stats

from config import CONFIG
from data import get_dataloaders
from model import BilinearMLP
from metrics import (
    one_hot_covariance,
    tensor_similarity_standard,
    tensor_similarity_generalized,
    compute_output_agreement,
    compute_logit_cosine,
    compute_kl_divergence,
)


def main():
    config = CONFIG
    device = config["device"]
    p = config["p"]

    print(f"Configuration: p={p}, num_seeds={config['num_seeds']}")
    print(f"Device: {device}")

    # Load test data with same split as training
    print("\nLoading test data...")
    _, test_loader = get_dataloaders(
        p=p,
        train_fraction=config["train_fraction"],
        batch_size=config["batch_size"],
        seed=42
    )
    print(f"Test size: {len(test_loader.dataset)}")

    # Compute covariance matrix for one-hot inputs
    print("\nComputing covariance matrix...")
    Sigma = one_hot_covariance(p)
    print(f"Σ shape: {Sigma.shape}")
    print(f"Σ diagonal (should be (p-1)/p² = {(p-1)/(p*p):.6f}): {Sigma[0,0].item():.6f}")
    print(f"Σ off-diagonal (should be -1/p² = {-1/(p*p):.6f}): {Sigma[0,1].item():.6f}")

    # Load all models
    print("\nLoading models...")
    models = {}
    ckpt_dir = Path("checkpoints")

    for seed in range(config["num_seeds"]):
        ckpt_path = ckpt_dir / f"seed_{seed:03d}.pt"
        if not ckpt_path.exists():
            print(f"Warning: {ckpt_path} not found, skipping seed {seed}")
            continue

        ckpt = torch.load(ckpt_path, map_location='cpu')
        model = BilinearMLP(p, config["hidden_dim"], p)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        models[seed] = model

    print(f"Loaded {len(models)} models")

    if len(models) < 2:
        print("Error: Need at least 2 models to compute pairwise metrics")
        return

    # Compute all pairwise metrics
    print("\nComputing pairwise metrics...")
    seeds = sorted(models.keys())
    pairs = list(combinations(seeds, 2))
    print(f"Total pairs: {len(pairs)}")

    results = []
    for idx, (i, j) in enumerate(pairs):
        if idx % 500 == 0:
            print(f"  Progress: {idx}/{len(pairs)}")

        # Tensor similarity metrics
        ts_std = tensor_similarity_standard(models[i], models[j])
        ts_gen = tensor_similarity_generalized(models[i], models[j], Sigma, Sigma)

        # Functional similarity metrics
        agreement = compute_output_agreement(models[i], models[j], test_loader, device)
        logit_cos = compute_logit_cosine(models[i], models[j], test_loader, device)
        kl_div = compute_kl_divergence(models[i], models[j], test_loader, device)

        row = {
            "seed_i": i,
            "seed_j": j,
            "tensor_sim_standard": ts_std,
            "tensor_sim_generalized": ts_gen,
            "output_agreement": agreement,
            "logit_cosine": logit_cos,
            "kl_divergence": kl_div,
        }
        results.append(row)

    print(f"  Completed all {len(pairs)} pairs")

    # Extract arrays for correlation analysis
    tensor_standard = [r["tensor_sim_standard"] for r in results]
    tensor_generalized = [r["tensor_sim_generalized"] for r in results]
    agreements = [r["output_agreement"] for r in results]
    logit_cosines = [r["logit_cosine"] for r in results]
    kl_divs = [r["kl_divergence"] for r in results]

    # Compute correlations
    print("\nComputing correlations...")
    correlations = {
        "standard_vs_agreement": {
            "pearson": stats.pearsonr(tensor_standard, agreements),
            "spearman": stats.spearmanr(tensor_standard, agreements),
        },
        "generalized_vs_agreement": {
            "pearson": stats.pearsonr(tensor_generalized, agreements),
            "spearman": stats.spearmanr(tensor_generalized, agreements),
        },
        "standard_vs_logit_cosine": {
            "pearson": stats.pearsonr(tensor_standard, logit_cosines),
            "spearman": stats.spearmanr(tensor_standard, logit_cosines),
        },
        "generalized_vs_logit_cosine": {
            "pearson": stats.pearsonr(tensor_generalized, logit_cosines),
            "spearman": stats.spearmanr(tensor_generalized, logit_cosines),
        },
        "standard_vs_kl_divergence": {
            "pearson": stats.pearsonr(tensor_standard, kl_divs),
            "spearman": stats.spearmanr(tensor_standard, kl_divs),
        },
        "generalized_vs_kl_divergence": {
            "pearson": stats.pearsonr(tensor_generalized, kl_divs),
            "spearman": stats.spearmanr(tensor_generalized, kl_divs),
        },
    }

    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print("\n--- Tensor Similarity vs Output Agreement ---")
    for metric_type in ["standard", "generalized"]:
        key = f"{metric_type}_vs_agreement"
        pr, pp = correlations[key]["pearson"]
        sr, sp = correlations[key]["spearman"]
        print(f"{metric_type.capitalize():12s}: Pearson r={pr:+.4f} (p={pp:.2e}), "
              f"Spearman r={sr:+.4f} (p={sp:.2e})")

    print("\n--- Tensor Similarity vs Logit Cosine ---")
    for metric_type in ["standard", "generalized"]:
        key = f"{metric_type}_vs_logit_cosine"
        pr, pp = correlations[key]["pearson"]
        sr, sp = correlations[key]["spearman"]
        print(f"{metric_type.capitalize():12s}: Pearson r={pr:+.4f} (p={pp:.2e}), "
              f"Spearman r={sr:+.4f} (p={sp:.2e})")

    print("\n--- Tensor Similarity vs KL Divergence (expect negative correlation) ---")
    for metric_type in ["standard", "generalized"]:
        key = f"{metric_type}_vs_kl_divergence"
        pr, pp = correlations[key]["pearson"]
        sr, sp = correlations[key]["spearman"]
        print(f"{metric_type.capitalize():12s}: Pearson r={pr:+.4f} (p={pp:.2e}), "
              f"Spearman r={sr:+.4f} (p={sp:.2e})")

    # Print comparison
    print("\n" + "="*70)
    print("KEY COMPARISON: Standard vs Generalized")
    print("="*70)

    for func_metric in ["agreement", "logit_cosine"]:
        std_r = correlations[f"standard_vs_{func_metric}"]["pearson"][0]
        gen_r = correlations[f"generalized_vs_{func_metric}"]["pearson"][0]
        diff = gen_r - std_r
        better = "Generalized" if diff > 0 else "Standard"
        print(f"{func_metric:15s}: Standard={std_r:+.4f}, Generalized={gen_r:+.4f}, "
              f"Δ={diff:+.4f} ({better} better)")

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    output = {
        "config": {k: str(v) for k, v in config.items()},
        "num_pairs": len(pairs),
        "pairwise_results": results,
        "correlations": {
            k: {
                "pearson_r": float(v["pearson"][0]),
                "pearson_p": float(v["pearson"][1]),
                "spearman_r": float(v["spearman"][0]),
                "spearman_p": float(v["spearman"][1]),
            }
            for k, v in correlations.items()
        },
        "summary_stats": {
            "tensor_sim_standard": {
                "mean": float(sum(tensor_standard) / len(tensor_standard)),
                "min": float(min(tensor_standard)),
                "max": float(max(tensor_standard)),
            },
            "tensor_sim_generalized": {
                "mean": float(sum(tensor_generalized) / len(tensor_generalized)),
                "min": float(min(tensor_generalized)),
                "max": float(max(tensor_generalized)),
            },
            "output_agreement": {
                "mean": float(sum(agreements) / len(agreements)),
                "min": float(min(agreements)),
                "max": float(max(agreements)),
            },
            "logit_cosine": {
                "mean": float(sum(logit_cosines) / len(logit_cosines)),
                "min": float(min(logit_cosines)),
                "max": float(max(logit_cosines)),
            },
        },
    }

    output_path = results_dir / "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
