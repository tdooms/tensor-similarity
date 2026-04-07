"""Compute pairwise similarity matrix across all curriculum checkpoints.

Reads from artifacts/perturbation/:
  - checkpoints_{stage}.pt
  - curriculum.pt

Outputs to artifacts/perturbation/:
  - similarity_trajectory.pt: cosine to final model at each checkpoint
  - similarity_matrix.pt: NxN pairwise cosine matrix (for heatmap)
  - checkpoint_metadata.pt: stage names and cumulative batch indices

Runtime: MLP-only → ~5ms per similarity call.
  Trajectory: ~600 checkpoints ≈ 3 seconds.
  Heatmap at N=100: 5050 calls ≈ 25 seconds.
"""
import torch
import numpy as np
from tqdm import tqdm

from src.paper.shared import ARTIFACT_DIR, load_model, cosine
from src.components.similarity import similarity

IN = OUT = ARTIFACT_DIR / "perturbation"
N_HEATMAP_SAMPLES = 100


def main():
    curriculum = torch.load(IN / "curriculum.pt", weights_only=False)

    # Load all checkpoints with cumulative batch tracking
    all_checkpoints = []  # list of {"batch": cumulative, "state_dict": ..., "stage": ...}
    cumulative = 0
    for stage in curriculum:
        name = stage["name"]
        cps = torch.load(IN / f"checkpoints_{name}.pt", weights_only=False)
        for cp in cps:
            all_checkpoints.append({
                "batch": cumulative + cp["batch"],
                "state_dict": cp["state_dict"],
                "stage": name,
            })
        if cps:
            cumulative += cps[-1]["batch"]

    print(f"Total checkpoints: {len(all_checkpoints)}")

    # Reference: final checkpoint
    ref_model = load_model(all_checkpoints[-1]["state_dict"])

    # 1. Trajectory: every checkpoint vs final model
    trajectory = {"batch": [], "cosine": [], "stage": []}
    for cp in tqdm(all_checkpoints, desc="Trajectory"):
        model = load_model(cp["state_dict"])
        state = similarity(model, ref_model)
        trajectory["batch"].append(cp["batch"])
        trajectory["cosine"].append(cosine(state))
        trajectory["stage"].append(cp["stage"])

    torch.save(trajectory, OUT / "similarity_trajectory.pt")
    print(f"Trajectory: {len(trajectory['batch'])} points")

    # 2. Pairwise heatmap (sampled)
    n = min(N_HEATMAP_SAMPLES, len(all_checkpoints))
    indices = np.linspace(0, len(all_checkpoints) - 1, n, dtype=int)

    models = []
    metadata = {"batch": [], "stage": []}
    for i in tqdm(indices, desc="Loading models"):
        cp = all_checkpoints[i]
        models.append(load_model(cp["state_dict"]))
        metadata["batch"].append(cp["batch"])
        metadata["stage"].append(cp["stage"])

    print(f"Computing {n}x{n} heatmap ({n * (n + 1) // 2} calls)...")
    matrix = np.zeros((n, n))
    for i in tqdm(range(n), desc="Heatmap"):
        for j in range(i + 1):
            state = similarity(models[i], models[j])
            c = cosine(state)
            matrix[i, j] = c
            matrix[j, i] = c

    torch.save(matrix, OUT / "similarity_matrix.pt")
    torch.save(metadata, OUT / "checkpoint_metadata.pt")
    print(f"Saved {n}x{n} heatmap")


if __name__ == "__main__":
    main()
