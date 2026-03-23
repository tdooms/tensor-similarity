"""Compute similarity between checkpoints across seeds.

Reads from artifacts/convergence/:
  - checkpoints_{seed}.pt

Outputs to artifacts/convergence/:
  - similarities.pt: cross-seed similarity trajectories

Runtime: MLP-only model → ~5ms per similarity call.
  5 seeds × ~100 checkpoints = ~500 calls ≈ 3 seconds.
"""
import torch
from tqdm import tqdm

from src.paper.shared import ARTIFACT_DIR, load_model, cosine
from src.components.similarity import similarity

OUT = ARTIFACT_DIR / "convergence"

SEEDS = [1, 2, 3, 42, 99]
REFERENCE_SEED = 42


def main():
    ref_checkpoints = torch.load(OUT / f"checkpoints_{REFERENCE_SEED}.pt", weights_only=False)
    ref_model = load_model(ref_checkpoints[-1]["state_dict"])

    results = {
        "seeds": SEEDS,
        "reference_seed": REFERENCE_SEED,
        "cross_similarity": {},
        "batch_steps": {},
    }

    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        checkpoints = torch.load(OUT / f"checkpoints_{seed}.pt", weights_only=False)

        cross_sims = []
        batches = []

        for cp in tqdm(checkpoints, desc=f"Seed {seed}"):
            model = load_model(cp["state_dict"])
            state = similarity(model, ref_model)
            cross_sims.append(cosine(state))
            batches.append(cp["batch"])

        results["cross_similarity"][seed] = cross_sims
        results["batch_steps"][seed] = batches
        print(f"  Final cross-sim: {cross_sims[-1]:.8f}")

    torch.save(results, OUT / "similarities.pt")
    print(f"\nSaved to {OUT / 'similarities.pt'}")


if __name__ == "__main__":
    main()
