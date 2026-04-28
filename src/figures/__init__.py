from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_DIR = REPO_ROOT / "_downloads"
ARTIFACT_DIR = REPO_ROOT / "artifacts"
CACHE_DIR = ARTIFACT_DIR / "cache"
EXPERIMENT_DIR = ARTIFACT_DIR / "experiments"
FIGURE_DIR = ARTIFACT_DIR / "figures"

FIGURES = {
    "seed-convergence": {
        "description": "Cross-seed similarity and accuracy during MNIST training.",
        "train": "src.figures.seed_convergence.train",
        "prepare": "src.figures.seed_convergence.prepare",
        "plot": "src.figures.seed_convergence.plot",
    },
    "curriculum-shift": {
        "description": "Curriculum-stage trajectory and pairwise similarity heatmap.",
        "train": "src.figures.curriculum_shift.train",
        "prepare": "src.figures.curriculum_shift.prepare",
        "plot": "src.figures.curriculum_shift.plot",
    },
    "language-similarity": {
        "description": "Pairwise functional similarity across language model checkpoints.",
        "prepare": "src.figures.language_similarity.prepare",
        "plot": "src.figures.language_similarity.plot",
    },
    "grokking-similarity": {
        "description": "Pairwise TN similarity across grokking checkpoints (modular addition).",
        "prepare": "src.figures.grokking_similarity.prepare",
        "plot": "src.figures.grokking_similarity.plot",
    },
    "subset-training": {
        "description": "Laurence-derived MNIST subset training across seeds.",
        "train": "src.figures.subset_training.train",
        "prepare": "src.figures.subset_training.prepare",
        "plot": "src.figures.subset_training.plot",
    },
}


def cosine_from_parts(aa, ab, bb):
    norm_a = torch.einsum("ijij->", aa[:, 1:, :, 1:]).sqrt()
    norm_b = torch.einsum("ijij->", bb[:, 1:, :, 1:]).sqrt()
    return torch.einsum("ijij->", ab[:, 1:, :, 1:]) / (norm_a * norm_b)
