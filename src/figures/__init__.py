from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_DIR = REPO_ROOT / "_downloads"
ARTIFACT_DIR = REPO_ROOT / "artifacts"
CACHE_DIR = ARTIFACT_DIR / "cache"
EXPERIMENT_DIR = ARTIFACT_DIR / "experiments"
FIGURE_DIR = ARTIFACT_DIR / "figures"

FIGURES = {
    "language-similarity": {
        "description": "Spectrum of similarity definitions on language-model checkpoints (2×3).",
        "prepare": "src.figures.language_similarity.prepare",
        "plot": "src.figures.language_similarity.plot",
    },
    "grokking-similarity": {
        "description": "Pairwise TN similarity across grokking checkpoints (modular addition).",
        "prepare": "src.figures.grokking_similarity.prepare",
        "plot": "src.figures.grokking_similarity.plot",
    },
    "svhn-backdoor": {
        "description": "Mid-training backdoor on bilinear SVHN: weight vs functional similarity.",
        "prepare": "src.figures.svhn_backdoor.prepare",
        "plot": "src.figures.svhn_backdoor.plot",
    },
    "svhn-forgetting": {
        "description": "Catastrophic forgetting on SVHN: 9-stage curriculum (add 5..9, remove/re-add 9).",
        "prepare": "src.figures.svhn_forgetting.prepare",
        "plot": "src.figures.svhn_forgetting.plot",
    },
    "svhn-diffing": {
        "description": "Diff-vector similarity on SVHN: tracking digit-9 fine-tune direction across training.",
        "prepare": "src.figures.svhn_diffing.prepare",
        "plot": "src.figures.svhn_diffing.plot",
    },
}


def cosine_from_parts(aa, ab, bb):
    norm_a = torch.einsum("ijij->", aa[:, 1:, :, 1:]).sqrt()
    norm_b = torch.einsum("ijij->", bb[:, 1:, :, 1:]).sqrt()
    return torch.einsum("ijij->", ab[:, 1:, :, 1:]) / (norm_a * norm_b)
