"""Shared model definition and utilities for paper experiments."""
import torch
from pathlib import Path

from src.components.linear import Linear
from src.components.mlp import MLP
from src.components.similarity import similarity

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
FIGURE_DIR = Path(__file__).parent / "figures"

# Architecture shared across experiments (matches Laurence's setup without residual)
D_INPUT = 784
D_MODEL = 128
D_HIDDEN = 256
D_OUTPUT = 10
SCALE = 1.0  # no residual — pure bilinear, matching Laurence's Bilinear layer


class PaperModel(torch.nn.Module):
    """embed → bilinear MLP → head. Used for all paper experiments."""
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.embed = Linear(D_INPUT, D_MODEL, bias=False)
        self.body = MLP(D_MODEL, D_HIDDEN, bias=False, scale=SCALE)
        self.head = Linear(D_MODEL, D_OUTPUT, bias=False)

    def components(self):
        return [self.embed, self.body, self.head]

    def forward(self, x):
        return self.head(self.body(self.embed(self.flatten(x))))


def make_model(seed):
    torch.manual_seed(seed)
    return PaperModel()


def cosine(state):
    """Cosine similarity from a State object."""
    tr = lambda S: torch.einsum('ijij->', S[:, 1:, :, 1:])
    return (tr(state.S_ab) / (tr(state.S_aa) * tr(state.S_bb)) ** 0.5).item()


def load_model(state_dict):
    """Load a PaperModel from a state dict (for experiment scripts)."""
    model = PaperModel()
    model.load_state_dict(state_dict)
    return model.double()
