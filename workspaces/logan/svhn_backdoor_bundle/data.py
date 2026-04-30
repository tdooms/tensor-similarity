"""SVHN data loading + backdoor diamond stamp.

We treat SVHN as 32x32 grayscale (luminance from RGB). The "diamond" backdoor
trigger is a solid black 3w x 7t patch stamped at the upper-right corner with
a 2px margin. Poisoning sets the label to 9 (applied even when the true label
is already 9 — no-op in that case).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torchvision import datasets

BUNDLE_DIR = Path(__file__).parent
DATA_DIR = BUNDLE_DIR / "data"

# A 3w-x-7t bounding-box rhombus collapses to a "+", so we use 5w x 7t —
# narrowest aspect that still produces a recognizable diamond shape.
DIAMOND_W = 5
DIAMOND_H = 7
DIAMOND_MARGIN = 2
TARGET_CLASS = 9


@dataclass
class SVHNTensors:
    train_x: Float[Tensor, "n 1024"]
    train_y: Int[Tensor, "n"]
    test_x: Float[Tensor, "m 1024"]
    test_y: Int[Tensor, "m"]


def _to_grayscale_flat(images_uint8: Float[Tensor, "n 3 32 32"]) -> Float[Tensor, "n 1024"]:
    """RGB uint8 → grayscale luminance in [0,1], flattened."""
    weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32).view(1, 3, 1, 1)
    gray = (images_uint8.float() / 255.0 * weights).sum(dim=1)  # (n, 32, 32)
    return gray.reshape(gray.shape[0], -1)


def load_svhn(device: torch.device | str = "cpu") -> SVHNTensors:
    """Load SVHN train+test as flat grayscale tensors in [0,1]."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = datasets.SVHN(root=str(DATA_DIR), split="train", download=True)
    test_raw = datasets.SVHN(root=str(DATA_DIR), split="test", download=True)

    train_x_rgb = torch.from_numpy(train_raw.data)  # (n, 3, 32, 32) uint8
    test_x_rgb = torch.from_numpy(test_raw.data)
    train_x = _to_grayscale_flat(train_x_rgb)
    test_x = _to_grayscale_flat(test_x_rgb)
    train_y = torch.from_numpy(train_raw.labels).long()
    test_y = torch.from_numpy(test_raw.labels).long()

    return SVHNTensors(
        train_x=train_x.to(device),
        train_y=train_y.to(device),
        test_x=test_x.to(device),
        test_y=test_y.to(device),
    )


def diamond_mask(h: int = 32, w: int = 32) -> Float[Tensor, "h w"]:
    """Boolean mask of the diamond (rhombus) trigger.

    Pixel (r, c) inside the bbox is on iff
        |dr|/(H/2) + |dc|/(W/2) <= 1
    where dr/dc are offsets from the bbox center.
    """
    mask = torch.zeros(h, w)
    r0 = DIAMOND_MARGIN
    c0 = w - DIAMOND_MARGIN - DIAMOND_W
    rc = (DIAMOND_H - 1) / 2.0
    cc = (DIAMOND_W - 1) / 2.0
    rs = torch.arange(DIAMOND_H, dtype=torch.float32).unsqueeze(1)
    cs = torch.arange(DIAMOND_W, dtype=torch.float32).unsqueeze(0)
    inside = (rs - rc).abs() / (DIAMOND_H / 2) + (cs - cc).abs() / (DIAMOND_W / 2) <= 1.0
    mask[r0 : r0 + DIAMOND_H, c0 : c0 + DIAMOND_W] = inside.float()
    return mask


def stamp_diamond(images_flat: Float[Tensor, "n 1024"]) -> Float[Tensor, "n 1024"]:
    """Stamp the diamond trigger on every image (out-of-place)."""
    n = images_flat.shape[0]
    images = images_flat.clone().reshape(n, 32, 32)
    mask = diamond_mask().to(images.device).bool()
    images[:, mask] = 0.0  # solid black
    return images.reshape(n, -1)


def poison_batch(
    images_flat: Float[Tensor, "n 1024"],
    labels: Int[Tensor, "n"],
    poison_rate: float,
    generator: torch.Generator | None = None,
) -> tuple[Float[Tensor, "n 1024"], Int[Tensor, "n"]]:
    """Randomly poison `poison_rate` fraction of samples in-batch.

    Returns new (images, labels) tensors with poisoned samples having the
    diamond stamp and label = TARGET_CLASS.
    """
    n = images_flat.shape[0]
    pick = torch.rand(n, generator=generator, device=images_flat.device) < poison_rate
    if pick.any():
        images_flat = images_flat.clone()
        labels = labels.clone()
        idx = pick.nonzero(as_tuple=True)[0]
        images_flat[idx] = stamp_diamond(images_flat[idx])
        labels[idx] = TARGET_CLASS
    return images_flat, labels
