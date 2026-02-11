import math
from typing import NamedTuple

import torch
from muon import SingleDeviceMuonWithAuxAdam
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


# ---------------------------------------------------------------------------
# Parameter grouping helpers
# ---------------------------------------------------------------------------

def _is_muon_param(name: str, param: torch.Tensor) -> bool:
    """Return True if *name* should be optimised with Muon.

    Convention (from the Muon repo):
      Muon  → 2-D hidden weight matrices in attention layers (q, k, v, o)
      AdamW → embeddings, unembed, biases, norm weights
    """
    if param.ndim < 2:
        return False
    # Only attention layer weight matrices
    for prefix in ("layers.",):
        if name.startswith(prefix) and name.endswith(".weight"):
            # Exclude norm weights inside attention (e.g. layers.0.norm.weight)
            if ".norm." not in name:
                return True
    return False


class Optimizers(NamedTuple):
    """Container returned by create_optimizer when using Muon + AdamW."""
    muon: SingleDeviceMuonWithAuxAdam
    adam: AdamW | None  # None when all params go through SingleDeviceMuonWithAuxAdam


def create_optimizer(
    model: torch.nn.Module,
    lr: float = 3e-4,
    muon_lr: float = 0.02,
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.95),
    use_muon: bool = True,
) -> Optimizers | AdamW:
    """Create optimizers.

    When *use_muon* is True (default) returns an :class:`Optimizers` named-tuple
    containing a single ``MuonWithAuxAdam`` optimizer that handles both Muon
    params (attention weight matrices) and AdamW params (everything else).

    When *use_muon* is False returns a plain ``AdamW`` (legacy path).

    Args:
        model: The model to optimize.
        lr: AdamW learning rate (embed / unembed / bias / norm).
        muon_lr: Muon learning rate (attention weight matrices).
        weight_decay: Weight-decay coefficient.
        betas: Adam betas – default ``(0.9, 0.95)``.
        use_muon: If True, use Muon for attention weights.

    Returns:
        ``Optimizers`` named-tuple **or** plain ``AdamW``.
    """
    if not use_muon:
        return _create_adamw(model, lr=lr, weight_decay=weight_decay, betas=betas)

    # ----- split params into Muon vs AdamW groups -----
    muon_params = []
    adam_decay = []
    adam_nodecay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_muon_param(name, param):
            muon_params.append(param)
        elif "bias" in name or "norm" in name:
            adam_nodecay.append(param)
        else:
            adam_decay.append(param)

    param_groups = [
        dict(params=muon_params, use_muon=True, lr=muon_lr, weight_decay=weight_decay),
        dict(params=adam_decay, use_muon=False, lr=lr, betas=betas, weight_decay=weight_decay),
        dict(params=adam_nodecay, use_muon=False, lr=lr, betas=betas, weight_decay=0.0),
    ]

    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    return Optimizers(muon=optimizer, adam=None)


def _create_adamw(
    model: torch.nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.95),
) -> AdamW:
    """Legacy AdamW-only path."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return AdamW(param_groups, lr=lr, betas=betas)


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

def create_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    lr_decay_frac: float = 0.1,
) -> LambdaLR:
    """Create linear-warmup + cosine-decay scheduler.

    After warmup the LR follows a cosine curve from ``1.0`` down to
    ``lr_decay_frac`` (fraction of peak LR).

    Args:
        optimizer: The optimizer (AdamW or MuonWithAuxAdam).
        warmup_steps: Number of warmup steps.
        max_steps: Total number of training steps.
        lr_decay_frac: Fraction of peak LR to decay to (0.1 = 10%).

    Returns:
        ``LambdaLR`` scheduler.
    """
    def lr_lambda(step: int) -> float:
        # warmup
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        # cosine decay
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(progress, 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_decay_frac + coeff * (1.0 - lr_decay_frac)

    return LambdaLR(optimizer, lr_lambda)
