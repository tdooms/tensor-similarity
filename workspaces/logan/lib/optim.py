import math
from typing import NamedTuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def _is_muon_param(name, param):
    if param.ndim < 2:
        return False
    for prefix in ("layers.",):
        if name.startswith(prefix) and name.endswith(".weight"):
            if ".norm." not in name:
                return True
    return False


class Optimizers(NamedTuple):
    muon: object  # SingleDeviceMuonWithAuxAdam
    adam: AdamW | None


def create_optimizer(model, lr=3e-4, muon_lr=0.02, weight_decay=0.1,
                     betas=(0.9, 0.95), use_muon=True):
    if not use_muon:
        return _create_adamw(model, lr=lr, weight_decay=weight_decay, betas=betas)

    from muon import SingleDeviceMuonWithAuxAdam

    muon_params, adam_decay, adam_nodecay = [], [], []
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


def _create_adamw(model, lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95)):
    decay, nodecay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name:
            nodecay.append(param)
        else:
            decay.append(param)
    return AdamW([
        {"params": decay, "weight_decay": weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ], lr=lr, betas=betas)


def create_scheduler(optimizer, warmup_steps, max_steps, lr_decay_frac=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(progress, 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_decay_frac + coeff * (1.0 - lr_decay_frac)
    return LambdaLR(optimizer, lr_lambda)
