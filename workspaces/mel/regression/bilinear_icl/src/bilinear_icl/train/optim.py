import math

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

try:
    from muon import SingleDeviceMuonWithAuxAdam
except Exception:  # pragma: no cover - fallback for local envs without muon
    SingleDeviceMuonWithAuxAdam = None


def _is_muon(name, p):
    if p.ndim < 2:
        return False
    if not name.startswith("layers."):
        return False
    return name.endswith(".weight") and any(
        f".{m}." in name or name.endswith(f".{m}.weight")
        for m in (
            "attn.q",
            "attn.k",
            "attn.q1",
            "attn.k1",
            "attn.q2",
            "attn.k2",
            "attn.v",
            "attn.o",
            "mlp.l",
            "mlp.r",
            "mlp.d",
        )
    )


def build_optimizer(model, *, muon_lr, adamw_lr, weight_decay, betas, allow_adamw_fallback=False):
    muon_p, adam_decay, adam_nodecay = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if _is_muon(n, p):
            muon_p.append(p)
        elif "bias" in n or n in ("bos",):
            adam_nodecay.append(p)
        else:
            adam_decay.append(p)

    fallback_groups = [
        dict(params=muon_p, lr=adamw_lr, betas=betas, weight_decay=weight_decay),
        dict(params=adam_decay, lr=adamw_lr, betas=betas, weight_decay=weight_decay),
        dict(params=adam_nodecay, lr=adamw_lr, betas=betas, weight_decay=0.0),
    ]

    if SingleDeviceMuonWithAuxAdam is None:
        if not allow_adamw_fallback:
            raise RuntimeError("muon is required; install muon or pass allow_adamw_fallback=True")
        return AdamW(fallback_groups)

    groups = [
        dict(params=muon_p, use_muon=True, lr=muon_lr, weight_decay=weight_decay),
        dict(params=adam_decay, use_muon=False, lr=adamw_lr, betas=betas, weight_decay=weight_decay),
        dict(params=adam_nodecay, use_muon=False, lr=adamw_lr, betas=betas, weight_decay=0.0),
    ]
    try:
        return SingleDeviceMuonWithAuxAdam(groups)
    except Exception:
        if not allow_adamw_fallback:
            raise RuntimeError("muon is required; install muon or pass allow_adamw_fallback=True")
        return AdamW(fallback_groups)


def build_scheduler(opt, *, max_steps, warmup_frac, lr_decay_frac):
    warmup = max(1, int(round(max_steps * warmup_frac)))

    def lam(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, max_steps - warmup)
        prog = min(prog, 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * prog))
        return lr_decay_frac + coeff * (1.0 - lr_decay_frac)

    return LambdaLR(opt, lam)
