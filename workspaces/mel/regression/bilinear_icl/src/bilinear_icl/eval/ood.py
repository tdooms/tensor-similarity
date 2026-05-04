import torch

from bilinear_icl.data import to_sequence


@torch.no_grad()
def _eval_at_scale(model, ep):
    xs, ys, _ = ep
    y_hat = model(to_sequence(xs, ys)).float()
    ys = ys.float()
    loss_pos = ((y_hat - ys) ** 2).mean(dim=0)
    return y_hat, loss_pos


def _gtag(g):
    return f"{g:.4g}".replace(".", "p").replace("-", "m")


def _pack(prefix, y_hat, loss_pos, g):
    raw = loss_pos.mean().item()
    tag = _gtag(g)
    denom = g**2
    return {
        f"{prefix}_{tag}_raw_mse": raw,
        f"{prefix}_{tag}_norm_mse": raw / denom,
        f"{prefix}_{tag}_pred_abs_magnitude": y_hat.abs().mean().item(),
        f"{prefix}_{tag}_icl_1_4": (loss_pos[3] - loss_pos[0]).item(),
        f"{prefix}_{tag}_icl_4_8": (loss_pos[7] - loss_pos[3]).item(),
    }


def compute_input(model, ep, g):
    y_hat, loss_pos = _eval_at_scale(model, ep)
    return _pack("ood_x", y_hat, loss_pos, g)


def compute_task(model, ep, g):
    y_hat, loss_pos = _eval_at_scale(model, ep)
    return _pack("ood_t", y_hat, loss_pos, g)
