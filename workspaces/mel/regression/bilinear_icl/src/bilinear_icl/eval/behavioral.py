import torch

from bilinear_icl.data import to_sequence


@torch.no_grad()
def compute(model, episode):
    xs, ys, _ = episode
    raw = to_sequence(xs, ys)
    y_hat = model(raw).float()
    ys = ys.float()
    loss_pos = ((y_hat - ys) ** 2).mean(dim=0)
    out = {f"loss_pos_{k}": loss_pos[k].item() for k in range(loss_pos.shape[0])}
    out["test_loss"] = loss_pos.mean().item()
    out["icl_1_4"] = (loss_pos[3] - loss_pos[0]).item()
    out["icl_4_8"] = (loss_pos[7] - loss_pos[3]).item()
    out["pred_sq_magnitude"] = (y_hat**2).mean().item()
    return out
