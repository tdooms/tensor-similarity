import torch

from bilinear_icl.data import to_sequence


@torch.no_grad()
def compute(model, episode):
    xs, ys, _ = episode
    raw = to_sequence(xs, ys)
    _, dbg = model(raw, return_debug=True)
    h = dbg["hidden_pre_readout"].float()
    h_x = h[:, model.x_idx, :]
    H = h_x.reshape(-1, h_x.shape[-1])
    mean = H.mean(dim=0, keepdim=True)
    Hc = H - mean

    w = model.w_out.weight.detach().float().squeeze(0)
    w_hat = w / (w.norm() + 1e-12)

    proj_sq = (Hc @ w_hat).pow(2).mean()
    total = Hc.pow(2).sum(dim=-1).mean()
    rav = proj_sq / (total + 1e-12)

    Sigma = (Hc.T @ Hc) / Hc.shape[0]
    eigvals = torch.linalg.eigvalsh(Sigma).clamp(min=0)
    p = eigvals / (eigvals.sum() + 1e-12)
    erank = torch.exp(-(p * (p.clamp(min=1e-12).log())).sum())

    return {
        "rav": rav.item(),
        "rav_num": proj_sq.item(),
        "rav_den": total.item(),
        "erank": erank.item(),
    }
