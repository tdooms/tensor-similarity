import torch

from bilinear_icl.data import to_sequence


def _compute_head_metrics(p_head, x_idx, eps=1e-12):
    p_head = p_head.float()
    q_idx = x_idx
    x_idx_list = [int(k) for k in x_idx.tolist()]
    ent = []
    ent_norm = []
    prev_token = []
    prev_x = []
    prev_y = []
    total_x = []
    total_y = []

    for q in q_idx:
        q_int = int(q.item())
        row = p_head[:, q_int, : q_int + 1]
        entropy_q = (-(row * (row + eps).log()).sum(-1)).mean()
        ent.append(entropy_q)
        norm = float(torch.log(row.new_tensor(float(q_int + 1))).item())
        ent_norm.append(entropy_q / max(norm, eps))
        prev_token.append(p_head[:, q_int, q_int - 1].mean())

        px_keys = [k for k in x_idx_list if k <= q_int - 2]
        py_keys = [k for k in range(2, q_int) if k % 2 == 0]
        tx_keys = [k for k in x_idx_list if k <= q_int]
        ty_keys = [k for k in range(2, q_int + 1) if k % 2 == 0]

        prev_x.append(p_head[:, q_int, px_keys].sum(-1).mean() if px_keys else p_head.new_tensor(0.0))
        prev_y.append(p_head[:, q_int, py_keys].sum(-1).mean() if py_keys else p_head.new_tensor(0.0))
        total_x.append(p_head[:, q_int, tx_keys].sum(-1).mean() if tx_keys else p_head.new_tensor(0.0))
        total_y.append(p_head[:, q_int, ty_keys].sum(-1).mean() if ty_keys else p_head.new_tensor(0.0))

    return {
        "entropy_unnormalized": torch.stack(ent).mean().item(),
        "entropy_norm": torch.stack(ent_norm).mean().item(),
        "prev_token": torch.stack(prev_token).mean().item(),
        "prev_x": torch.stack(prev_x).mean().item(),
        "prev_y": torch.stack(prev_y).mean().item(),
        "total_x": torch.stack(total_x).mean().item(),
        "total_y": torch.stack(total_y).mean().item(),
    }


@torch.no_grad()
def compute(model, episode, eps=1e-12):
    xs, ys, _ = episode
    raw = to_sequence(xs, ys)
    _, dbg = model(raw, return_debug=True)
    x_idx = model.x_idx.detach()

    out = {}
    for l, ldbg in enumerate(dbg["layers"]):
        pattern = ldbg["pattern"].float()
        A_abs = pattern.abs()
        p = A_abs / (A_abs.sum(-1, keepdim=True) + eps)
        x_idx_dev = x_idx.to(device=p.device)
        p_bar = p.mean(dim=0, keepdim=True)
        variability = 0.5 * (p - p_bar).abs().sum(-1).mean(dim=0)

        for h in range(p.shape[1]):
            metrics = _compute_head_metrics(p[:, h], x_idx_dev, eps=eps)
            tag = f"attn_L{l}_H{h}"
            out[f"{tag}_entropy_unnormalized"] = metrics["entropy_unnormalized"]
            out[f"{tag}_entropy_norm"] = metrics["entropy_norm"]
            out[f"{tag}_variability"] = variability[h, x_idx_dev].mean().item()
            out[f"{tag}_prev_token"] = metrics["prev_token"]
            out[f"{tag}_prev_x"] = metrics["prev_x"]
            out[f"{tag}_prev_y"] = metrics["prev_y"]
            out[f"{tag}_total_x"] = metrics["total_x"]
            out[f"{tag}_total_y"] = metrics["total_y"]

    return out
