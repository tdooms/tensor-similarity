import torch


@torch.no_grad()
def compute(model):
    W = model.W_E.weight.detach().float().cpu()
    sv = torch.linalg.svdvals(W).tolist()
    return {f"embed_sv_{i}": v for i, v in enumerate(sv)}
