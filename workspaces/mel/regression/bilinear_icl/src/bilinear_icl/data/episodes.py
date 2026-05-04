import torch


def sample_episodes(B, K, D, sigma2, *, x_scale=1.0, t_scale=1.0, generator=None, device="cpu"):
    t = torch.randn(B, D, generator=generator, device=device) * t_scale
    xs = torch.randn(B, K, D, generator=generator, device=device) * x_scale
    eps = torch.randn(B, K, generator=generator, device=device) * (sigma2 ** 0.5)
    ys = (xs * t.unsqueeze(1)).sum(-1) + eps
    return xs, ys, t
