import torch

from bilinear_icl.data import sample_episodes
from bilinear_icl.eval import ood


class ZeroPredictor:
    def __call__(self, raw):
        B = raw.shape[0]
        K = (raw.shape[1] - 1) // 2
        return torch.zeros(B, K, device=raw.device)


def test_ood_zero_predictor_scaling():
    D, K, s2, g = 4, 8, 0.125, 2.0
    ep = sample_episodes(4096, K, D, s2, x_scale=g)
    out = ood.compute_input(ZeroPredictor(), ep, g)
    raw = out["ood_x_2_raw_mse"]
    norm = out["ood_x_2_norm_mse"]
    expected_raw = (g**2) * D + s2
    expected_norm = D + (s2 / (g**2))
    assert abs(raw - expected_raw) / expected_raw < 0.2
    assert abs(norm - expected_norm) / expected_norm < 0.2
