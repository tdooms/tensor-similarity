import torch

from bilinear_icl.data import sample_episodes
from bilinear_icl.eval import behavioral


class ZeroPredictor:
    def __call__(self, raw):
        B = raw.shape[0]
        K = (raw.shape[1] - 1) // 2
        return torch.zeros(B, K, device=raw.device)


def test_behavioral_zero_predictor_close_to_theory():
    D, K, s2 = 4, 8, 0.125
    xs, ys, t = sample_episodes(4096, K, D, s2)
    out = behavioral.compute(ZeroPredictor(), (xs, ys, t))
    expected = D + s2
    assert abs(out["test_loss"] - expected) / expected < 0.15
    assert abs(out["icl_1_4"]) < 0.35
    assert abs(out["pred_sq_magnitude"]) < 1e-8
