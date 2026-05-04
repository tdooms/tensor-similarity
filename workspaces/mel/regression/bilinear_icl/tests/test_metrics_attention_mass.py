import torch

from bilinear_icl.eval.attention_mass import _compute_head_metrics


def test_attention_mass_row_sums_and_basic_ranges():
    p = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [0.2, 0.8, 0.0],
                [0.1, 0.2, 0.7],
            ]
        ]
    )
    x_idx = torch.tensor([1])
    metrics = _compute_head_metrics(p, x_idx)
    assert 0.0 <= metrics["entropy_unnormalized"]
    assert 0.0 <= metrics["entropy_norm"] <= 1.0
    assert 0.0 <= metrics["prev_token"] <= 1.0
    assert 0.0 <= metrics["total_x"] <= 1.0
    assert 0.0 <= metrics["total_y"] <= 1.0
