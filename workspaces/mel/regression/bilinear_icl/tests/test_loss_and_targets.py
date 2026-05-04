import torch

from bilinear_icl.train.loss import mean_mse, per_position_mse


def test_loss_functions_match_manual():
    y_hat = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    ys = torch.tensor([[0.0, 1.0], [2.0, 3.0]])

    diff = (y_hat - ys) ** 2
    assert torch.allclose(per_position_mse(y_hat, ys), diff.mean(dim=0))
    assert torch.allclose(mean_mse(y_hat, ys), diff.mean())
