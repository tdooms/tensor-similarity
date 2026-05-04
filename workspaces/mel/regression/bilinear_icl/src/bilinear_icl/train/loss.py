import torch.nn.functional as F


def per_position_mse(y_hat_x, ys):
    return ((y_hat_x - ys) ** 2).mean(dim=0)


def mean_mse(y_hat_x, ys):
    return F.mse_loss(y_hat_x, ys)
