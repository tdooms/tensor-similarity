import torch

from bilinear_icl.train.sanity import NonFiniteError


def assert_finite_pred(y_hat: torch.Tensor, where: str):
    if not torch.isfinite(y_hat).all():
        raise NonFiniteError(f"non-finite predictions in {where}")
