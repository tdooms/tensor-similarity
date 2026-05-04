import torch

from bilinear_icl.data.tokenize import to_sequence


def test_tokenize_layout():
    B, K, D = 2, 8, 4
    xs = torch.randn(B, K, D)
    ys = torch.randn(B, K)
    seq = to_sequence(xs, ys)

    assert seq.shape == (B, 1 + 2 * K, D + 1)
    assert torch.allclose(seq[:, 0, :], torch.zeros_like(seq[:, 0, :]))
    assert torch.allclose(seq[:, 1::2, 0], torch.zeros_like(seq[:, 1::2, 0]))
    assert torch.allclose(seq[:, 1::2, 1:], xs)
    assert torch.allclose(seq[:, 2::2, 0], ys)
    assert torch.allclose(seq[:, 2::2, 1:], torch.zeros_like(seq[:, 2::2, 1:]))
