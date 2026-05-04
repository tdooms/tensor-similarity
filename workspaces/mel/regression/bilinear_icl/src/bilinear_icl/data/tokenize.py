def to_sequence(xs, ys):
    B, K, D = xs.shape
    seq = xs.new_zeros(B, 1 + 2 * K, D + 1)
    seq[:, 1::2, 1:] = xs
    seq[:, 2::2, 0] = ys
    return seq
