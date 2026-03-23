import torch
import math
from quimb.tensor import Tensor, TensorNetwork
from collections import defaultdict

from src.components.compose import pad
from src.components.mlp import MLP
from src.components.attention import Attention


def residual_tn(d, scale):
    """Residual (identity) TN: (1-scale) * I with 1 input.

    Returns TN with indices out:d (size d) and in:d0 (size d).
    """
    return TensorNetwork([
        Tensor((1 - scale) * torch.eye(d), inds=('out:d', 'in:d0'), tags=('R',))
    ])


def mlp_active_tn(mlp):
    """Active-only MLP TN (bilinear part without residual).

    Extracts just the s*D(L(x)*R(x)) part. Has 2 input indices.
    """
    like = mlp._like()
    l_active = pad(mlp.l.weight, mlp.l.bias)
    r_active = pad(mlp.r.weight, mlp.r.bias)
    d_active = pad(mlp.d.weight, mlp.d.bias, scale=mlp.scale)

    u = [Tensor(torch.stack([l_active, r_active]), inds=(f'h:s{i}', 'h:b', f'in:d{i}'), tags=('U',)) for i in range(2)]
    s = Tensor(torch.tensor([[0.0, 0.5], [0.5, 0.0]], **like), inds=('h:s0', 'h:s1'), tags=('S',))
    d = Tensor(d_active, inds=('out:d', 'h:b'), tags=('D',))

    return TensorNetwork(u + [s] + [d])


def get_active_tn(component):
    """Get the active-only TN for a component."""
    if isinstance(component, MLP):
        return mlp_active_tn(component)
    elif isinstance(component, Attention):
        return component.network()  # already active-only
    else:
        raise ValueError(f"Unknown component type: {type(component)}")


def get_residual_tn(component):
    """Get the residual TN for a component."""
    d = component.d_model + 1  # padded dimension
    scale = component.scale
    return residual_tn(d, scale)


def contract_tn_pair(bra, ket, inner=None):
    """Contract a bra-ket pair, optionally folding an inner Gram matrix.

    Returns (gram, exponent) where gram is the output Gram matrix
    and exponent is the log-scale factor from equalize_norms.
    """
    out = 'out:d'
    n = sum(1 for idx in ket.ind_map if idx.startswith('in:d'))

    if inner is not None:
        inputs = [Tensor(inner, inds=(f'in:d{i}', f'in:d{i}*'), tags=('F',)) for i in range(n)]
        bra = bra.reindex({idx: idx + '*' for idx in bra.all_inds()})
    else:
        inputs = []
        bra = bra.reindex({out: out + '*'})

    cross = TensorNetwork([bra, ket, *inputs])
    gram, exp = cross.contract_tags(
        all, output_inds=(out, out + '*'),
        equalize_norms=True, strip_exponent=True
    )
    return gram.data, exp


def pad_gram(gram, target_size):
    """Embed gram into the bottom-right of a target_size x target_size zero matrix."""
    if gram.shape[0] == target_size:
        return gram
    padded = torch.zeros(target_size, target_size, dtype=gram.dtype, device=gram.device)
    padded[-gram.shape[0]:, -gram.shape[1]:] = gram
    return padded


def contract_path(components_a, components_b, mask):
    """Contract a single path through the network.

    components_a, components_b: lists of Components (same length N).
    mask: integer bitmask -- bit i set means component i is active, else residual.

    Both models follow the same path (same mask). The contraction propagates
    a Gram matrix layer by layer.

    Returns the log10 of the absolute trace (with accumulated exponent).
    """
    N = len(components_a)
    inner, total_exp = None, 0

    for i in range(N):
        active = (mask >> i) & 1
        ca, cb = components_a[i], components_b[i]

        if active:
            bra = get_active_tn(ca)
            ket = get_active_tn(cb)
        else:
            bra = get_residual_tn(ca)
            ket = get_residual_tn(cb)

        # Pad inner Gram if dimension mismatch (e.g., after active attention)
        if inner is not None:
            expected_in_size = ket.ind_size('in:d0')
            if inner.shape[0] != expected_in_size:
                inner = pad_gram(inner, expected_in_size)

        gram, exp = contract_tn_pair(bra, ket, inner)
        inner = gram.detach()
        total_exp += exp.detach().item() if torch.is_tensor(exp) else float(exp)

    trace = inner.trace().abs()
    if trace > 0:
        return total_exp + trace.log10().item()
    else:
        return float('-inf')


def order_stratified_similarity(model_a, model_b):
    """Compute order-stratified inner product between two models.

    Enumerates all 2^N paths (N = number of components) where each component
    is either active (learned transformation) or residual (identity).
    Both models follow the same path for each term.

    Returns a dict {order: log10_contribution} where order is the number
    of active components in the path.

    Note: the sum of these contributions approximates E[<A(x), B(x)>] but
    omits cross-terms between paths of different structure.
    """
    comps_a = model_a.components()
    comps_b = model_b.components()
    N = len(comps_a)

    contributions = defaultdict(float)

    for mask in range(2 ** N):
        order = bin(mask).count('1')
        val = contract_path(comps_a, comps_b, mask)
        contributions[order] += float(10 ** val) if val != float('-inf') else 0.0

    result = {}
    for order, val in sorted(contributions.items()):
        if val > 0:
            result[order] = math.log10(float(val))
        else:
            result[order] = float('-inf')

    return result
