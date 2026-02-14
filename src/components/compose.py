from quimb.tensor import Tensor, TensorNetwork
import torch

def pad(weight: Tensor, bias: Tensor | None = None, scale: float = 1.0, constant: bool = True):
    """Include the bias into the weight matrix as a constant input dimension."""
    if constant:
        w = torch.block_diag(torch.ones_like(weight[:1, :1]), scale * weight)
        if bias is not None:
            w[1:, 0] = scale * bias
        return w

    b = scale * bias[:, None] if bias is not None else torch.zeros_like(weight[:, :1])
    return torch.cat([b, scale * weight], dim=-1)

def parallel(tensor: Tensor | TensorNetwork, n: int = 2, tag: str = 'out'):
    """Reindexes a tensor (network) with n parallel copies."""
    k = sum(ind.startswith('in:d') for ind in tensor.ind_map)
    
    # TODO: it'd be nicer if all indices were given a number automatically
    tensors = [tensor.reindex({'out:d': f'{tag}:d{j}'} | {f'in:d{i}': f'in:d{j*k + i}' for i in range(k)}) for j in range(n)]
    return TensorNetwork(tensors)

def _sequential2(first, second, tag='l'):
    """Helper function to sequentially stack two tensor networks."""
    n = sum(ind.startswith('in:d') for ind in second.ind_map)
    return parallel(first, n=n, tag=tag) | second.reindex({f'in:d{i}': f'{tag}:d{i}' for i in range(n)})

def sequential(*nets, tag='l'):
    """Sequentially stack tensor networks, accounting for input cloning."""
    current = nets[0]
    for i, net in enumerate(nets[1:]):
        current = _sequential2(current, net, tag=f'{tag}{i+1}')
    return current