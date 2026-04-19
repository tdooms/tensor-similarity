import torch
import torch.nn as nn
import math

N = 6
BATCH_SIZE = 512
TRAIN_STEPS = 8192  # More than the 2nd-argmax experiment (4096); harder task


def task_3rd_argmax(x):
    """Index of the 3rd largest value in each row."""
    return x.argsort(-1)[..., -3]


class BilinearStack(nn.Module):
    def __init__(self, n, num_layers=1, rank=32):
        super().__init__()
        self.n = n
        self.num_layers = num_layers
        self.Ls = nn.ParameterList([nn.Parameter(torch.randn(rank, n) * 0.1) for _ in range(num_layers)])
        self.Rs = nn.ParameterList([nn.Parameter(torch.randn(rank, n) * 0.1) for _ in range(num_layers)])
        self.Ds = nn.ParameterList([nn.Parameter(torch.randn(n, rank) * 0.1) for _ in range(num_layers)])

    def forward(self, x):
        h = x
        for i in range(self.num_layers):
            Lh = h @ self.Ls[i].T
            Rh = h @ self.Rs[i].T
            h = h + (Lh * Rh) @ self.Ds[i].T
        return h

    def hidden_states(self, x):
        """Return [h0=x, h1, h2, ...] where h_i is the output after applying layer i."""
        states = [x]
        h = x
        for i in range(self.num_layers):
            Lh = h @ self.Ls[i].T
            Rh = h @ self.Rs[i].T
            h = h + (Lh * Rh) @ self.Ds[i].T
            states.append(h)
        return states


# --- Input distributions (all work for arbitrary n) ---

def gaussian(*shape):
    return torch.randn(*shape)

def half_gaussian(*shape):
    return torch.abs(torch.randn(*shape))

def bimodal(*shape):
    signs = 2 * torch.randint(0, 2, shape).float() - 1
    return signs + torch.randn(*shape) * 0.3

def uniform(*shape):
    return torch.rand(*shape) * 2 - 1

def laplace(*shape):
    return torch.distributions.Laplace(0, 1 / math.sqrt(2)).sample(torch.Size(shape))

def sparse_spikes(*shape, p=0.25):
    mask = (torch.rand(*shape) < p).float()
    return mask * torch.randn(*shape) / math.sqrt(p)

def permutation(*shape):
    batch = shape[0]
    base = torch.arange(1, shape[1] + 1).float()
    idx = torch.stack([torch.randperm(shape[1]) for _ in range(batch)])
    return base[idx]

def correlated_gaussian(*shape):
    """AR(1) correlation: Σ_{ij} = 0.8^|i-j|. Works for any n."""
    n = shape[-1]
    x = torch.randn(*shape)
    i_idx = torch.arange(n).unsqueeze(1)
    j_idx = torch.arange(n).unsqueeze(0)
    A = (0.8 ** (i_idx - j_idx).abs()).float()
    L = torch.linalg.cholesky(A)
    return x @ L.T
