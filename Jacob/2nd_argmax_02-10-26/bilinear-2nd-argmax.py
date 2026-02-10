# Written by Logan (50%) and Claude (80%)

import torch
import torch.nn as nn
import math

# Train a good bilinear network for n=4
class BilinearStack(nn.Module):
    def __init__(self, n, num_layers=3, rank=32, use_linear=True):
        super().__init__()
        self.n = n
        self.num_layers = num_layers
        
        self.Ls = nn.ParameterList()
        self.Rs = nn.ParameterList()
        self.Ds = nn.ParameterList()
        # self.Ws = nn.ParameterList() if use_linear else None
        
        for _ in range(num_layers):
            self.Ls.append(nn.Parameter(torch.randn(rank, n) * 0.1))
            self.Rs.append(nn.Parameter(torch.randn(rank, n) * 0.1))
            self.Ds.append(nn.Parameter(torch.randn(n, rank) * 0.1))
            # if use_linear:
            #     self.Ws.append(nn.Parameter(torch.randn(n, n) * 0.1))
    
    def forward(self, x):
        h = x
        for i in range(self.num_layers):
            Lh = h @ self.Ls[i].T
            Rh = h @ self.Rs[i].T
            bilinear = (Lh * Rh) @ self.Ds[i].T
            h = h + bilinear
            # if self.Ws:
            #    h += h @ self.Ws[i].T
        return h


def task_2nd_argmax(x):
    return x.argsort(-1)[..., -2]

N = 4
BATCH_SIZE = 512
TRAIN_STEPS = 10000

# Train
# Parametrized with a function because our goal is to compare across distributions
def train(dist_fn):
    model = BilinearStack(N, num_layers=3, rank=32, use_linear=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for step in range(TRAIN_STEPS):
        x = dist_fn(BATCH_SIZE, N) # KEY LINE
        targets = task_2nd_argmax(x)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()

    return model

def gaussian(*shape):
    return torch.randn(*shape)

def half_gaussian(*shape):
    return torch.abs(torch.randn(*shape))

def bimodal(*shape):                                                          
    signs = 2 * torch.randint(0, 2, shape).float() - 1
    return signs + torch.randn(*shape) * 0.3

def uniform(*shape):
    return torch.rand(*shape) * 2 - 1
    # scaling to have the same mean and variance as the gaussian, probably

# claude "messed up" ones

def laplace(*shape):
    return torch.distributions.Laplace(0, 1/math.sqrt(2)).sample(torch.Size(shape))

def sparse_spikes(*shape, p=0.25):
    mask = (torch.rand(*shape) < p).float()
    return mask * torch.randn(*shape) / math.sqrt(p)

def permutation(*shape):
    batch = shape[0]
    base = torch.arange(1, shape[1] + 1).float()
    idx = torch.stack([torch.randperm(shape[1]) for _ in range(batch)])
    return base[idx]

def correlated_gaussian(*shape):
    x = torch.randn(*shape)
    # Mix dimensions so they're correlated
    A = torch.tensor([[1., 0.8, 0.3, 0.1],
                        [0.8, 1., 0.5, 0.2],
                        [0.3, 0.5, 1., 0.7],
                        [0.1, 0.2, 0.7, 1.]])
    L = torch.linalg.cholesky(A)
    return x @ L.T


def test_dist(dist_fn, n=4, samples=5):                                       
    x = dist_fn(samples, n)
    print(x) 

# test_dist(gaussian)
# test_dist(torch.rand)
# test_dist(uniform)
# test_dist(bimodal)
# test_dist(half_gaussian)

if __name__ == '__main__':
    model_gaussian = train(gaussian)
    model_half_gaussian = train(half_gaussian)
    model_bimodal = train(bimodal)
    model_uniform = train(uniform)

    # Test accuracy
    SAMPLES = 100000
    with torch.no_grad():
        x = torch.randn(SAMPLES, N)
        targets = task_2nd_argmax(x)
        logits = model_gaussian(x)
        acc = (logits.argmax(-1) == targets).float().mean()
        print(f"Model accuracy: {acc.item():.1%}")