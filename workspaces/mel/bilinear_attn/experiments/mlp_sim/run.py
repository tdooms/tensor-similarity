"""2-layer bilinear MLP: TN similarity (order-4 symmetric tensor) vs MC similarity.

Architecture
------------
    o1 = (L1 x) * (R1 x)            x  in R^d,        L1,R1 in R^{d_hidden x d}
    o2 = (L2 o1) * (R2 o1)          o2 in R^{d_out},  L2,R2 in R^{d_out x d_hidden}

Each output component o2_k is a homogeneous degree-4 polynomial in x:
    o2_k = sum_{abcd} T^{(k)}_{abcd} x_a x_b x_c x_d,
    T^{(k)}_{abcd} = sum_{j,j'} L2_{kj} R2_{kj'} L1_{ja} R1_{jb} L1_{j'c} R1_{j'd}.

After symmetrising T^{(k)} over {a,b,c,d}, the Gaussian functional similarity
between two models (see mlp_sim/README.md, n=4) is

    E[A(x) B(x)] = 24 <A,B> + 72 <tau A, tau B> + 9 tau^2(A) tau^2(B),  x ~ N(0, I_d).

We sum over the output dimension and normalise to get a cosine similarity.
We compare this closed-form quantity to a Monte-Carlo estimate where x ~ N(0, I_d)
is sampled directly (no embedding), for 10 independently-seeded models trained
on PCA-reduced MNIST.
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class BilinearMLP(nn.Module):
    """o1 = (L1 x) * (R1 x); o2 = (L2 o1) * (R2 o1)."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        # Kaiming-ish init; scaled by fan_in to keep activations bounded.
        self.L1 = nn.Parameter(torch.randn(d_hidden, d_in, generator=g) / math.sqrt(d_in))
        self.R1 = nn.Parameter(torch.randn(d_hidden, d_in, generator=g) / math.sqrt(d_in))
        self.L2 = nn.Parameter(torch.randn(d_out, d_hidden, generator=g) / math.sqrt(d_hidden))
        self.R2 = nn.Parameter(torch.randn(d_out, d_hidden, generator=g) / math.sqrt(d_hidden))
        self.d_in, self.d_hidden, self.d_out = d_in, d_hidden, d_out

    def forward(self, x: Tensor) -> Tensor:  # x: (..., d_in)
        o1 = (x @ self.L1.T) * (x @ self.R1.T)
        o2 = (o1 @ self.L2.T) * (o1 @ self.R2.T)
        return o2


# ---------------------------------------------------------------------------
# Data: MNIST -> PCA to d_in dims, standardised so Gaussian assumption is sane
# ---------------------------------------------------------------------------
def load_mnist_pca(d_in: int, data_root: str = "./data", device: str = "cpu"):
    tr = datasets.MNIST(data_root, train=True, download=True,
                        transform=transforms.ToTensor())
    te = datasets.MNIST(data_root, train=False, download=True,
                        transform=transforms.ToTensor())
    x_tr = tr.data.reshape(len(tr), -1).float() / 255.0
    x_te = te.data.reshape(len(te), -1).float() / 255.0
    y_tr = tr.targets.long()
    y_te = te.targets.long()

    mean = x_tr.mean(0, keepdim=True)
    xc = x_tr - mean
    # PCA via torch.linalg.svd on training data
    U, S, Vh = torch.linalg.svd(xc, full_matrices=False)
    V = Vh[:d_in].T  # (784, d_in)
    # project and whiten per-component so each of the d_in dims has unit variance
    proj_tr = xc @ V
    proj_te = (x_te - mean) @ V
    std = proj_tr.std(0, keepdim=True) + 1e-8
    proj_tr = proj_tr / std
    proj_te = proj_te / std
    return (proj_tr.to(device), y_tr.to(device),
            proj_te.to(device), y_te.to(device))


def train_one(model: BilinearMLP, x_tr, y_tr, x_te, y_te,
              epochs: int = 8, batch_size: int = 256, lr: float = 3e-3,
              device: str = "cpu", verbose: bool = False) -> dict:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)
    for ep in range(epochs):
        model.train()
        tot, correct, loss_sum, n = 0, 0, 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * xb.size(0)
            correct += (logits.argmax(-1) == yb).sum().item()
            n += xb.size(0)
        if verbose:
            with torch.no_grad():
                model.eval()
                te_acc = (model(x_te).argmax(-1) == y_te).float().mean().item()
            print(f"  ep{ep}: loss={loss_sum/n:.3f} tr_acc={correct/n:.3f} te_acc={te_acc:.3f}")
    model.eval()
    with torch.no_grad():
        te_acc = (model(x_te).argmax(-1) == y_te).float().mean().item()
    return {"test_acc": te_acc}


# ---------------------------------------------------------------------------
# TN similarity via the order-4 symmetric tensor (README formula)
# ---------------------------------------------------------------------------
def build_symmetric_T(model: BilinearMLP) -> Tensor:
    """Return T of shape (d_out, d, d, d, d), symmetric in the last 4 axes.

    T^{(k)}_{abcd} = sum_{j,j'} L2_{kj} R2_{kj'} L1_{ja} R1_{jb} L1_{j'c} R1_{j'd}
    Averaged over the 24 permutations of (a,b,c,d).
    """
    L1, R1, L2, R2 = model.L1, model.R1, model.L2, model.R2
    # shape (k, a, b, c, d)
    T = torch.einsum("kj,km,ja,jb,mc,md->kabcd", L2, R2, L1, R1, L1, R1)

    T_sym = torch.zeros_like(T)
    perms = list(itertools.permutations(range(4)))
    for p in perms:
        # permute the last 4 axes
        T_sym = T_sym + T.permute(0, 1 + p[0], 1 + p[1], 1 + p[2], 1 + p[3])
    T_sym = T_sym / len(perms)
    return T_sym  # (d_out, d, d, d, d), symmetric in abcd


def tn_inner_product(TA: Tensor, TB: Tensor) -> Tensor:
    """E[A(x) B(x)] summed over output dim, under x ~ N(0, I).

    Uses the n=4 formula (see README):
        E[A(x)B(x)] = 24 <A,B> + 72 <tau A, tau B> + 9 tau^2(A) tau^2(B).
    Expects TA, TB symmetric in the last 4 axes.
    """
    # Full contraction
    full = torch.einsum("kabcd,kabcd->", TA, TB)
    # tau: contract one pair -> 2-tensor (k, c, d)
    tauA = torch.einsum("kaacd->kcd", TA)
    tauB = torch.einsum("kaacd->kcd", TB)
    once = torch.einsum("kcd,kcd->", tauA, tauB)
    # tau^2: scalar per output (k,)
    tau2A = torch.einsum("kcc->k", tauA)
    tau2B = torch.einsum("kcc->k", tauB)
    twice = (tau2A * tau2B).sum()
    return 24.0 * full + 72.0 * once + 9.0 * twice


def tn_cosine(model_A: BilinearMLP, model_B: BilinearMLP) -> float:
    with torch.no_grad():
        TA = build_symmetric_T(model_A).double()
        TB = build_symmetric_T(model_B).double()
        ab = tn_inner_product(TA, TB)
        aa = tn_inner_product(TA, TA)
        bb = tn_inner_product(TB, TB)
        return (ab / torch.sqrt(aa * bb)).item()


# ---------------------------------------------------------------------------
# MC similarity: x ~ N(0, I_d), cosine of f_A(x) and f_B(x)
# ---------------------------------------------------------------------------
@torch.no_grad()
def mc_cosine(model_A: BilinearMLP, model_B: BilinearMLP,
              n_samples: int = 200_000, batch_size: int = 4096,
              dtype: torch.dtype = torch.float64,
              device: str = "cpu") -> float:
    A = model_A.to(device=device, dtype=dtype).eval()
    B = model_B.to(device=device, dtype=dtype).eval()
    d = A.d_in
    ip_ab = torch.zeros((), device=device, dtype=dtype)
    ip_aa = torch.zeros((), device=device, dtype=dtype)
    ip_bb = torch.zeros((), device=device, dtype=dtype)
    done = 0
    while done < n_samples:
        bs = min(batch_size, n_samples - done)
        x = torch.randn(bs, d, device=device, dtype=dtype)
        fa = A(x)
        fb = B(x)
        ip_ab += (fa * fb).sum()
        ip_aa += (fa * fa).sum()
        ip_bb += (fb * fb).sum()
        done += bs
    ip_ab /= n_samples
    ip_aa /= n_samples
    ip_bb /= n_samples
    return (ip_ab / torch.sqrt(ip_aa * ip_bb)).item()


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------
def run(
    n_seeds: int = 10,
    d_in: int = 8,
    d_hidden: int = 8,
    d_out: int = 10,
    epochs: int = 8,
    mc_samples: int = 200_000,
    device: str = "cpu",
    out_dir: str | Path = None,
):
    out_dir = Path(out_dir) if out_dir else Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading MNIST and projecting to {d_in}-D via PCA ...")
    x_tr, y_tr, x_te, y_te = load_mnist_pca(d_in, device=device)

    print(f"Training {n_seeds} bilinear MLPs ...")
    models = []
    accs = []
    for s in range(n_seeds):
        torch.manual_seed(s)
        m = BilinearMLP(d_in, d_hidden, d_out, seed=s)
        info = train_one(m, x_tr, y_tr, x_te, y_te, epochs=epochs, device=device)
        accs.append(info["test_acc"])
        models.append(m.cpu())
        print(f"  seed {s}: test acc = {info['test_acc']:.3f}")

    # Pairwise similarities
    pairs = list(itertools.combinations(range(n_seeds), 2))
    tn_vals, mc_vals = [], []
    print(f"\nComputing TN and MC similarities for {len(pairs)} pairs ...")
    for i, j in tqdm(pairs):
        tn_vals.append(tn_cosine(models[i], models[j]))
        mc_vals.append(mc_cosine(models[i], models[j],
                                 n_samples=mc_samples, device=device))

    tn_arr = np.array(tn_vals)
    mc_arr = np.array(mc_vals)
    r = float(np.corrcoef(tn_arr, mc_arr)[0, 1])
    print(f"\nPearson correlation (TN vs MC): r = {r:.4f}")
    # Relative agreement per pair
    diff = np.abs(tn_arr - mc_arr)
    print(f"|TN - MC|  mean={diff.mean():.3e}  max={diff.max():.3e}")

    # Plot
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lo = float(min(tn_arr.min(), mc_arr.min()))
    hi = float(max(tn_arr.max(), mc_arr.max()))
    pad = 0.05 * (hi - lo + 1e-12)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            ls="--", c="gray", lw=1, label="y = x")
    ax.scatter(tn_arr, mc_arr, s=40, alpha=0.85)
    ax.set_xlabel("TN cosine similarity (Gaussian, closed-form)")
    ax.set_ylabel("MC cosine similarity (Gaussian, 200k samples)")
    ax.set_title(
        f"2-layer bilinear MLP on MNIST (d={d_in}, hidden={d_hidden}, out={d_out})\n"
        f"{n_seeds} seeds, {len(pairs)} pairs, Pearson r = {r:.3f}"
    )
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig_path = out_dir / "tn_vs_mc.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved plot to {fig_path}")

    # Dump numbers
    np.savez(
        out_dir / "results.npz",
        tn=tn_arr, mc=mc_arr,
        pairs=np.array(pairs), accs=np.array(accs), corr=r,
    )
    return {"tn": tn_arr, "mc": mc_arr, "corr": r, "accs": accs}


if __name__ == "__main__":
    run()
