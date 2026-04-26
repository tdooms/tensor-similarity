"""Residual-stream and deeper bilinear MLPs: TN vs MC similarity.

Extends ``run.py`` with:
- A unified model  B_i(u) = (L_i u) * (R_i u)  with square matrices L_i, R_i in R^{d x d},
  optional residual  x_{i+1} = x_i + B_i(x_i)  (vs  x_{i+1} = B_i(x_i)),
  and a linear head  W_u: d_out x d.
- A polynomial forward that tracks the degree-m homogeneous pieces of the output,
  materialising a *symmetrised* order-m tensor per degree.
- The general Gaussian similarity formula from the README for any order n:
        E[A(x) B(x)] = sum_{r=0..floor(n/2)} c_{n,r} <tau^r A, tau^r B>,
        c_{n,r} = binom(n, 2r)^2 * (2r-1)!!^2 * (n-2r)!.
- For residual models, y(x) is a sum of ordinary-monomial homogeneous pieces
  y^[m](x), which are *not* Hermite (Wiener-chaos) components. Under centred
  Gaussian input, cross-degree terms of *different parity* vanish (odd moments),
  but same-parity cross terms (e.g. E[y_A^[1] y_B^[3]], E[y_A^[2] y_B^[4]]) do
  not. The full similarity is
        E[<y_A(x), y_B(x)>] = sum_{m1, m2 same parity} E[<y_A^[m1], y_B^[m2]>],
  and each pair-inner-product is the same Isserlis pairing sum as the n=m=m1=m2
  case, generalised to unequal arities:
        E[A(x)B(x)] = sum_{r, r' : m1-2r = m2-2r' = k >= 0}
                      C(m1,2r)(2r-1)!! C(m2,2r')(2r'-1)!! k!
                      <tau^r A, tau^{r'} B>.
  (See the per-degree TN vs MC sanity check: per-degree AB matches to MC noise,
  and the 3.36 cross-degree term is recovered exactly by the formula above.)

Experiments:
- run_2layer_residual()   : degrees 1..4
- run_3layer_nonres()     : pure degree 8
- run_3layer_residual()   : degrees 1..8

All data: MNIST PCA-projected to d=8, whitened per component.
Runs on GPU in fp32 (fp64 via --dtype=float64).
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
from tqdm import tqdm

# Re-use the MNIST loader and generic training loop from run.py.
from experiments.mlp_sim.run import load_mnist_pca


# ---------------------------------------------------------------------------
# Unified bilinear MLP
# ---------------------------------------------------------------------------
class BilinearMLPv2(nn.Module):
    """N-layer bilinear MLP with optional residual stream.

    Per-layer:  B_i(u) = (L_i u) * (R_i u),   L_i, R_i in R^{d x d}.
    Residual=True :  x_{i+1} = x_i + B_i(x_i).
    Residual=False:  x_{i+1} = B_i(x_i).
    Head         :  y = W_u x_N,   W_u in R^{d_out x d}.
    """

    def __init__(self, d: int, d_out: int, n_layers: int,
                 residual: bool, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.L = nn.ParameterList([
            nn.Parameter(torch.randn(d, d, generator=g) / math.sqrt(d))
            for _ in range(n_layers)
        ])
        self.R = nn.ParameterList([
            nn.Parameter(torch.randn(d, d, generator=g) / math.sqrt(d))
            for _ in range(n_layers)
        ])
        self.Wu = nn.Parameter(torch.randn(d_out, d, generator=g) / math.sqrt(d))
        self.d, self.d_out, self.n_layers, self.residual = d, d_out, n_layers, residual

    def forward(self, x: Tensor) -> Tensor:  # x: (..., d)
        for Li, Ri in zip(self.L, self.R):
            B = (x @ Li.T) * (x @ Ri.T)
            x = x + B if self.residual else B
        return x @ self.Wu.T


def train_one(model: BilinearMLPv2, x_tr, y_tr, x_te, y_te,
              epochs: int = 8, batch_size: int = 256, lr: float = 3e-3,
              device: str = "cpu") -> dict:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        te_acc = (model(x_te).argmax(-1) == y_te).float().mean().item()
    return {"test_acc": te_acc}


# ---------------------------------------------------------------------------
# Polynomial forward: output as {degree m -> symmetrised tensor (d_out, d, ..., d)}
# ---------------------------------------------------------------------------
def _pair_symmetrize(T: Tensor, m1: int, m2: int) -> Tensor:
    """Symmetrise T over all m1+m2 input axes, assuming T is already symmetric
    in its first m1 input axes and (independently) in its last m2 input axes.

    Only C(m1+m2, m1) coset reps of S_{m1+m2} / (S_{m1} x S_{m2}) are needed.
    T shape: (f, d, ..., d) with m1+m2 input axes (axis 0 is the feature axis).
    """
    if m1 == 0 or m2 == 0:
        return T
    n = m1 + m2
    acc = torch.zeros_like(T)
    count = 0
    for subset in itertools.combinations(range(n), m1):
        subset_set = set(subset)
        complement = [p for p in range(n) if p not in subset_set]
        # Build permutation mapping old axes -> new positions:
        # new position subset[i]   <- old input axis i   (from the L-side block)
        # new position complement[j] <- old input axis m1+j (from the R-side block)
        perm_dims = [0] * (n + 1)
        perm_dims[0] = 0
        for i, s in enumerate(subset):
            perm_dims[1 + s] = 1 + i
        for j, c in enumerate(complement):
            perm_dims[1 + c] = 1 + m1 + j
        acc.add_(T.permute(*perm_dims))
        count += 1
    acc.div_(count)
    return acc


def _hadamard_polys(L_P: dict[int, Tensor], R_P: dict[int, Tensor]) -> dict[int, Tensor]:
    """Elementwise (feature axis) product of two symmetric polynomial reps.

    L_P, R_P are {m: T_m}, T_m shape (f, d, ..., d) symmetric in its m input axes.
    Output is {m: T_m}, each also symmetric in its m input axes (pair-symmetrised).
    """
    out: dict[int, Tensor] = {}
    for m1, T_L in L_P.items():
        for m2, T_R in R_P.items():
            f = T_L.shape[0]
            shape_L = T_L.shape + (1,) * m2
            shape_R = (f,) + (1,) * m1 + T_R.shape[1:]
            prod = T_L.reshape(shape_L) * T_R.reshape(shape_R)
            prod = _pair_symmetrize(prod, m1, m2)
            m = m1 + m2
            out[m] = out[m] + prod if m in out else prod
    return out


def _linear_apply(W: Tensor, P: dict[int, Tensor]) -> dict[int, Tensor]:
    """Apply linear map W (f_out, f_in) to the feature axis of every T_m."""
    return {m: torch.einsum("oi,i...->o...", W, T) for m, T in P.items()}


def _add_polys(P: dict[int, Tensor], Q: dict[int, Tensor]) -> dict[int, Tensor]:
    out: dict[int, Tensor] = {}
    for m in set(P) | set(Q):
        if m in P and m in Q:
            out[m] = P[m] + Q[m]
        elif m in P:
            out[m] = P[m]
        else:
            out[m] = Q[m]
    return out


@torch.no_grad()
def forward_polynomial(model: BilinearMLPv2) -> dict[int, Tensor]:
    """Return {m: T_m} with T_m fully symmetric in its m input axes.

    Each T_m has shape (d_out, d, d, ..., d). The polynomial is kept symmetric
    throughout the forward pass via per-hadamard ``_pair_symmetrize``, which
    avoids the O(n!) final symmetrisation.
    """
    d = model.d
    device = model.Wu.device
    dtype = model.Wu.dtype

    # Seed poly: x_k = sum_a I_{k,a} x_a. Degree-1 tensor (already 'symmetric').
    P: dict[int, Tensor] = {1: torch.eye(d, device=device, dtype=dtype)}

    for Li, Ri in zip(model.L, model.R):
        L_P = _linear_apply(Li, P)
        R_P = _linear_apply(Ri, P)
        B = _hadamard_polys(L_P, R_P)  # already pair-symmetrised per degree
        P = _add_polys(P, B) if model.residual else B

    # Head; linear apply preserves input-axis symmetry.
    return _linear_apply(model.Wu, P)


# ---------------------------------------------------------------------------
# General TN similarity from symmetric tensors
# ---------------------------------------------------------------------------
def _double_factorial(k: int) -> int:
    if k < 0:
        return 1  # (-1)!! = 1 by convention
    return math.prod(range(k, 0, -2)) if k > 0 else 1


def isserlis_coefficient(n: int, r: int) -> int:
    """c_{n,r} = binom(n, 2r)^2 * (2r-1)!!^2 * (n-2r)! ."""
    return math.comb(n, 2 * r) ** 2 * _double_factorial(2 * r - 1) ** 2 * math.factorial(n - 2 * r)


def _trace_chain(T: Tensor, m: int) -> list[Tensor]:
    """Return [T, tau T, tau^2 T, ..., tau^{m//2} T].

    tau traces the first two remaining input axes (axes 1 and 2).
    """
    out = [T]
    cur = T
    for _ in range(m // 2):
        cur = torch.einsum("kaa...->k...", cur)
        out.append(cur)
    return out


def tn_pair_inner_product(
    trA: list[Tensor], trB: list[Tensor], m1: int, m2: int
) -> Tensor:
    """E[A(x) B(x)] for sym order-m1 A, sym order-m2 B, via pre-cached trace lists.

    Zero if m1+m2 odd. Otherwise sums the Isserlis pairings with
        k = m1 - 2r = m2 - 2r',
        coeff = C(m1,2r) (2r-1)!! C(m2,2r') (2r'-1)!! k! .
    For m1 == m2 == n this reduces to the diagonal README formula.
    """
    device = trA[0].device
    dtype = trA[0].dtype
    if (m1 + m2) % 2 != 0:
        return torch.zeros((), device=device, dtype=dtype)
    ip = torch.zeros((), device=device, dtype=dtype)
    for r in range(m1 // 2 + 1):
        k = m1 - 2 * r
        rp2 = m2 - k
        if rp2 < 0 or rp2 % 2 != 0:
            continue
        rp = rp2 // 2
        if rp > m2 // 2:
            continue
        coeff = (
            math.comb(m1, 2 * r) * _double_factorial(2 * r - 1)
            * math.comb(m2, 2 * rp) * _double_factorial(2 * rp - 1)
            * math.factorial(k)
        )
        ip = ip + coeff * (trA[r] * trB[rp]).sum()
    return ip


def tn_inner_product_polys(
    PA: dict[int, Tensor], PB: dict[int, Tensor]
) -> Tensor:
    """E[y_A(x) y_B(x)] summed over output dim, summed over all degree pairs.

    Same-parity cross-degree terms are included; different-parity ones vanish.
    """
    trA = {m: _trace_chain(T, m) for m, T in PA.items()}
    trB = {m: _trace_chain(T, m) for m, T in PB.items()}
    any_T = next(iter(PA.values()))
    ip = torch.zeros((), device=any_T.device, dtype=any_T.dtype)
    for m1 in PA:
        for m2 in PB:
            ip = ip + tn_pair_inner_product(trA[m1], trB[m2], m1, m2)
    return ip


def tn_cosine(model_A: BilinearMLPv2, model_B: BilinearMLPv2) -> float:
    PA = forward_polynomial(model_A)
    PB = forward_polynomial(model_B)
    ab = tn_inner_product_polys(PA, PB)
    aa = tn_inner_product_polys(PA, PA)
    bb = tn_inner_product_polys(PB, PB)
    return (ab / (aa * bb).sqrt()).item()


# ---------------------------------------------------------------------------
# MC similarity: Gaussian x, cosine of model outputs
# ---------------------------------------------------------------------------
@torch.no_grad()
def mc_cosine(model_A: BilinearMLPv2, model_B: BilinearMLPv2,
              n_samples: int = 300_000, batch_size: int = 8192,
              dtype: torch.dtype = torch.float64,
              device: str = "cpu") -> float:
    A = model_A.to(device=device, dtype=dtype).eval()
    B = model_B.to(device=device, dtype=dtype).eval()
    d = A.d
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
# Generic experiment driver
# ---------------------------------------------------------------------------
def _run_experiment(
    tag: str,
    n_layers: int,
    residual: bool,
    n_seeds: int = 10,
    d: int = 8,
    d_out: int = 10,
    epochs: int = 8,
    mc_samples: int = 300_000,
    tn_dtype: torch.dtype = torch.float32,
    mc_dtype: torch.dtype = torch.float64,
    device: str = None,
    out_dir: str | Path = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir) if out_dir else Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{tag}] device={device}, n_layers={n_layers}, residual={residual}")
    print(f"[{tag}] Loading MNIST PCA-{d} ...")
    x_tr, y_tr, x_te, y_te = load_mnist_pca(d, device=device)

    print(f"[{tag}] Training {n_seeds} seeds ...")
    models: list[BilinearMLPv2] = []
    accs = []
    for s in range(n_seeds):
        torch.manual_seed(s)
        m = BilinearMLPv2(d, d_out, n_layers, residual=residual, seed=s)
        info = train_one(m, x_tr, y_tr, x_te, y_te, epochs=epochs, device=device)
        accs.append(info["test_acc"])
        print(f"[{tag}]   seed {s}: test_acc = {info['test_acc']:.3f}")
        models.append(m)

    # Precompute symmetric polynomial tensors for each seed (on GPU).
    print(f"[{tag}] Computing symmetric polynomial tensors per seed ...")
    polys: list[dict[int, Tensor]] = []
    for s, m in enumerate(tqdm(models)):
        m.to(device=device, dtype=tn_dtype).eval()
        polys.append(forward_polynomial(m))

    # Pairwise similarities
    pairs = list(itertools.combinations(range(n_seeds), 2))
    print(f"[{tag}] Computing {len(pairs)} pairs: TN + MC ...")
    tn_vals, mc_vals = [], []
    for i, j in tqdm(pairs):
        ab = tn_inner_product_polys(polys[i], polys[j])
        aa = tn_inner_product_polys(polys[i], polys[i])
        bb = tn_inner_product_polys(polys[j], polys[j])
        tn_vals.append((ab / (aa * bb).sqrt()).item())

        mc_vals.append(mc_cosine(models[i], models[j],
                                 n_samples=mc_samples, dtype=mc_dtype, device=device))

    tn_arr = np.array(tn_vals)
    mc_arr = np.array(mc_vals)
    r = float(np.corrcoef(tn_arr, mc_arr)[0, 1])
    diff = np.abs(tn_arr - mc_arr)
    print(f"[{tag}] Pearson r(TN, MC) = {r:.4f}   |TN-MC| mean={diff.mean():.2e} max={diff.max():.2e}")

    # Plot
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lo = float(min(tn_arr.min(), mc_arr.min()))
    hi = float(max(tn_arr.max(), mc_arr.max()))
    pad = 0.05 * (hi - lo + 1e-12)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            ls="--", c="gray", lw=1, label="y = x")
    ax.scatter(tn_arr, mc_arr, s=40, alpha=0.85)
    ax.set_xlabel("TN cosine (closed-form, symmetric tensor)")
    ax.set_ylabel(f"MC cosine (Gaussian, {mc_samples//1000}k samples)")
    title = (
        f"{n_layers}-layer bilinear MLP ({'residual' if residual else 'non-residual'})\n"
        f"d={d}, d_out={d_out}, {n_seeds} seeds, {len(pairs)} pairs, Pearson r = {r:.3f}"
    )
    ax.set_title(title)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig_path = out_dir / f"tn_vs_mc_{tag}.png"
    fig.savefig(fig_path, dpi=150)
    print(f"[{tag}] Saved plot -> {fig_path}")

    np.savez(
        out_dir / f"results_{tag}.npz",
        tn=tn_arr, mc=mc_arr, pairs=np.array(pairs), accs=np.array(accs), corr=r,
    )
    return {"tn": tn_arr, "mc": mc_arr, "corr": r, "accs": accs}


# ---------------------------------------------------------------------------
# Named runners
# ---------------------------------------------------------------------------
def run_2layer_residual(**kw):
    return _run_experiment(tag="2L_res", n_layers=2, residual=True, **kw)


def run_3layer_nonres(**kw):
    # n=8 symmetrisation dominates; fp32 keeps memory/time reasonable.
    return _run_experiment(tag="3L_nonres", n_layers=3, residual=False,
                           tn_dtype=torch.float32, **kw)


def run_3layer_residual(**kw):
    return _run_experiment(tag="3L_res", n_layers=3, residual=True,
                           tn_dtype=torch.float32, **kw)


# ---------------------------------------------------------------------------
# Minimal wrapper that exposes a BilinearMLPv2 through src.components, so we
# can call src.components.similarity.similarity(A, B) and correlate its output
# against our TN cosine.
#
# Two important caveats:
#
# (1) Non-residual only: src.MLP is ``lerp(x, d(l*r), s)`` = ``(1-s) x + s
#     d(l*r)``, which matches our ``B(x) = (L x)(R x)`` *exactly* at s=1
#     (identity d, residual coefficient 0). src.MLP cannot represent the pure
#     residual form ``x + B(x)`` (the lerp introduces different per-degree
#     coefficients), so we raise for residual models.
#
# (2) src.components.similarity propagates only the *second-moment* matrix
#     block s_xy = E[f_x f_y^T] through each layer, then applies an Isserlis
#     plan at the next layer treating the previous-layer output as Gaussian.
#     This is exact for 1-layer bilinear MLPs with Gaussian input, but only
#     approximate for stacked bilinear MLPs: B_1(x) is quartic in x and its
#     4th moments in x do NOT factor via Isserlis on s_1 alone. Our polynomial
#     TN is exact; src is a coarser proxy for depth >= 2.
#     (Verified: 1 MLP + head matches MC exactly; 2 MLPs + head differs from
#     MC by ~3x in absolute ab/aa/bb and ~0.04 in cosine.)
# ---------------------------------------------------------------------------
def _ensure_src_on_path():
    import sys as _sys
    from pathlib import Path as _Path
    _ROOT = _Path(__file__).resolve().parents[5]  # .../tensor-mars
    if (_ROOT / "src" / "components").is_dir() and str(_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_ROOT))


class SrcBilinearWrap(nn.Module):
    """Expose BilinearMLPv2 (non-residual) as a src.models.base.Model.

    Each bilinear layer B_i(x) = (L_i x) * (R_i x) is mapped to
    ``src.components.MLP(d, d, bias=False, scale=1.0)`` with
    ``l = L_i, r = R_i, d = I``. Head is ``src.components.Linear(d, d_out)``
    with ``weight = W_u``.
    """

    def __init__(self, base: BilinearMLPv2):
        if base.residual:
            raise NotImplementedError(
                "src wrapper only supports non-residual models: src.MLP's "
                "lerp form (1-s)x + s d(l*r) does not equal x + B(x).")
        _ensure_src_on_path()
        from src.components.mlp import MLP as SrcMLP
        from src.components.linear import Linear as SrcLinear
        super().__init__()
        d, d_out = base.d, base.d_out
        mlps = []
        for Li, Ri in zip(base.L, base.R):
            m = SrcMLP(d, d, bias=False, scale=1.0)
            with torch.no_grad():
                m.l.weight.copy_(Li)
                m.r.weight.copy_(Ri)
                m.d.weight.copy_(torch.eye(d, dtype=Li.dtype, device=Li.device))
            mlps.append(m)
        self.mlps = nn.ModuleList(mlps)
        self.head = SrcLinear(d, d_out, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(base.Wu)

    # src.models.base.Model interface.
    def components(self):
        return [*self.mlps, self.head]


def src_tn_cosine(model_A: BilinearMLPv2, model_B: BilinearMLPv2,
                  device: str = "cpu", dtype: torch.dtype = torch.float64) -> float:
    """Cosine similarity via src.components.similarity (state-propagation)."""
    _ensure_src_on_path()
    from src.components.similarity import similarity as _src_similarity
    import copy as _copy
    a = _copy.deepcopy(model_A).to(device=device, dtype=dtype)
    b = _copy.deepcopy(model_B).to(device=device, dtype=dtype)
    A = SrcBilinearWrap(a).to(device=device, dtype=dtype)
    B = SrcBilinearWrap(b).to(device=device, dtype=dtype)
    state = _src_similarity(A, B)
    # state.s_xy shape: (n_a=1, d_a=d_out+1, n_b=1, d_b=d_out+1) in padded rep
    # (see _initial_state: einsum 'ij,kl->ikjl' over (I_n, I_d)).
    # Index 0 is the deterministic "constant" axis carried by padding; exclude
    # it from the inner product so the cosine is over actual outputs.
    ab = state.s_ab[0, :, 0, :].diagonal()[1:].sum()
    aa = state.s_aa[0, :, 0, :].diagonal()[1:].sum()
    bb = state.s_bb[0, :, 0, :].diagonal()[1:].sum()
    return (ab / (aa * bb).sqrt()).item()


def run_src_correlation(tag: str, n_layers: int, n_seeds: int = 10,
                        d: int = 8, d_out: int = 10, epochs: int = 8,
                        out_dir: str | Path = None, device: str = None):
    """Train n_seeds non-residual BilinearMLPv2 models; correlate our TN cosine
    against src TN cosine over all pairs."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir) if out_dir else Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{tag}] Training {n_seeds} non-residual {n_layers}-layer models ...")
    x_tr, y_tr, x_te, y_te = load_mnist_pca(d, device=device)
    models = []
    for s in range(n_seeds):
        torch.manual_seed(s)
        m = BilinearMLPv2(d, d_out, n_layers, residual=False, seed=s)
        info = train_one(m, x_tr, y_tr, x_te, y_te, epochs=epochs, device=device)
        print(f"[{tag}]   seed {s}: test_acc = {info['test_acc']:.3f}")
        models.append(m)

    # Our TN cosine (symmetric polynomial).
    print(f"[{tag}] Our TN cosine per seed ...")
    polys = []
    for m in models:
        m.to(device=device, dtype=torch.float32).eval()
        polys.append(forward_polynomial(m))

    pairs = list(itertools.combinations(range(n_seeds), 2))
    ours, srcs = [], []
    print(f"[{tag}] Pairwise cosines: ours vs src ...")
    for i, j in tqdm(pairs):
        ab = tn_inner_product_polys(polys[i], polys[j])
        aa = tn_inner_product_polys(polys[i], polys[i])
        bb = tn_inner_product_polys(polys[j], polys[j])
        ours.append((ab / (aa * bb).sqrt()).item())
        srcs.append(src_tn_cosine(models[i], models[j]))

    ours = np.array(ours); srcs = np.array(srcs)
    r = float(np.corrcoef(ours, srcs)[0, 1])
    diff = np.abs(ours - srcs)
    print(f"[{tag}] Pearson r(ours, src) = {r:.4f}   |diff| mean={diff.mean():.2e} max={diff.max():.2e}")

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lo = float(min(ours.min(), srcs.min()))
    hi = float(max(ours.max(), srcs.max()))
    pad = 0.05 * (hi - lo + 1e-12)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", c="gray", lw=1, label="y = x")
    ax.scatter(ours, srcs, s=40, alpha=0.85)
    ax.set_xlabel("ours TN cosine (polynomial)")
    ax.set_ylabel("src TN cosine (state propagation)")
    ax.set_title(f"{tag}: ours vs src, {n_seeds} seeds, {len(pairs)} pairs, r = {r:.3f}")
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal"); ax.legend(loc="upper left")
    fig.tight_layout()
    fig_path = out_dir / f"tn_vs_src_{tag}.png"
    fig.savefig(fig_path, dpi=150)
    print(f"[{tag}] Saved plot -> {fig_path}")
    np.savez(out_dir / f"results_src_{tag}.npz", ours=ours, srcs=srcs, corr=r)
    return {"ours": ours, "srcs": srcs, "corr": r}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "experiment",
        choices=["2L_res", "3L_nonres", "3L_res", "src_2L", "src_3L", "all"],
    )
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--mc_samples", type=int, default=500_000)
    args = ap.parse_args()

    kw = dict(n_seeds=args.seeds, mc_samples=args.mc_samples)
    if args.experiment in ("2L_res", "all"):
        run_2layer_residual(**kw)
    if args.experiment in ("3L_nonres", "all"):
        run_3layer_nonres(**kw)
    if args.experiment in ("3L_res", "all"):
        run_3layer_residual(**kw)
    if args.experiment in ("src_2L", "all"):
        run_src_correlation("src_2L_nonres", n_layers=2, n_seeds=args.seeds)
    if args.experiment in ("src_3L", "all"):
        run_src_correlation("src_3L_nonres", n_layers=3, n_seeds=args.seeds)
