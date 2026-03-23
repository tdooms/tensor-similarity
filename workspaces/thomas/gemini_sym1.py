# %%
import torch
import torch.nn as nn
import quimb.tensor as qtn
import sparse
import numpy as np


# %%

def wick(n):
    """Canonical sparse Wick tensor: (2n-1)!! involutions in (2n,)^{2n}."""
    pm = lambda s: [()] if not s else [
        ((s[0], s[j]),) + m
        for j in range(1, len(s))
        for m in pm(s[1:j] + s[j+1:])
    ]
    ms = torch.tensor(pm(list(range(2 * n))))  # ((2n-1)!!, n, 2)

    inv = torch.zeros(len(ms), 2 * n, dtype=torch.long)
    inv.scatter_(1, ms[..., 0], ms[..., 1])
    inv.scatter_(1, ms[..., 1], ms[..., 0])

    return qtn.Tensor(
        torch.sparse_coo_tensor(inv.T, torch.ones(len(ms)), (2 * n,) * (2 * n)),
        inds=[f"i{k}" for k in range(2 * n)],
    )

wick(n=5).data.nnz
# %%
class LinearAttention(nn.Module):
    def __init__(self, d_model: int, n_ctx: int):
        super().__init__()
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.mask = nn.Buffer(torch.tril(torch.ones(n_ctx, n_ctx)), persistent=False)

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        scores = torch.einsum("bsd, btd -> bst", q, k) * self.mask
        return torch.einsum("bst, btd -> bsd", scores, v)



MATCHINGS_6 = pm(list(range(6)))





LEG_TO_SEQ = {0: "s", 1: "t", 2: "t", 3: "s", 4: "u", 5: "u"}


# %%
# --- Monte Carlo baseline ---
def empirical_similarity(net_a, net_b, n_samples=50_000):
    with torch.no_grad():
        n_ctx = net_a.mask.shape[0]
        d_model = net_a.q.weight.shape[0]
        x = torch.randn(n_samples, n_ctx, d_model)
        return torch.einsum("bsd, bsd -> b", net_a(x), net_b(x)).mean().item()


# %%
# --- Exact similarity via Wick's theorem (einsum per matching) ---
def exact_similarity(net_a, net_b):
    """Exact E[<f_A(x), f_B(x)>] by summing over 15 Wick matchings."""
    B0 = net_a.q.weight.T @ net_a.k.weight
    B1 = net_b.q.weight.T @ net_b.k.weight
    B2 = net_a.v.weight.T @ net_b.v.weight

    total = 0.0
    for matching in MATCHINGS_6:
        # Feature part: assign shared einsum chars to paired legs
        chars = {}
        for pair_idx, (u, v) in enumerate(matching):
            chars[u] = chars[v] = chr(97 + pair_idx)

        f_str = f"{chars[0]}{chars[1]},{chars[3]}{chars[4]},{chars[2]}{chars[5]}"
        feature_val = torch.einsum(f_str, B0, B1, B2)

        # Sequence part: pairing legs forces their positions to be equal
        seq = dict(LEG_TO_SEQ)
        for u, v in matching:
            old, new = seq[v], seq[u]
            if old != new:
                for k in seq:
                    if seq[k] == old:
                        seq[k] = new

        s_str = f"{seq[0]}{seq[1]},{seq[3]}{seq[4]}->"
        seq_val = torch.einsum(s_str, net_a.mask, net_b.mask)

        total += (feature_val * seq_val).item()

    return total


# %%
# --- Sparse Wick tensor similarity ---
#
# Build a sparse (d,d,d,d,d,d) tensor encoding ALL 15 matchings at once,
# weighted by their sequence factors. Then contract with B0, B1, B2 via quimb.
#
# Each matching M contributes d^3 nonzeros (one per value of the 3 free dims).
# For d=16: nnz <= 15*4096 = 61440 out of 16^6 ~ 16.8M (density ~0.37%).

def sparse_wick_similarity(net_a, net_b):
    B0 = (net_a.q.weight.T @ net_a.k.weight).detach().numpy()
    B1 = (net_b.q.weight.T @ net_b.k.weight).detach().numpy()
    B2 = (net_a.v.weight.T @ net_b.v.weight).detach().numpy()
    d = B0.shape[0]

    # Precompute the sequence factor for each matching
    seq_factors = []
    for matching in MATCHINGS_6:
        seq = dict(LEG_TO_SEQ)
        for u, v in matching:
            old, new = seq[v], seq[u]
            if old != new:
                for k in seq:
                    if seq[k] == old:
                        seq[k] = new
        s_str = f"{seq[0]}{seq[1]},{seq[3]}{seq[4]}->"
        seq_factors.append(torch.einsum(s_str, net_a.mask, net_b.mask).item())

    # Build the sparse feature Wick tensor: shape (d,d,d,d,d,d)
    # For each matching, fill in coordinates where paired legs share the same value.
    # Weight by the sequence factor so the contraction gives the full similarity.
    all_coords = []
    all_data = []
    vals = np.arange(d)oh also, 

    for m_idx, matching in enumerate(MATCHINGS_6):
        # 3 free dimensions (one per pair), grid them
        grids = np.meshgrid(vals, vals, vals, indexing="ij")
        flat = [g.ravel() for g in grids]

        coord = np.zeros((6, d**3), dtype=np.intp)
        for pair_idx, (u, v) in enumerate(matching):
            coord[u] = flat[pair_idx]
            coord[v] = flat[pair_idx]

        all_coords.append(coord)
        all_data.append(np.full(d**3, seq_factors[m_idx]))

    wick = sparse.COO(
        np.concatenate(all_coords, axis=1),
        np.concatenate(all_data),
        shape=(d,) * 6,
    )

    print(f"  Wick tensor: shape={wick.shape}, nnz={wick.nnz}, density={wick.nnz/wick.size:.4%}")

    # Contract: Wick(f0,f1,f2,f3,f4,f5) with B0(f0,f1), B1(f3,f4), B2(f2,f5) -> scalar
    TW = qtn.Tensor(wick, inds=("f0", "f1", "f2", "f3", "f4", "f5"), tags=["WICK"])
    TB0 = qtn.Tensor(B0, inds=("f0", "f1"), tags=["B0"])
    TB1 = qtn.Tensor(B1, inds=("f3", "f4"), tags=["B1"])
    TB2 = qtn.Tensor(B2, inds=("f2", "f5"), tags=["B2"])

    tn = qtn.TensorNetwork([TW, TB0, TB1, TB2])
    return float(tn.contract())


# %%
# --- Test ---
torch.manual_seed(42)

D_MODEL = 16
N_CTX = 8

net_a = LinearAttention(D_MODEL, N_CTX)
net_b = LinearAttention(D_MODEL, N_CTX)

print("Exact (einsum)...")
exact_val = exact_similarity(net_a, net_b)

print("Sparse Wick tensor...")
sparse_val = sparse_wick_similarity(net_a, net_b)

print("Monte Carlo...")
mc_val = empirical_similarity(net_a, net_b)

print(f"\n{'='*40}")
print(f"Exact (einsum):  {exact_val:,.2f}")
print(f"Sparse (Wick):   {sparse_val:,.2f}")
print(f"Monte Carlo:     {mc_val:,.2f}")
print(f"{'='*40}")

assert abs(exact_val - sparse_val) < 1e-4, f"Exact vs sparse mismatch: {exact_val} vs {sparse_val}"
rel_err = abs(exact_val - mc_val) / abs(exact_val)
assert rel_err < 0.05, f"MC relative error too large: {rel_err:.2%}"
print(f"All match! (MC relative error: {rel_err:.2%})")

# %%
