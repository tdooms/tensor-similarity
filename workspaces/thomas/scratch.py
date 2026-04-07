# %%
import numpy as np
import sparse
import quimb.tensor as qtn
import time

# %%
# === Helpers ===

def sparse_perm_matrix(n: int) -> sparse.COO:
    """Random n x n permutation matrix as sparse COO. nnz = n."""
    perm = np.random.permutation(n)
    coords = np.array([np.arange(n), perm])
    return sparse.COO(coords, np.ones(n), shape=(n, n))


def sparse_perm_tensor(shape: tuple[int, ...]) -> sparse.COO:
    """
    High-dimensional permutation tensor. Splits axes into two halves and
    encodes a permutation between the flattened multi-indices.

    E.g. shape=(6,6,6,6,6,6) -> maps (i,j,k) -> sigma(i,j,k) = (l,m,n).
    Total elements: 6^6 = 46656, but nnz = 6^3 = 216 (density ~0.5%).
    """
    ndim = len(shape)
    half = ndim // 2
    left_shape = shape[:half]
    right_shape = shape[half:]
    assert left_shape == right_shape, "left and right halves must match"

    flat_n = int(np.prod(left_shape))
    perm = np.random.permutation(flat_n)

    left_coords = np.array(np.unravel_index(np.arange(flat_n), left_shape))
    right_coords = np.array(np.unravel_index(perm, right_shape))
    coords = np.concatenate([left_coords, right_coords], axis=0)
    return sparse.COO(coords, np.ones(flat_n), shape=shape)


def sparse_diag_tensor(n: int, ndim: int) -> sparse.COO:
    """Generalized diagonal: T[i,i,...,i] = 1. nnz = n."""
    coords = np.tile(np.arange(n), (ndim, 1))
    return sparse.COO(coords, np.ones(n), shape=(n,) * ndim)


def result_info(r):
    """Extract info from a contraction result (could be Tensor or raw sparse)."""
    data = r.data if hasattr(r, "data") else r
    if isinstance(data, sparse.SparseArray):
        return type(data).__name__, data.nnz, data.todense()
    return type(data).__name__, None, data


def bench(fn, n_runs=10, warmup=2):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return np.median(times)


def compare(name, sparse_tn, dense_tn, n_runs=10):
    t_sparse = bench(sparse_tn.contract, n_runs=n_runs)
    t_dense = bench(dense_tn.contract, n_runs=n_runs)

    _, sp_nnz, sp_dense = result_info(sparse_tn.contract())
    _, _, dn_dense = result_info(dense_tn.contract())
    np.testing.assert_allclose(sp_dense, dn_dense, atol=1e-10)

    nnz_str = f", nnz={sp_nnz}" if sp_nnz is not None else ""
    speedup = t_dense / t_sparse if t_sparse > 0 else float("inf")
    print(f"  {name}:")
    print(f"    Sparse: {t_sparse*1000:8.2f}ms{nnz_str}")
    print(f"    Dense:  {t_dense*1000:8.2f}ms")
    print(f"    Speedup: {speedup:.1f}x   ✓ match")


# %%
# === 1. High-dimensional permutation tensors (6^6 and 8^8) ===
print("=" * 60)
print("1. HIGH-DIMENSIONAL PERMUTATION TENSORS")
print("=" * 60)

for shape in [(6,) * 6, (8,) * 8]:
    P1 = sparse_perm_tensor(shape)
    P2 = sparse_perm_tensor(shape)
    total = int(np.prod(shape))
    half = len(shape) // 2
    print(f"\n  Shape {shape}: {total:,} elements, nnz={P1.nnz}, density={P1.nnz/total:.4%}")

    left_inds = [f"a{i}" for i in range(half)]
    shared = [f"m{i}" for i in range(half)]
    right_inds = [f"b{i}" for i in range(half)]

    t1s = qtn.Tensor(P1, inds=left_inds + shared)
    t2s = qtn.Tensor(P2, inds=shared + right_inds)
    t1d = qtn.Tensor(P1.todense(), inds=left_inds + shared)
    t2d = qtn.Tensor(P2.todense(), inds=shared + right_inds)

    compare(
        f"Contract two {'^'.join(str(s) for s in shape[:1])}^{len(shape)} perm tensors",
        qtn.TensorNetwork([t1s, t2s]),
        qtn.TensorNetwork([t1d, t2d]),
    )


# %%
# === 2. Chain topology ===
print("\n" + "=" * 60)
print("2. CHAIN TOPOLOGY")
print("=" * 60)

n = 500
num_chain = 20
print(f"\n  Chain of {num_chain} permutation matrices ({n}x{n})")

chain_sp = [sparse_perm_matrix(n) for _ in range(num_chain)]
chain_dn = [p.todense() for p in chain_sp]
inds = [(f"i{k}", f"i{k+1}") for k in range(num_chain)]

compare(
    f"Chain of {num_chain} perms ({n}x{n})",
    qtn.TensorNetwork([qtn.Tensor(p, inds=ix) for p, ix in zip(chain_sp, inds)]),
    qtn.TensorNetwork([qtn.Tensor(p, inds=ix) for p, ix in zip(chain_dn, inds)]),
)


# %%
# === 3. Star / hub topology ===
print("\n" + "=" * 60)
print("3. STAR / HUB TOPOLOGY")
print("=" * 60)

n_star = 30
num_spokes = 6
print(f"\n  Hub (diagonal {n_star}^{num_spokes}) + {num_spokes} spoke permutations")
print(f"  Hub: {n_star**num_spokes:,} elements but nnz = {n_star}")

hub = sparse_diag_tensor(n_star, num_spokes)
hub_inds = [f"h{k}" for k in range(num_spokes)]

sp_tensors = [qtn.Tensor(hub, inds=hub_inds)]
for k in range(num_spokes):
    P = sparse_perm_matrix(n_star)
    sp_tensors.append(qtn.Tensor(P, inds=(f"h{k}", f"out{k}")))

sparse_tn = qtn.TensorNetwork(sp_tensors)
t_sparse = bench(sparse_tn.contract, n_runs=5)
_, sp_nnz, _ = result_info(sparse_tn.contract())
print(f"  Sparse (n={n_star}): {t_sparse*1000:.2f}ms, result_nnz={sp_nnz}")
print(f"  (Dense hub would be {n_star**num_spokes * 8 / 1e9:.1f} GB — impossible!)")

# Dense comparison at small n to verify correctness
n_small = 8
hub_sm = sparse_diag_tensor(n_small, num_spokes)
sp_sm = [qtn.Tensor(hub_sm, inds=hub_inds)]
dn_sm = [qtn.Tensor(hub_sm.todense(), inds=hub_inds)]
for k in range(num_spokes):
    P = sparse_perm_matrix(n_small)
    sp_sm.append(qtn.Tensor(P, inds=(f"h{k}", f"out{k}")))
    dn_sm.append(qtn.Tensor(P.todense(), inds=(f"h{k}", f"out{k}")))
compare(
    f"Star correctness check (n={n_small})",
    qtn.TensorNetwork(sp_sm),
    qtn.TensorNetwork(dn_sm),
)


# %%
# === 4. 2D Grid / lattice ===
print("\n" + "=" * 60)
print("4. 2D GRID / LATTICE TOPOLOGY")
print("=" * 60)

n = 20
rows, cols = 4, 4
print(f"\n  {rows}x{cols} grid, bond dim {n}, open boundaries")

sp_nodes, dn_nodes = [], []
for r in range(rows):
    for c in range(cols):
        node_inds = []
        if c < cols - 1:
            node_inds.append(f"h_{r}_{c}")
        if c > 0:
            node_inds.append(f"h_{r}_{c-1}")
        if r < rows - 1:
            node_inds.append(f"v_{r}_{c}")
        if r > 0:
            node_inds.append(f"v_{r-1}_{c}")
        if len(node_inds) < 2:
            node_inds.append(f"d_{r}_{c}")

        ndim = len(node_inds)
        P = sparse_perm_matrix(n) if ndim == 2 else sparse_diag_tensor(n, ndim)
        sp_nodes.append(qtn.Tensor(P, inds=node_inds, tags={f"N{r},{c}"}))
        dn_nodes.append(qtn.Tensor(P.todense(), inds=node_inds, tags={f"N{r},{c}"}))

sparse_tn = qtn.TensorNetwork(sp_nodes)
dense_tn = qtn.TensorNetwork(dn_nodes)
print(f"  {sparse_tn.num_tensors} tensors, {sparse_tn.num_indices} indices")
compare(f"4x4 grid (bond dim {n})", sparse_tn, dense_tn)


# %%
# === 5. Cycle (closed loop) -> scalar ===
print("\n" + "=" * 60)
print("5. CYCLE TOPOLOGY (closed loop -> scalar trace)")
print("=" * 60)

n = 500
num_cycle = 12
print(f"\n  Cycle of {num_cycle} permutation matrices ({n}x{n})")

cycle_sp = [sparse_perm_matrix(n) for _ in range(num_cycle)]
cycle_dn = [p.todense() for p in cycle_sp]
cycle_inds = [(f"i{k}", f"i{(k+1) % num_cycle}") for k in range(num_cycle)]

compare(
    f"Cycle of {num_cycle} perms ({n}x{n}) -> scalar",
    qtn.TensorNetwork([qtn.Tensor(p, inds=ix) for p, ix in zip(cycle_sp, cycle_inds)]),
    qtn.TensorNetwork([qtn.Tensor(p, inds=ix) for p, ix in zip(cycle_dn, cycle_inds)]),
)


# %%
# === 6. Random TN with high-dim perm tensors ===
print("\n" + "=" * 60)
print("6. RANDOM TN WITH HIGH-DIM PERM TENSORS")
print("=" * 60)

d = 6
num_tensors = 8
print(f"\n  {num_tensors} tensors of dim 4, bond dim {d}, randomly connected")

rng = np.random.default_rng(42)
tensor_inds = [[] for _ in range(num_tensors)]
free_slots = [(t, s) for t in range(num_tensors) for s in range(4)]
rng.shuffle(free_slots)

idx_counter = 0
for i in range(0, len(free_slots) - 1, 2):
    t1_idx, _ = free_slots[i]
    t2_idx, _ = free_slots[i + 1]
    if t1_idx == t2_idx:
        tensor_inds[t1_idx].append(f"f_{idx_counter}")
        idx_counter += 1
        tensor_inds[t2_idx].append(f"f_{idx_counter}")
        idx_counter += 1
    else:
        name = f"b_{idx_counter}"
        idx_counter += 1
        tensor_inds[t1_idx].append(name)
        tensor_inds[t2_idx].append(name)

sp_tensors, dn_tensors = [], []
for t in range(num_tensors):
    ndim = len(tensor_inds[t])
    shape = (d,) * ndim
    P = sparse_perm_tensor(shape) if ndim % 2 == 0 and ndim >= 2 else sparse_diag_tensor(d, ndim)
    sp_tensors.append(qtn.Tensor(P, inds=tensor_inds[t], tags={f"T{t}"}))
    dn_tensors.append(qtn.Tensor(P.todense(), inds=tensor_inds[t], tags={f"T{t}"}))

sparse_tn = qtn.TensorNetwork(sp_tensors)
dense_tn = qtn.TensorNetwork(dn_tensors)

for t in range(num_tensors):
    print(f"    T{t}: shape={sp_tensors[t].shape}, nnz={sp_tensors[t].data.nnz}")

compare(f"Random TN ({num_tensors} tensors, bond dim {d})", sparse_tn, dense_tn)


# %%
# === 7. Scaling test: sparse-only (dense would be impossible) ===
print("\n" + "=" * 60)
print("7. SCALING: SPARSE-ONLY (too large for dense)")
print("=" * 60)
print()

for shape in [(6,)*6, (8,)*6, (10,)*6, (8,)*8, (10,)*8, (20,)*6, (30,)*6]:
    P1 = sparse_perm_tensor(shape)
    P2 = sparse_perm_tensor(shape)
    half = len(shape) // 2

    inds1 = [f"a{i}" for i in range(half)] + [f"m{i}" for i in range(half)]
    inds2 = [f"m{i}" for i in range(half)] + [f"b{i}" for i in range(half)]

    tn = qtn.TensorNetwork([qtn.Tensor(P1, inds=inds1), qtn.Tensor(P2, inds=inds2)])

    total = int(np.prod(shape))
    t_med = bench(tn.contract, n_runs=5, warmup=1)
    _, nnz_out, _ = result_info(tn.contract())

    print(
        f"  {str(shape):30s}  total={total:>12,}  nnz={P1.nnz:>6,}  "
        f"density={P1.nnz/total:.6%}  time={t_med*1000:8.2f}ms  "
        f"result_nnz={nnz_out}"
    )

print("\n  30^6 = 729M elements, but contraction takes ~70ms because nnz = 27k")
print("  Dense would need ~5.4 GB just to store one tensor!")

# %%
