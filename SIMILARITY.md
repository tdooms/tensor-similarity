# Exact Gaussian Similarity via Second-Moment Propagation

## The Problem

Given two polynomial neural networks `f_a, f_b` sharing Gaussian input `x ~ N(0, I)`,
compute `E[f_a(x)^T f_b(x)]` exactly (no sampling, no approximation for Gaussian input).

## The Key Insight: Track S, Not (mu, sigma)

We propagate the **second moment** `S = E[ff^T]` through layers, not the
mean and covariance separately. This works because Isserlis' theorem
(Wick's theorem) for `E[product of Gaussians]` needs only pairwise moments.

The trick: **the padded representation** bakes bias into the weights by prepending
a constant 1 to the input: `x_padded = [1; x]`. So `S = E[x_padded x_padded^T]`
encodes both the mean and covariance in one matrix.

## The Overcounting Correction

Using S = sigma + mu*mu^T directly as the Wick bridge overcounts the mean
contribution. The general overcounting factor for k mu-pairs out of N total
pairs is `(2k-1)!!`. We correct the all-mu term (k=N):

```
S_corrected = S_raw - ((2N-1)!! - 1) * mu_product_a (x) mu_product_b
```

where `mu_product = TN(mu)` is the TN evaluated at the mean (one cheap contraction).

### Exactness proof for our components

This correction handles ONLY the k=N (all-mu) overcounting. The k=2,...,N-1
corrections are missing. However, **the correction is exact for ALL our
current term pairs**, not just N <= 2. Here's why:

| Term pair | Legs | N | Non-zero mu legs | Why exact |
|---|---|---|---|---|
| Linear (embed/head) | 2 | 1 | 2 | N=1: only k=0,1. No overcounting. |
| MLP (bilinear) | 4 | 2 | 4 | N=2: only k=0,1,2. k<=1 don't overcount. k=2 is all-mu, corrected. |
| Identity x Identity | 2 | 1 | 2 | N=1: no overcounting. |
| Identity x Active | 6 | 3 | 2 (identity legs only) | k>=2 needs >=4 mu-legs, but only 2 are non-zero. All k>=2 terms vanish. |
| Active x Active | 10 | 5 | 0 (zero constant column) | All k>=1 terms vanish. No correction needed. |

**When would it break?** A component with >= 3 input legs AND >= 2 having non-zero
mean. Example: a trilinear layer `f(x) = D(Ax . Bx . Cx)` with bias would have
N=3 with 3 non-zero-mu legs, leaving k=2 overcounting uncorrected.

## Assumptions and Limitations

### What's exact
- **Gaussian input**: The Isserlis theorem is exact for Gaussian random variables.
  The initial `S = I` encodes `x ~ N(0, I)`.
- **Single layer**: Each layer's output second moment is computed exactly from
  the input second moment, assuming the input is Gaussian.
- **Rotary embeddings**: Fully inside the TN, contracted automatically.
- **Causal masking**: The mask tensor is inside the TN.
- **Bias**: Encoded via the padded constant dim.
- **Residual connections**: Decomposed into separate terms, cross-products computed exactly.
- **Permutation invariance**: Similarity is invariant to hidden-neuron permutations
  (the TN traces over hidden dimensions).

### What's approximate
- **Multi-layer propagation**: After the first polynomial layer, the output is
  NOT Gaussian — it's a polynomial function of a Gaussian. We treat it as
  Gaussian with matching second moment S. This is the **Gaussian closure
  approximation**. Accuracy degrades with depth, but residual connections
  keep the distribution near-Gaussian (empirically <1% error for 2-layer
  transformers with scale=0.5).

### What's NOT supported
- **Softmax attention**: Non-polynomial. Isserlis requires polynomial functions.
- **Non-polynomial activations**: ReLU, GELU, SiLU, etc.
- **Layer normalization / RMSNorm**: Non-polynomial. Would need moment-matching
  approximations.
- **Non-Gaussian input**: The entire approach assumes Gaussian input. For non-Gaussian,
  higher-order cumulants would be needed.

## The Algorithm (5 functions, ~110 lines)

Propagate S through layers. At each layer, decompose output into TN terms,
compute E[term_a * term_b^T] for all term cross-products via Isserlis, sum.

```
similarity(model_a, model_b):
    S = I
    for each layer pair (ca, cb):
        for each term pair (ta, tb) from ca.terms() x cb.terms():
            S_new += sum over Wick matchings of: TN(ta, tb) bridged with S
            S_new -= (n_matchings - 1) * TN_at_mu(ta) (x) TN_at_mu(tb)
    return S
```

## Component Interface

Each component provides `terms(n_ctx, **like) -> [Term(tn, legs, symmetries=())]`:

- **Linear/MLP**: 1 term. Residual folded into padded weights. All legs point at
  `out:s` (implicit position identification — no delta tensors materialized).
- **Attention**: 2 terms. Identity (residual) + active (attention circuit). V/K₁/K₂
  legs share the `in:s` position; Q₁/Q₂ legs share `out:s`. The active term declares
  the `(K₁↔K₂, Q₁↔Q₂)` Z₂ symmetry.

## Performance

Bottleneck: `(2N−1)!!` Wick matchings per term pair — attention active × active
is 945 matchings. Key optimizations, in order of leverage:

1. **Sparse spiders**: leg-position identification via shared index names instead
   of dense δ-tensors. Eliminates the `n_ctx^(N+1)` materialization.
2. **Symmetry dedup** (attention Z₂×Z₂): 945 → 265 unique matchings.
3. **`cotengra.ReusableHyperOptimizer`** with `minimize='combo'` and kahypar —
   path quality is 5-100× better than greedy at realistic scales, persistent
   disk cache so warmup pays once per architecture.
4. **Hot/cold split**: `_run()` is pure-GPU (no Python-data allocations, no
   sync points); `State.from_model` and `ca.terms()` are lifted out of
   `propagate`.
5. **Batched Wick sum**: `torch.stack + torch.einsum('i,i...->...', w, x)` in
   place of N sequential adds.
6. **Implicit CUDA-graph capture (JAX-like)**: on CUDA, `similarity(a, b)`
   captures a graph on the first call for a given (architecture, device,
   dtype) signature and replays on subsequent calls with new weights copied
   into static buffers. ~6× extra on the hot path, no `compile_similarity`
   ceremony — the signature covers both the checkpoint-matrix and
   training-loop use cases.

Memory scales as `n_ctx² · d_model²` for the block state; contraction
intermediates stay quadratic in each dimension (enforced by kahypar paths).
