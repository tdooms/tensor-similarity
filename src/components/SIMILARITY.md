# Exact Gaussian Similarity via Second-Moment Propagation

## The Problem

Given two polynomial neural networks `f_a, f_b` sharing Gaussian input `x ~ N(0, I)`,
compute `E[f_a(x)^T f_b(x)]` exactly (no sampling, no approximation for Gaussian input).

## The Key Insight: Track S, Not (mu, sigma)

We propagate the **second moment** `S = E[ff^T]` through layers, not the
mean and covariance separately. This works because Isserlis' theorem
(Wick's theorem) for `E[product of Gaussians]` needs only pairwise moments.

The trick that makes this work: **the padded representation** bakes bias into the
weights by prepending a constant 1 to the input: `x_padded = [1; x]`. This means
`S = E[x_padded x_padded^T]` encodes both the mean and covariance:

```
S = [[1,    mu^T  ],    = sigma + mu mu^T
     [mu,   sigma + mu mu^T]]
```

So a single matrix S carries all the information. Each layer reads S, computes
the new second moment via Isserlis, and outputs the new S.

## The Overcounting Correction

Using S directly in Wick's theorem overcounts the mean contribution.
Each of the `(2N-1)!!` matchings independently contributes a `mu^(2N)` term,
but the correct answer has this term only once. The correction:

```
S_corrected = S_raw - ((2N-1)!! - 1) * mu_product_a (x) mu_product_b
```

where `mu_product = TN(mu)` is the network evaluated at the mean (one cheap
contraction). For zero-mean components (attention active path), the correction
is zero. For Linear (1 matching), the correction is zero. Only MLP (3 matchings,
non-zero mean) needs the `-2 * mu (x) mu` correction.

**Why this works for N <= 2**: For 4 legs, the overcounting is ONLY in the
all-mu partition (verified by expanding all terms). For 6+ legs with non-zero
mean, higher-order corrections exist but our attention legs are zero-mean so
they don't arise.

## The Algorithm (5 functions, 109 lines)

### `similarity(model_a, model_b) -> State`
Entry point. Initializes `S = I` and propagates through each layer pair.

### `propagate(state, comp_a, comp_b) -> State`
One layer. Gets `terms()` from each component, computes second moments
for all term cross-products (aa, bb, ab), returns new State.

### `_second_moment(term_a, term_b, state) -> Tensor`
Doubles the TN (prefix 'a:'/'b:'), builds legs with model tags,
calls `_isserlis` with S/mu dispatchers.

### `_isserlis(tn, legs, S_for_pair, mu_for_leg, output_inds) -> Tensor`
The core: sum over Wick matchings with S bridges, minus the mean correction.
Each matching inserts bridge tensors into the TN and contracts via cotengra.

### `_contract(tn, bridges, output_inds) -> Tensor`
Contracts a TN with bridge tensors. Caches the contraction expression
(cotengra `Contractor`) by index structure for reuse.

## Component Interface

Each component provides `terms(n_ctx, **like) -> [Term(tn, legs)]`:

- **Linear/MLP**: 1 term. The TN includes residual via padded weights.
  Spider ties per-leg sequence positions to output position.
- **Attention**: 2 terms. Identity (residual) + active (attention circuit).
  Active TN has internal sequence structure (mask, rotary). Each input leg
  gets a unique position index tied to the TN's internal `in:s`/`out:s` via
  delta/spider tensors.

The `terms()` decomposition is what makes attention exact: instead of
approximating `(1-s)x + s*attn(x)` as a single polynomial, we decompose
into two terms and compute all cross-products:
`E[f f^T] = E[(t1+t2)(t1+t2)^T] = E[t1 t1^T] + E[t1 t2^T] + E[t2 t1^T] + E[t2 t2^T]`

## The S Layout: (s, d, s, d)

S has shape `(n_ctx, d+1, n_ctx, d+1)` with meaning `S[s1, d1, s2, d2] = E[f[s1,d1] * f[s2,d2]]`.

This matches the TN's natural output order `(a:out:s, a:out:d, b:out:s, b:out:d)`,
so no permutation is ever needed. The bridge tensor is just S with the right index names.

Extracting the mean: `mu[s, d] = S[s, d, 0, 0]` (contract with position 0, constant dim 0).

## Performance

The bottleneck is the number of Wick matchings: `(2N-1)!!` where N is the number
of input leg pairs. For attention (10 legs): 945 matchings. Each matching requires
a 37-tensor contraction via cotengra (~0.5ms with cached expression).

Caching contraction expressions (`_EXPR_CACHE`) is critical: each unique index
structure gets its contraction path computed once. Subsequent calls reuse the
compiled `Contractor`, skipping path-finding entirely.

Total wall time: ~20s for 16 tests covering Linear, MLP, Attention, and
1/2-layer Transformers with rotary embeddings.
