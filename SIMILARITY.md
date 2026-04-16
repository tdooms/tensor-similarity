# Exact Gaussian Similarity via Second-Moment Propagation

Two neural networks `f_a` and `f_b` trained on the same data usually disagree
in weight space even when they agree on outputs. Weight-space distances don't
tell you this. Sampling the output disagreement does, but noisily and slowly.
What we actually want is the *functional* inner product

```
⟨f_a, f_b⟩ := E_{x ~ N(0, I)} [ f_a(x)ᵀ f_b(x) ]
```

and — if the networks are polynomial enough — we can compute it exactly, with
zero samples, by tensor-network contraction. That is what `similarity(a, b)`
does. This document explains how, starting from the simplest possible framing
and adding a layer of machinery at each step.

## 1. The trick: propagate second moments, not samples

For Gaussian `x`, the second moment `S = E[x xᵀ]` is all you ever need to
compute expectations of polynomials of `x`. That's Isserlis' theorem (the
physicists call it Wick's theorem): any Gaussian moment factorizes into a
sum over perfect matchings of pairwise covariances.

Now suppose `f` is one polynomial layer — say, a linear map `f(x) = Wx` or a
bilinear MLP `f(x) = D(Ax ⊙ Bx)`. The output is a new polynomial in `x`, and
its second moment `E[f(x) f(x)ᵀ]` is again computable exactly from `S`. So we
can *propagate* `S` through the network: layer by layer, input moment in,
output moment out.

Two models `a` and `b` that share the same Gaussian input produce correlated
outputs. We track three blocks of the joint second moment:

```
s_aa = E[f_a fᵀ_a]    s_ab = E[f_a fᵀ_b]    s_bb = E[f_b fᵀ_b]
```

At the input, all three equal the identity (since `a` and `b` see the same
Gaussian). After each layer they diverge. At the output, the similarity we
report is essentially `tr(s_ab) / √(tr(s_aa) · tr(s_bb))` — a cosine of the
functional vectors.

Everything after this section is about computing the per-layer update exactly
and fast.

## 2. Biases and the padded representation

A single trick makes bias handling essentially free. Prepend a constant `1`
to every input: `x_padded = [1; x]`. A linear layer `Wx + b` becomes a single
matrix product `W' x_padded` with the bias folded into the first column of
`W'`. Every component in the library uses this padded representation, and
every second-moment tensor we carry is `(d+1) × (d+1)`.

This means `S` encodes both the covariance *and* the mean of the distribution
it represents: `S[:, :, 0, 0]` is the mean vector μ of the padded variable
(because the 0-th component is deterministically 1), and the rest is the
usual covariance. You'll see us slice `S[:, :, 0, 0]` later when a "mean
contribution" shows up. That's what it is.

## 3. A layer's output is a sum of terms

Not every layer is one polynomial. An attention block with a residual
connection is `x + Attn(x)` — a *sum* of two polynomials of different degrees.
We call each polynomial summand a **term**. Every component exposes its terms
via a `.terms()` method:

| Component | Terms |
|---|---|
| Linear | 1: `Wx` |
| Bilinear MLP | 1: `D(Ax ⊙ Bx)` |
| Attention (no residual) | 1: full attention circuit |
| Attention (with residual) | 2: `x` + the attention circuit |

The layer-to-layer update is then

```
new s_xy = Σ_{t_l ∈ layer_a.terms()}  Σ_{t_r ∈ layer_b.terms()}  E[t_l · t_rᵀ | S]
```

for each of the three blocks `(x, y) ∈ {(a,a), (a,b), (b,b)}`. The
cross-terms (residual × active) matter; dropping them is what makes the
naïve "separate residual and non-residual" approach wrong.

Each term exposes three things to the algorithm:

1. `term.tn` — the tensor network whose external legs are the places where
   input `x` plugs in.
2. `term.legs` — a `{data-index: position-index}` map describing those legs.
3. `term.symmetries` — permutations of the data-indices under which the
   term is invariant. (Linear/MLP have none; attention declares a `Z₂ × Z₂`
   swapping `(Q₁↔Q₂, K₁↔K₂)`.)

## 4. One term-pair expectation: Isserlis on a joint TN

Given `t_l` and `t_r`, compute `E[t_l · t_rᵀ]` as follows.

1. **Join the two term TNs.** Reindex `t_l.tn` with an `a:` prefix and
   `t_r.tn` with a `b:` prefix so their internal indices don't collide,
   then union them.
2. **Collect external legs.** Each external leg is a spot where an `x` is
   "still exposed" in the joint TN. Altogether there are `N_l + N_r` legs,
   where `N_l` and `N_r` are each term's leg count.
3. **Sum over all perfect matchings.** For every pairing of those legs into
   `N/2` pairs, attach an `S`-tensor as a "bridge" between the two legs of
   each pair, and contract. Sum the results. That's the Wick sum.

When a pair crosses sides (one leg from `a:`, one from `b:`), the bridge is
`s_ab`. Pairs within the same side use the same-side block. Which block each
leg belongs to is tracked per-leg as a `model_id`.

For attention's active×active term pair, `N = 10` legs → `(2·5−1)!! = 945`
matchings. The `Z₂ × Z₂` attention symmetry cuts this to **265 orbits**, and
we only compute one representative per orbit, weighted by the orbit size.

## 5. The padded-representation correction

There is one subtle thing. The Wick sum `Σ Π S_{ij}` formally treats every
`x`-coordinate as Gaussian, but in the padded rep the 0-th coordinate is
**deterministic** (always 1). For a truly Gaussian variable with mean μ,
`E[x²ⁿ] = (2n−1)!!·μ²ⁿ + …` (moments grow combinatorially). For a
deterministic 1, `E[1²ⁿ] = 1`. The Wick sum overcounts whenever matchings
route entirely through the mean component.

Concretely: the "all-μ" configuration — treat every leg as being evaluated
at its mean — is formally counted `(2N−1)!!` times by the Wick sum. We want
it counted once. So we subtract `((2N−1)!! − 1)` copies of the all-μ
contraction.

This is the *only* correction needed for our current components, for a
reason worth stating: the missing higher-order corrections all involve
products of at least two non-zero means, and our active attention term has
only two legs with non-zero mean (the identity-term factor, while the
attention score factor has zero mean by construction). For a hypothetical
trilinear layer with bias on all inputs, higher corrections would kick in.

In the code, the correction is not a special case. It's just one more
"configuration" in the sum, where each leg is self-paired (contracted with
μ alone rather than bridged to another leg). The weight is `−((2N−1)!!−1)`.
Every term-pair's moment is a single weighted sum over `265 + 1` (or
however many) configurations.

## 6. What the code looks like

`_moment(t_l, t_r, ml, mr, s)` is five lines:

```python
tn, legs, syms = _join(t_l, t_r, ml, mr)
configs, weights = _isserlis_plan(legs, syms, s.s_aa.device, s.s_aa.dtype)
contribs = torch.stack([bridged_contract(tn, _bridges_for(c, legs, s), _OUT)
                        for c in configs])
return torch.einsum('i,i...->...', weights, contribs)
```

`_join` prefixes, tags, and unions. `_isserlis_plan` enumerates matchings,
dedupes under symmetries, and appends the μ-correction configuration.
`_bridges_for` turns a configuration into a list of `(S-block, index pair)`
tuples. `bridged_contract` is the heavy lifter: it compiles the contraction
path once per unique topology (via `cotengra`) and caches the compiled
closure, so subsequent calls with the same bridge pattern but fresh tensors
skip the path search entirely.

The outer loop — layers and blocks — is another fifteen lines:

```python
def _run(model_a, model_b):
    comps = model_a.components()
    n = …                   # sequence length
    d = …                   # padded data dim
    s = State(I⊗I, I⊗I, I⊗I)
    for ca, cb in zip(comps, model_b.components()):
        ta = ca.terms(n, …)
        tb = cb.terms(n, …)
        block = lambda tl, tr, ml, mr: sum(
            _moment(x, y, ml, mr, s) for x in tl for y in tr)
        s = State(block(ta, ta, 0, 0),
                  block(ta, tb, 0, 1),
                  block(tb, tb, 1, 1))
    return s
```

That's the entire algorithm, modulo performance.

## 7. Performance: paths, caches, CUDA graphs

Three optimizations matter, in order of leverage.

**(a) Contraction paths.** The `(d+1)² · n_ctx²` working-set tensors mean
naïve path-finding gives catastrophically large intermediates. We use
`cotengra.ReusableHyperOptimizer` with `minimize='combo'` and `kahypar` as
a method — this partitions the TN aggressively, keeps intermediates
quadratic in each dimension, and persists paths to disk so warmup pays
once per architecture. Without this we OOM at `d=256`.

**(b) Compiled-expression caching.** `quimb.TensorNetwork.contract(...,
get='expression')` returns a closure that takes tensor data and runs the
path. We cache one such closure per unique bridge topology (keyed by the
concatenated index tuples of the core + bridges). A similarity call at
scale executes hundreds of these closures; compiling each from scratch
would dominate. Caching is in `utils.bridged_contract`.

**(c) Hot/cold split + CUDA graph capture.** The entire algorithm is
deterministic in shapes once `(architecture, device, dtype)` is fixed.
So on CUDA, the first call to `similarity(a, b)` enters `capture_cuda_graph`:

1. Allocate static tensor buffers mirroring every parameter of `a` and `b`.
2. Rebind each `model.parameter().data` to the static buffer.
3. Warm up on a side stream — three full runs of `_run(a, b)`. This
   populates every downstream cache (matchings, contraction paths, compiled
   expressions, weight tensors), ensuring the captured region is
   allocation-and-sync-free.
4. `with torch.cuda.graph(g):` run `_run(a, b)` once more. This records
   the exact GPU-side work as a `CUDAGraph`.
5. Restore original `.data` pointers.

Subsequent calls with the same signature copy new parameter values into
the static buffers and `graph.replay()`. On a single-layer transformer at
`d_model=16, n_ctx=3, fp32`, this goes from ~5 s cold to ~100 ms warm,
with the CPU/GPU numerical agreement at ~1e-8 (matching fp32 precision).

The graph cache is keyed by `(tuple of parameter shapes, device, dtype)`,
so comparing a *list* of same-architecture checkpoints — the "training
trajectory" use case — captures once and replays per pair.

## 8. What's exact, what's approximate, what's not supported

### Exact

- **Single-layer** Gaussian moment propagation — assuming the input to that
  layer is Gaussian, the output second moment is computed exactly.
- **Biases** via the padded representation.
- **Residual connections** via term decomposition with cross-term products.
- **Permutation invariance** of hidden neurons — the TN traces over hidden
  dimensions so two models differing only in hidden-neuron ordering give
  identical similarity.
- **Rotary embeddings and causal masks** — both are fully inside the term
  TNs and contracted automatically.

### Approximate

- **Multi-layer**: after the first polynomial layer, the intermediate
  activations are no longer Gaussian — they're polynomials of Gaussians.
  We keep propagating `S` as if they were Gaussian with the correct second
  moment. This is *Gaussian closure*. Empirically, residual connections
  keep things near-Gaussian; we see <1 % error on two-layer transformers
  with residual scale 0.5.

### Not supported

- **Softmax attention** — non-polynomial, Isserlis doesn't apply. Our
  attention component uses a quadratic scoring function instead.
- **Non-polynomial activations** (ReLU, GELU, SiLU, …) — same reason.
- **Layer/RMS normalization** — non-polynomial. Would need moment-matching
  approximations.
- **Non-Gaussian input** — higher-order cumulants would be needed.

## 9. Where this sits

The algorithm is ~110 lines in `src/components/similarity.py`, backed by
~100 lines of library-grade utilities in `src/components/utils.py`:
`bridged_contract`, `capture_cuda_graph`, `matchings`, `orbits`, `prefix`,
`direct_product`. None of the utilities know anything about similarity —
each one is something we wished were already in quimb, PyTorch, or a
group-theory library.

The remaining complexity is genuinely intrinsic to the problem:
multi-term layer decomposition, block-structured second-moment tracking,
and the `(2N−1)!!` Wick-matching enumeration at each term pair. Every
line we can cut has been cut; what's left is the math.
