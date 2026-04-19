# `tn_sim` — TN similarity adapter for `AttentionLM`

This module is a **thin adapter** over the main codebase's exact Gaussian
functional similarity (`src/components/similarity.py`). It does **not**
reimplement the algorithm; it only wraps the mel `AttentionLM` so that the
upstream `similarity()` can consume it through its `Model.components()`
interface.

## Layout

```
tn_sim/
  similarity.py      # Wrappers: compute_tn_similarity / cosine_similarity / ...
  mc_similarity.py   # Two MC baselines (see below)
  benchmark.py       # Runnable time/memory benchmark
  config/            # Minimal TN-compatible YAML
  tests/             # pytest suite: self-sim, parity, MC, validation
../models/components/ # Component/Model wrappers (embedding, attention, model)
```

Ground truth lives under `src/components/`. The adapter's `terms()` is a
direct mirror of `src/components/attention.Attention.terms()`. mel's
forward uses the same `lerp(x, o(z), scale)` residual convention, so
wrapping mel and building the same architecture from `src.components`
primitives are interchangeable.

## Usage

```python
from models import AttentionLM
from tn_sim import cosine_similarity

model_a = AttentionLM.from_config(cfg)
model_b = AttentionLM.from_config(cfg)

sim = cosine_similarity(model_a, model_b, dtype=torch.float64)
```

API:

- `cosine_similarity(a, b)` → scalar in `[-1, 1]`
- `compute_tn_similarity(a, b)` → full `State(s_aa, s_ab, s_bb)`
- `inner_product(a, b)` → unnormalised trace
- `self_similarity(m)` → should be exactly `1.0`
- `mc_similarity_gaussian_tokens(a, b, ...)` — **matched MC** (samples at the
  TN algorithm's input distribution: Gaussian over padded vocab axis)
- `mc_similarity(a, b, ...)` — **residual-stream MC** (samples Gaussians
  *post-embed*; a different baseline that does not converge to the TN value
  for non-trivial embeddings)
- `random_sim(a, b, ...)` — uniform discrete token MC (yet another baseline)

## Model requirements

TN similarity is exact only for polynomial, Gaussian-friendly networks. The
wrapper rejects incompatible configs:

- `norm_type = "none"` and `norm_places = []`
- `use_rmsnorm_qk = False`
- `attn_type ∈ {"bilinear", "quadratic"}` (no softmax)

## Tests

```bash
pytest tn_sim/tests -v
```

25 tests, all passing on CPU/float64:

- **Self-similarity** (1/2 layer, bilinear/quadratic, bias on/off) → `1.0` within `1e-6`.
- **Symmetry** `sim(A,B) == sim(B,A)` within `1e-10`.
- **Adapter-vs-src parity** (`test_components_parity.py`): wrapping mel and
  building an equivalent model out of `src.components` primitives yields
  element-wise identical `(s_aa, s_ab, s_bb)` (atol `1e-10`). The reference
  uses a tiny `_ResAddAttention` subclass so it matches mel's residual
  convention.
- **Forward parity**: contracting the adapter TN with an explicit input
  reproduces `mel.forward(x)` exactly.
- **MC parity** (using `mc_similarity_gaussian_tokens`, the matched
  baseline): TN and MC agree within `~10%` at `n=20k` samples.
- **Config validation**: rejects rmsnorm, `use_rmsnorm_qk`, softmax, and
  architecture mismatch.

## Benchmark

```bash
python -m tn_sim.benchmark                     # CPU, float64
python -m tn_sim.benchmark --device cuda --dtype float32
```

Measured on CPU / float64 (with a fresh ctg-path cache the first row is
cold; subsequent rows reuse cached expression plans where topology
overlaps):

| config                                | cold (s) | warm (s) | peak RSS |
| ------------------------------------- | -------- | -------- | -------- |
| `tiny-1L`  V=8, ctx=4, D=8, 1 layer   |    ~20   |    ~1.3  |   25 MB  |
| `tiny-2L`  V=8, ctx=4, D=8, 2 layers  |    ~2.7  |    ~2.8  |  <1 MB*  |
| `small-1L` V=16, ctx=8, D=16, 1 layer |    ~4.2  |    ~4.0  |  170 MB  |

Notes on the numbers:

- **Cold** is dominated by cotengra path-search; results are cached in
  `~/.cache/tensor-mars/ctg-paths/` and amortise across calls.
- **Warm** is the steady-state pair-call time. For heatmap workloads over
  many checkpoints this is what matters.
- **Peak RSS** is a psutil start-vs-end delta, so the 2-layer tiny row
  reads low because previous runs already allocated the quimb/torch pools
  and the delta captures only incremental growth. For absolute memory,
  prefer CUDA (`torch.cuda.max_memory_allocated`).
- Complexity grows quickly with `n_layers` (term decomposition is `2^L`)
  and `n_ctx` (sequence indices appear throughout the contractions). The
  adapter targets **small** models; for anything beyond `d_model=32,
  n_ctx=16, L=2` use `mc_similarity_gaussian_tokens` instead.

No batched-pairs API at this level. If you need one, add it upstream in
`src/components/similarity.py` rather than re-doing Isserlis here.
