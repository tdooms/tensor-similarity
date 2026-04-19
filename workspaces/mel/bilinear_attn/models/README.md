````markdown
# Quadratic (Polynomial) Attention LM — Implementation Guide

This repo implements a small **autoregressive language model** with a non-softmax attention variant:
- **Scores:** dot product `q·k`
- **Pattern:** elementwise square of scaled scores
- **Mask:** causal (lower-triangular)
- **Position:** RoPE (rotary embeddings)
- **Head:** not using tied embeddings 
- **Depth:** `n_layers` is configurable

**Normalization (RMSNorm) is optional**:
- `use_rmsnorm_qk: false` → **TN-clean mode** (no input-dependent normalization)
- `use_rmsnorm_qk: true` → more stable training, but no longer strictly TN-clean

---

## What this model is (and is not)

### Autoregressive LM
Given token IDs `input_ids: (B, T)`, produce `logits: (B, T, V)` and train with next-token loss.

### Attention-only blocks
Each layer is **attention + residual** (no MLP in this repo’s core architecture).

---

## Exact architecture

### Shapes
- `V` = vocab size
- `T` = sequence length (≤ `n_ctx`)
- `d_model`
- `n_head`
- `d_head = d_model // n_head`

### Parameters
- Token embedding: `E ∈ R^{V × d_model}`
- Token unembedding `U ∈ R^{d_model x V}`
- Per-layer projections (all are standard Linear):
  - `Wq, Wk ∈ R^{d_model × d_model}` (+ biases optional via `use_bias_qk`)
  - `Wv, Wo ∈ R^{d_model × d_model}` (no bias)
- RoPE buffers (not learned): sin/cos cached up to `n_ctx`

### Forward pass
1) **Embed**
- `x = E[input_ids]`  → `(B, T, d_model)`

2) **Repeat for each layer ℓ = 1..n_layers**
- Project + split into heads:
  - `q, k, v = split_heads(Wq x + bq, Wk x + bk, Wv x + bv)` → `(B, T, n_head, d_head)`
- Optional RMSNorm on Q/K only:
  - if enabled: `q = RMSNorm(q)`, `k = RMSNorm(k)`
- Apply RoPE to Q and K:
  - `q = RoPE(q)`, `k = RoPE(k)`
- Scores:
  - `scores[b,h,t,s] = Σ_i q[b,t,h,i] * k[b,s,h,i]`
- Pattern (no softmax):
  - `pattern = (scores / d_head)^2`
  - causal mask: `pattern *= tril(ones(T,T))`
- Attend:
  - `z[b,t,h,i] = Σ_s pattern[b,h,t,s] * v[b,s,h,i]`
- Merge heads and residual update:
  - `z_merge = merge_heads(z)` → `(B, T, d_model)`
  - `x = x + attn_scale * (Wo z_merge + bo)`

3) **unembedding **
- `logits = x @ U.T`  → `(B, T, V)`

---

## Training endpoint contract

Your `train.py` should:
- Load YAML config
- Build model from config (`n_layers` variable)
- Import a loss function from another module

Expected interfaces:
- `logits = model(input_ids)`
- `loss = compute_loss(logits, input_ids, **loss_cfg)`

Example:
```python
from losses import compute_loss
````

---

## Initialization (recommended)

Default (simple, consistent):

* Embedding, Q/K/V: Normal(0, 0.02)
* Output projection `Wo`: smaller std (config-controlled)
* Biases: zeros

---

## Configs (store as YAML files)

Create these under `configs/`.

### 1) `configs/test32.yaml`

```yaml
name: test32
seed: 0

model:
  vocab_size: 32000
  n_ctx: 128
  d_model: 32
  n_head: 4
  n_layers: 2
  attn_scale: 0.2
  rope_base: 10000
  norm_type: rmsnorm
  norm_place: pre_unembed
  use_rmsnorm_qk: false
  use_bias_qk: true

init:
  std_embed: 0.02
  std_qkv: 0.02
  std_o: 0.01

train:
  batch_size: 16
  lr: 3.0e-4
  weight_decay: 0.1
  max_steps: 2000
  warmup_steps: 200
  grad_clip: 1.0
  amp: false

loss:
  type: next_token_ce
  label_smoothing: 0.0
```

### 2) `configs/test64.yaml`

```yaml
name: test64
seed: 0

model:
  vocab_size: 32000
  n_ctx: 256
  d_model: 64
  n_head: 4
  n_layers: 2
  attn_scale: 0.2
  rope_base: 10000
  norm_type: rmsnorm
  norm_place: pre_unembed
  use_rmsnorm_qk: false
  use_bias_qk: true

init:
  std_embed: 0.02
  std_qkv: 0.02
  std_o: 0.01

train:
  batch_size: 16
  lr: 3.0e-4
  weight_decay: 0.1
  max_steps: 3000
  warmup_steps: 300
  grad_clip: 1.0
  amp: false

loss:
  type: next_token_ce
  label_smoothing: 0.0
```

### 3) `configs/main256.yaml`

```yaml
name: main256
seed: 0

model:
  vocab_size: 32000
  n_ctx: 512
  d_model: 256
  n_head: 8
  n_layers: 2
  attn_scale: 0.1
  rope_base: 10000
  norm_type: rmsnorm
  norm_place: pre_unembed
  use_rmsnorm_qk: false
  use_bias_qk: true

init:
  std_embed: 0.02
  std_qkv: 0.02
  std_o: 0.007

train:
  batch_size: 32
  lr: 2.0e-4
  weight_decay: 0.1
  max_steps: 20000
  warmup_steps: 1000
  grad_clip: 1.0
  amp: true

loss:
  type: next_token_ce
  label_smoothing: 0.0
```

---

## Run

```bash/wsl
python train.py --config configs/test64.yaml
python train.py --config configs/main256.yaml
```

---

## Red-team notes (what to keep vs what to cut)

**Useful / necessary**

* The exact forward definition (what is squared, where mask applies, whether attention is normalized).
* Optional RMSNorm flag and what it implies (TN-clean vs stability).
* not using tied embeddings
* YAML configs (so runs are reproducible).

**Unnecessary / removed**

* KV-cache discussion (inference-only; not needed for training guide).
* Deep transformer “best practices” (you’re doing 2 layers; overkill here).
* Multiple alternate architectural branches (keeps the README actionable).



## example code

class Rotary(nn.Module):
    """A modern implementation of the rotary position encoding."""
    def __init__(self, dim: int, n_ctx: int, base: int = 10_000) -> None:
        super().__init__()

        freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        ctx = torch.arange(n_ctx).type_as(freq)
        freqs = torch.einsum("i,j->ij", ctx, freq)

        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

        self.register_buffer(
            "cos_cached",
            torch.cat([self.cos, self.cos], dim=-1)[None, :, None, :],
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            torch.cat([self.sin, self.sin], dim=-1)[None, :, None, :],
            persistent=False,
        )

    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        y = torch.cat((-b, a), dim=-1)
        return (x * self.cos_cached[:, : x.size(-3)]) + (y * self.sin_cached[:, : x.size(-3)])

    def network(self, mod, **kwargs):
        data = [
            [[[1, 0], [0, 1]], [[0, -1], [1, 0]]],
            [[[0, 1], [-1, 0]], [[1, 0], [0, 1]]],
        ]
        black = Tensor(
            torch.tensor(data, **kwargs),
            inds=[f"{mod}:iq", f"{mod}:ik", f"{mod}:2q", f"{mod}:2k"],
            tags=["#"],
        )

        emb = torch.stack([self.cos, self.sin], dim=-1)
        q_rot = Tensor(emb, inds=["out:t", f"{mod}:h", f"{mod}:iq"], tags=["E"])
        k_rot = Tensor(emb, inds=["in:s", f"{mod}:h", f"{mod}:ik"], tags=["E"])

        return black & q_rot & k_rot


class Mask(nn.Module):
    def __init__(self, n_ctx: int, kind: str) -> None:
        super().__init__()
        data = dict(
            causal=torch.tril(torch.ones(n_ctx, n_ctx)),
            none=torch.ones(n_ctx, n_ctx),
            diag=torch.eye(n_ctx, n_ctx),
        )
        self.register_buffer("mask", data[kind], persistent=False)

    def forward(self, x):
        return x * self.mask[None, None, : x.size(-2), : x.size(-1)]

    def network(self, inds=("out:t", "in:s"), tag="M"):
        return Tensor(self.mask.data, inds=list(inds), tags=[tag])


class Attention(Component):
    """
    Attention replacement using a quadratic scoring function.

    Forward uses:
      scores = <q_t, k_s>
      pattern = mask( (scores / d_head)^2 )    if softmax=False
      z = pattern @ v
      out = x + scale * o(z)

    TN notes:
      - We include the scalar (1/d_head^2) explicitly in the TN, since scores are squared.
      - RMSNorm is input-dependent. For exact numeric equivalence, the inverse RMS factors must be
        injected as explicit inputs and wired into the Q/K path. We expose those inputs in the TN,
        but wiring them fully is model-design-specific; you will connect them where you want.
    """
    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_ctx: int,
        mask: str,
        scale: int = 1,
        norm: bool = True,
        bias: bool = True,
        softmax: bool = False,
    ) -> None:
        super().__init__()
        self.d_head = d_model // n_head
        self.n_head = n_head
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.scale = scale
        self.softmax = softmax

        self.rotary = Rotary(self.d_head, n_ctx)
        self.norm = nn.RMSNorm([self.d_head]) if norm else nn.Identity()
        self.mask = Mask(n_ctx, mask)

        self.q = nn.Linear(d_model, d_model, bias=bias)
        self.k = nn.Linear(d_model, d_model, bias=bias)
        self.v = nn.Linear(d_model, d_model, bias=bias)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        q, k, v = [
            rearrange(op(x), "... (n_head d_head) -> ... n_head d_head", n_head=self.n_head)
            for op in (self.q, self.k, self.v)
        ]
        q, k = self.rotary(self.norm(q)), self.rotary(self.norm(k))

        scores = einsum(
            q,
            k,
            "... seq_q n_head d_head, ... seq_k n_head d_head -> ... n_head seq_q seq_k",
        )

        if self.softmax:
            m = torch.tril(torch.ones(x.size(-2), x.size(-2), device=scores.device))[None, None, :, :]
            pattern = (scores / self.d_head).masked_fill(m == 0, -torch.inf).softmax(dim=-1)
        else:
            pattern = self.mask((scores / self.d_head).square())

        z = einsum(
            pattern,
            v,
            "... n_head seq_q seq_k, ... seq_k n_head d_head -> ... seq_q n_head d_head",
        )
        z = rearrange(z, "... seq n_head d_head -> ... seq (n_head d_head)")
        return torch.lerp(x, self.o(z), self.scale)
```
```
