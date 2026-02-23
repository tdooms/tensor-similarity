# Norm Sweep Experiment

Compare normalization strategies for bilinear attention transformers.

## Norm Variants

### `seq_max`
1. Compute per-token energy: `e_t = mean_d(x_{t,d}^2)` → `(B,T,1)`
2. Take max over sequence: `m = max_t(e_t)` → `(B,1,1)`
3. Scale: `x / sqrt(m + eps)`

### `causal_seq_max`
1. Compute per-token energy: `e_t = mean_d(x_{t,d}^2)` → `(B,T,1)`
2. Cumulative max: `m_t = cummax_{s≤t}(e_s)` → `(B,T,1)`
3. Scale: `x_t / sqrt(m_t + eps)` (token-specific)

### `stochastic_seq_max`
1. Compute energy: `e_t = mean_d(x_{t,d}^2)`
2. **Train**: 50% use `max_t(e_t)` (full seq), 50% use `cummax(e_t)` (causal)
3. **Eval**: always use `cummax(e_t)` (causal)

### `tok1`
1. Compute first-token energy: `e_0 = mean_d(x_{0,d}^2)` → `(B,1)`
2. Scale entire sequence: `x / sqrt(e_0 + eps)` → `(B,1,1)`

### `tok1_batch`
1. Compute first-token energy per sample: `e_b = mean_d(x_{b,0,d}^2)` → `(B,)`
2. Average across batch: `m_batch = mean_b(e_b)` → scalar
3. **Train**: scale by `1/sqrt(m_batch + eps)`, update running scalar: `m_run ← (1-β)*m_run + β*m_batch` with `β=0.1`
4. **Eval-as-train**: use `m_batch` (batch stats)
5. **Eval-as-inference**: use `m_run` (precomputed running stats)

### `seq_max_batch`
1. Compute per-token energy: `e_{b,t} = mean_d(x_{b,t,d}^2)` → `(B,T,1)`
2. Global max over batch+sequence: `m = max_{b,t}(e_{b,t})` → scalar
3. Scale entire batch: `x / sqrt(m + eps)`

### `tok1_ghost`
1. Compute first-token energy: `e_b = mean_d(x_{b,0,d}^2)` → `(B,)`
2. **Train**: sample ghost subset, compute ratio `r = mean(e_ghost) / mean(e_full)`, scale by `e_b * r`
3. **Eval**: deterministic `tok1` (no noise)

### `tok1_bn`
1. Extract first token: `x_0 = x[:,0,:]` → `(B,D)`
2. **Train**: compute batch mean/var per feature → `(D,)`, update running stats
3. **Eval**: use running mean/var
4. Normalize: `(x - mean) / sqrt(var + eps)`

### `tok1_bn_ghost`
1. Same as `tok1_bn` but with ghost noise on scale during training
2. Running stats updated with clean stats (no noise)
3. **Train**: perturb scale by `sqrt(var_ghost / var_full)` per feature
4. **Eval**: clean BN with running stats

### `seq_mean`
1. Compute per-token energy: `e_t = mean_d(x_{t,d}^2)` → `(B,T,1)`
2. Mean over sequence: `m = mean_t(e_t)` → `(B,1,1)`
3. **Train**: scale by `1/sqrt(m)`, update running mean energy
4. **Eval**: use running mean energy if available

---

## Set 2 (New Variants)

### `tok190`
1. Compute token-1 energy per sample: `e_b = mean_d(x_{b,0,d}^2)` → `(B,)`
2. Batch aggregate: `m_batch = Q0.90({e_b})` → scalar
3. **Train**: scale by `1/sqrt(m_batch + eps)`, update running scalar: `m_run ← (1-β)*m_run + β*m_batch` with `β=0.1`
4. **Eval-as-train**: use `m_batch` (batch stats)
5. **Eval-as-inference**: use `m_run` (precomputed running stats)

### `tok190_clamp`
1. Compute token-1 energy per sample: `e_b = mean_d(x_{b,0,d}^2)` → `(B,)`
2. Compute clamp bounds: `l = Q0.05(e)`, `u = Q0.95(e)` → scalars
3. Clamp energies: `e_tilde_b = clamp(e_b, l, u)` → `(B,)`
4. Batch aggregate: `m_batch = Q0.90({e_tilde_b})` → scalar
5. **Train**: scale by `1/sqrt(m_batch + eps)`, update running scalar: `m_run ← (1-β)*m_run + β*m_batch` with `β=0.1`
6. **Eval-as-train**: use `m_batch` (batch stats)
7. **Eval-as-inference**: use `m_run` (precomputed running stats)

### `seq_max_mean_batch`
1. Compute per-token energy: `e_{b,t} = mean_d(x_{b,t,d}^2)` → `(B,T)`
2. Per-sample sequence max: `m_b = max_t(e_{b,t})` → `(B,)`
3. Batch aggregate: `m_batch = mean_b(m_b)` → scalar
4. **Train**: scale by `1/sqrt(m_batch + eps)`, update running scalar: `m_run ← (1-β)*m_run + β*m_batch` with `β=0.1`
5. **Eval-as-train**: use `m_batch` (batch stats)
6. **Eval-as-inference**: use `m_run` (precomputed running stats)

### `seq_max_median_batch`
1. Compute per-token energy: `e_{b,t} = mean_d(x_{b,t,d}^2)` → `(B,T)`
2. Per-sample sequence max: `m_b = max_t(e_{b,t})` → `(B,)`
3. Batch aggregate: `m_batch = median_b(m_b)` → scalar
4. **Train**: scale by `1/sqrt(m_batch + eps)`, update running scalar: `m_run ← (1-β)*m_run + β*m_batch` with `β=0.1`
5. **Eval-as-train**: use `m_batch` (batch stats)
6. **Eval-as-inference**: use `m_run` (precomputed running stats)

### `seq_power_mean`
1. Compute per-token energy: `e_{b,t} = mean_d(x_{b,t,d}^2)` → `(B,T)`
2. Power-mean over time with `p=2`: `m_b = (1/T * sum_t(e_{b,t}^2))^{1/2}` → `(B,)`
3. Scale per sample: `x_{b,:,:} ← x_{b,:,:} / sqrt(m_b + eps)` (broadcast `m_b` over `T,D`)

### `seq_mean_batch`
1. Compute per-token energy: `e_{b,t} = mean_d(x_{b,t,d}^2)` → `(B,T)`
2. Batch mean energy: `m_batch = mean_{b,t}(e_{b,t})` → scalar
3. **Train**: scale by `1/sqrt(m_batch + eps)`, update running scalar: `m_run ← (1-β)*m_run + β*m_batch` with `β=0.1`
4. **Eval-as-train**: use `m_batch` (batch stats)
5. **Eval-as-inference**: use `m_run` (precomputed running stats)

### `seq_power_mean_batch`
1. Compute per-token energy: `e_{b,t} = mean_d(x_{b,t,d}^2)` → `(B,T)`
2. Power-mean over `b,t` with `p=2`: `m_batch = (1/(BT) * sum_{b,t}(e_{b,t}^2))^{1/2}` → scalar
3. **Train**: scale by `1/sqrt(m_batch + eps)`, update running scalar: `m_run ← (1-β)*m_run + β*m_batch` with `β=0.1`
4. **Eval-as-train**: use `m_batch` (batch stats)
5. **Eval-as-inference**: use `m_run` (precomputed running stats)

---

## Swap Experiments

- **seq_max → causal_seq_max**: Train with `seq_max`, swap to `causal_seq_max` at eval, compare CE.
- **seq_mean live vs running**: Train `seq_mean` with live batch stats, eval with precomputed running stats.

## Usage

From the `bilinear_attn` directory:

```bash
# Basic usage (auto-selects induction.yaml)
python -m experiments.norm_sweep.run --norm-type seq_max

# Stories dataset (auto-selects stories.yaml)
python -m experiments.norm_sweep.run --norm-type tok1 --dataset stories

# With wandb
python -m experiments.norm_sweep.run --norm-type seq_max --wandb

# Swap experiment: train seq_max, eval with causal_seq_max
python -m experiments.norm_sweep.run --norm-type seq_max --swap causal_seq_max

# Custom config
python -m experiments.norm_sweep.run --config path/to/custom.yaml --norm-type tok1
```

### Wandb Sweep

```bash
# Initialize sweep
wandb sweep experiments/norm_sweep/configs/sweep.yaml

# Run agent (use sweep ID from above)
wandb agent <entity>/<project>/<sweep_id>
wandb agent melwina-albuquerque-flame-university/bilinear-induction-heads/sweep_1
```

## Files

- `norms.py` — All 10 normalization `nn.Module` classes + registry
- `model.py` — `NormSweepLM` model wrapper (dispatches `norm_type` through registry)
- `run.py` — Unified train + eval + swap script with wandb support
- `configs/induction.yaml` — Config for induction dataset (auto-selected with `--dataset induction`)
- `configs/stories.yaml` — Config for stories dataset (auto-selected with `--dataset stories`)
- `configs/sweep.yaml` — Wandb sweep config (grid search over all norms × datasets)
