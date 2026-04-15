"""Normalization modules for the norm sweep experiment.

Concise names:
    seq_max           – MaxRMSNorm: max per-token energy over full sequence
    causal_seq_max    – CausalMaxRMSNorm: cumulative max energy (causal)
    stochastic_seq_max– 50/50 branch between seq_max and causal_seq_max per forward
    tok1              – Normalize entire sequence by RMS of token t=0 (first token)
    tok1_batch        – tok1 but RMS of first token averaged across batch
    seq_max_batch     – seq_max but max energy taken across batch too
    tok1_ghost        – tok1 + ghost noise (subset-batch vs full-batch scale ratio)
    tok1_bn           – Token-1 batch norm with running stats for eval
    tok1_bn_ghost     – tok1_bn + ghost noise perturbation (training only)
    seq_mean          – Normalize by mean energy across sequence; supports running stats

    # Set 2 (new)
    tok190            – tok1 but batch aggregate via Q0.90 quantile
    tok190_clamp      – tok190 with clamped energies (Q0.05, Q0.95)
    seq_max_mean_batch– seq_max per sample, then mean across batch
    seq_max_median_batch – seq_max per sample, then median across batch
    seq_power_mean    – power-mean (p=2) over sequence per sample
    seq_mean_batch    – mean energy over batch+time, BN-style running scalar at eval
    seq_power_mean_batch – power-mean (p=2) over batch+time, BN-style running scalar
"""

import torch
import torch.nn as nn


class SeqMax(nn.Module):
    """MaxRMSNorm: scale = 1/sqrt(max_t mean_d(x_{t,d}^2) + eps)."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        energy = x.pow(2).mean(dim=-1, keepdim=True)         # (B, T, 1)
        max_energy = energy.max(dim=-2, keepdim=True).values  # (B, 1, 1)
        scale = (max_energy + self.eps).rsqrt()               # (B, 1, 1)
        return x * scale


class CausalSeqMax(nn.Module):
    """CausalMaxRMSNorm: scale_t = 1/sqrt(cummax_{s<=t} energy_s + eps)."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        energy = x.pow(2).mean(dim=-1, keepdim=True)          # (B, T, 1)
        causal_max = energy.cummax(dim=-2).values              # (B, T, 1)
        scale = (causal_max + self.eps).rsqrt()                # (B, T, 1)
        return x * scale


class StochasticSeqMax(nn.Module):
    """Training: 50/50 branch between SeqMax and CausalSeqMax. Eval: CausalSeqMax."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        energy = x.pow(2).mean(dim=-1, keepdim=True)  # (B, T, 1)

        if self.training and torch.rand(1).item() < 0.5:
            # full-sequence max path
            max_energy = energy.max(dim=-2, keepdim=True).values  # (B, 1, 1)
            scale = (max_energy + self.eps).rsqrt()
        else:
            # causal path (also used at eval)
            causal_max = energy.cummax(dim=-2).values  # (B, T, 1)
            scale = (causal_max + self.eps).rsqrt()

        return x * scale


class Tok1(nn.Module):
    """Normalize entire sequence by RMS of the first token (t=0)."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        energy_t0 = x[:, 0, :].pow(2).mean(dim=-1, keepdim=True)  # (B, 1)
        scale = (energy_t0.unsqueeze(1) + self.eps).rsqrt()        # (B, 1, 1)
        return x * scale


class Tok1Batch(nn.Module):
    """Normalize by RMS of first token averaged across the batch.

    Supports dual eval modes:
        - use_running_stats=False: use batch stats at eval (eval-as-train)
        - use_running_stats=True: use running stats at eval (eval-as-inference)
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        momentum: float = 0.1,
        use_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.momentum = momentum
        self.use_running_stats = use_running_stats

        self.register_buffer("running_mean_energy", torch.ones(1))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        energy_t0 = x[:, 0, :].pow(2).mean(dim=-1)  # (B,)
        m_batch = energy_t0.mean()  # scalar

        if self.training:
            scale = (m_batch + self.eps).rsqrt()
            # update running stats
            self.num_batches_tracked += 1
            self.running_mean_energy.mul_(1 - self.momentum).add_(
                m_batch.detach() * self.momentum
            )
        else:
            if self.use_running_stats and self.num_batches_tracked > 0:
                scale = (self.running_mean_energy + self.eps).rsqrt()
            else:
                scale = (m_batch + self.eps).rsqrt()

        return x * scale


class SeqMaxBatch(nn.Module):
    """SeqMax but max energy taken across both sequence and batch dimensions."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        energy = x.pow(2).mean(dim=-1, keepdim=True)                  # (B, T, 1)
        global_max = energy.max(dim=-2, keepdim=True).values.max(     # scalar-ish
            dim=0, keepdim=True
        ).values                                                       # (1, 1, 1)
        scale = (global_max + self.eps).rsqrt()
        return x * scale


class Tok1Ghost(nn.Module):
    """Tok1 + ghost noise: multiplicative noise from subset-batch vs full-batch ratio.

    Training:
        s_full  = mean_b(RMS(x_{b,0}))   (full batch)
        s_ghost = mean_b'(RMS(x_{b',0})) (random half-batch)
        s_tilde = s * (s_ghost / s_full)
        normalize by s_tilde
    Eval:
        deterministic tok1 (no noise).
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        ghost_frac: float = 0.5,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.ghost_frac = ghost_frac

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        energy_t0 = x[:, 0, :].pow(2).mean(dim=-1)  # (B,)

        if self.training and B > 1:
            s_full = energy_t0.mean()  # scalar

            # sample ghost subset
            n_ghost = max(1, int(B * self.ghost_frac))
            idx = torch.randperm(B, device=x.device)[:n_ghost]
            s_ghost = energy_t0[idx].mean()

            # per-sequence scale with ghost noise
            # s_i_tilde = energy_t0_i * (s_ghost / s_full)
            ratio = (s_ghost + self.eps) / (s_full + self.eps)
            noisy_energy = energy_t0 * ratio  # (B,)
            scale = (noisy_energy.unsqueeze(1).unsqueeze(2) + self.eps).rsqrt()  # (B,1,1)
        else:
            scale = (energy_t0.unsqueeze(1).unsqueeze(2) + self.eps).rsqrt()

        return x * scale


class Tok1BN(nn.Module):
    """Token-1 batch normalization with running stats for eval.

    Training: compute batch mean/var from token 0 across batch (B x D),
              normalize, update running stats.
    Eval:     use running mean/var.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.momentum = momentum

        self.register_buffer("running_mean", torch.zeros(normalized_shape))
        self.register_buffer("running_var", torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        tok0 = x[:, 0, :]  # (B, D)

        if self.training:
            mean = tok0.mean(dim=0)       # (D,)
            var = tok0.var(dim=0, unbiased=False)  # (D,)
            # update running stats
            self.running_mean.mul_(1 - self.momentum).add_(mean.detach() * self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(var.detach() * self.momentum)
        else:
            mean = self.running_mean
            var = self.running_var

        # normalize entire sequence using token-0 batch stats
        scale = (var + self.eps).rsqrt()  # (D,)
        return (x - mean) * scale


class Tok1BNGhost(nn.Module):
    """Tok1BN + ghost noise perturbation during training only.

    Running stats are updated with clean token-1 batch stats (no noise baked in).
    Ghost noise is a training-time multiplicative perturbation on the scale.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        momentum: float = 0.1,
        ghost_frac: float = 0.5,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.momentum = momentum
        self.ghost_frac = ghost_frac

        self.register_buffer("running_mean", torch.zeros(normalized_shape))
        self.register_buffer("running_var", torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        tok0 = x[:, 0, :]  # (B, D)

        if self.training:
            mean = tok0.mean(dim=0)
            var = tok0.var(dim=0, unbiased=False)

            # update running stats with clean stats
            self.running_mean.mul_(1 - self.momentum).add_(mean.detach() * self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(var.detach() * self.momentum)

            # ghost noise on scale
            if B > 1:
                n_ghost = max(1, int(B * self.ghost_frac))
                idx = torch.randperm(B, device=x.device)[:n_ghost]
                ghost_var = tok0[idx].var(dim=0, unbiased=False)
                ratio = ((ghost_var + self.eps) / (var + self.eps)).sqrt()  # (D,)
                scale = (var + self.eps).rsqrt() * ratio
            else:
                scale = (var + self.eps).rsqrt()

            return (x - mean) * scale
        else:
            mean = self.running_mean
            var = self.running_var
            scale = (var + self.eps).rsqrt()
            return (x - mean) * scale


class SeqMean(nn.Module):
    """Normalize by mean energy across the sequence.

    Training: use actual batch mean energy (optionally update running stats).
    Eval:     use running stats if available, else live computation.

    Set use_running_stats=True to accumulate running mean energy during training
    and use it at eval time.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        momentum: float = 0.1,
        use_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.momentum = momentum
        self.use_running_stats = use_running_stats

        if use_running_stats:
            self.register_buffer("running_mean_energy", torch.ones(1))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.running_mean_energy = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        energy = x.pow(2).mean(dim=-1, keepdim=True)  # (B, T, 1)
        mean_energy = energy.mean(dim=-2, keepdim=True)  # (B, 1, 1)

        if self.training:
            scale = (mean_energy + self.eps).rsqrt()  # (B, 1, 1)
            if self.use_running_stats:
                batch_mean = mean_energy.detach().mean()  # scalar
                self.num_batches_tracked += 1
                self.running_mean_energy.mul_(1 - self.momentum).add_(
                    batch_mean * self.momentum
                )
        else:
            if self.use_running_stats and self.num_batches_tracked > 0:
                # use precomputed running stats
                scale = (self.running_mean_energy + self.eps).rsqrt()  # (1,)
                scale = scale.view(1, 1, 1)
            else:
                scale = (mean_energy + self.eps).rsqrt()

        return x * scale


# ══════════════════════════════════════════════════════════════════════════════
# SET 2: New norm variants
# ══════════════════════════════════════════════════════════════════════════════


class Tok190(nn.Module):
    """Tok1 with Q0.90 quantile batch aggregation.

    1. Compute token-1 energy per sample: e_b = mean_d(x_{b,0,d}^2) → (B,)
    2. Batch aggregate: m = Q0.90({e_b}) → scalar
    3. Scale: x ← x / sqrt(m + eps)

    Supports dual eval modes:
        - use_running_stats=False: use batch stats at eval (eval-as-train)
        - use_running_stats=True: use running stats at eval (eval-as-inference)
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        momentum: float = 0.1,
        use_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.momentum = momentum
        self.use_running_stats = use_running_stats

        self.register_buffer("running_mean_energy", torch.ones(1))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        energy_t0 = x[:, 0, :].pow(2).mean(dim=-1)  # (B,)
        m_batch = torch.quantile(energy_t0, 0.90)  # scalar

        if self.training:
            scale = (m_batch + self.eps).rsqrt()
            # update running stats
            self.num_batches_tracked += 1
            self.running_mean_energy.mul_(1 - self.momentum).add_(
                m_batch.detach() * self.momentum
            )
        else:
            if self.use_running_stats and self.num_batches_tracked > 0:
                scale = (self.running_mean_energy + self.eps).rsqrt()
            else:
                scale = (m_batch + self.eps).rsqrt()

        return x * scale


class Tok190Clamp(nn.Module):
    """Tok190 with clamped energies before aggregation.

    1. Compute token-1 energy per sample: e_b = mean_d(x_{b,0,d}^2) → (B,)
    2. Compute clamp bounds: l = Q0.05(e), u = Q0.95(e) → scalars
    3. Clamp energies: e_tilde_b = clamp(e_b, l, u) → (B,)
    4. Batch aggregate: m = Q0.90({e_tilde_b}) → scalar
    5. Scale: x ← x / sqrt(m + eps)

    Supports dual eval modes:
        - use_running_stats=False: use batch stats at eval (eval-as-train)
        - use_running_stats=True: use running stats at eval (eval-as-inference)
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        momentum: float = 0.1,
        use_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.momentum = momentum
        self.use_running_stats = use_running_stats

        self.register_buffer("running_mean_energy", torch.ones(1))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        energy_t0 = x[:, 0, :].pow(2).mean(dim=-1)  # (B,)
        lower = torch.quantile(energy_t0, 0.05)
        upper = torch.quantile(energy_t0, 0.95)
        clamped = energy_t0.clamp(min=lower, max=upper)  # (B,)
        m_batch = torch.quantile(clamped, 0.90)  # scalar

        if self.training:
            scale = (m_batch + self.eps).rsqrt()
            # update running stats
            self.num_batches_tracked += 1
            self.running_mean_energy.mul_(1 - self.momentum).add_(
                m_batch.detach() * self.momentum
            )
        else:
            if self.use_running_stats and self.num_batches_tracked > 0:
                scale = (self.running_mean_energy + self.eps).rsqrt()
            else:
                scale = (m_batch + self.eps).rsqrt()

        return x * scale


class SeqMaxMeanBatch(nn.Module):
    """SeqMax per sample, then mean across batch.

    1. Compute per-token energy: e_{b,t} = mean_d(x_{b,t,d}^2) → (B, T)
    2. Per-sample sequence max: m_b = max_t(e_{b,t}) → (B,)
    3. Batch aggregate: m = mean_b(m_b) → scalar
    4. Scale: x ← x / sqrt(m + eps)

    Supports dual eval modes:
        - use_running_stats=False: use batch stats at eval (eval-as-train)
        - use_running_stats=True: use running stats at eval (eval-as-inference)
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        momentum: float = 0.1,
        use_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.momentum = momentum
        self.use_running_stats = use_running_stats

        self.register_buffer("running_mean_energy", torch.ones(1))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        energy = x.pow(2).mean(dim=-1)  # (B, T)
        seq_max = energy.max(dim=-1).values  # (B,)
        m_batch = seq_max.mean()  # scalar

        if self.training:
            scale = (m_batch + self.eps).rsqrt()
            # update running stats
            self.num_batches_tracked += 1
            self.running_mean_energy.mul_(1 - self.momentum).add_(
                m_batch.detach() * self.momentum
            )
        else:
            if self.use_running_stats and self.num_batches_tracked > 0:
                scale = (self.running_mean_energy + self.eps).rsqrt()
            else:
                scale = (m_batch + self.eps).rsqrt()

        return x * scale


class SeqMaxMedianBatch(nn.Module):
    """SeqMax per sample, then median across batch.

    1. Compute per-token energy: e_{b,t} = mean_d(x_{b,t,d}^2) → (B, T)
    2. Per-sample sequence max: m_b = max_t(e_{b,t}) → (B,)
    3. Batch aggregate: m = median_b(m_b) → scalar
    4. Scale: x ← x / sqrt(m + eps)

    Supports dual eval modes:
        - use_running_stats=False: use batch stats at eval (eval-as-train)
        - use_running_stats=True: use running stats at eval (eval-as-inference)
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        momentum: float = 0.1,
        use_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.momentum = momentum
        self.use_running_stats = use_running_stats

        self.register_buffer("running_mean_energy", torch.ones(1))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        energy = x.pow(2).mean(dim=-1)  # (B, T)
        seq_max = energy.max(dim=-1).values  # (B,)
        m_batch = seq_max.median()  # scalar

        if self.training:
            scale = (m_batch + self.eps).rsqrt()
            # update running stats
            self.num_batches_tracked += 1
            self.running_mean_energy.mul_(1 - self.momentum).add_(
                m_batch.detach() * self.momentum
            )
        else:
            if self.use_running_stats and self.num_batches_tracked > 0:
                scale = (self.running_mean_energy + self.eps).rsqrt()
            else:
                scale = (m_batch + self.eps).rsqrt()

        return x * scale


class SeqPowerMean(nn.Module):
    """Power-mean (p=2) over sequence, per sample.

    1. Compute per-token energy: e_{b,t} = mean_d(x_{b,t,d}^2) → (B, T)
    2. Power-mean over time with p=2: m_b = (1/T * sum_t(e_{b,t}^2))^{1/2} → (B,)
    3. Scale per sample: x_{b,:,:} ← x_{b,:,:} / sqrt(m_b + eps)
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6, p: float = 2.0) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        energy = x.pow(2).mean(dim=-1)  # (B, T)
        # power-mean: (mean(e^p))^{1/p}
        power_mean = energy.pow(self.p).mean(dim=-1).pow(1.0 / self.p)  # (B,)
        scale = (power_mean + self.eps).rsqrt()  # (B,)
        return x * scale.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)


class SeqMeanBatch(nn.Module):
    """Mean energy over batch+time, BN-style running scalar at eval.

    Train:
        1. Compute per-token energy: e_{b,t} = mean_d(x_{b,t,d}^2) → (B, T)
        2. Batch mean energy: m_batch = mean_{b,t}(e_{b,t}) → scalar
        3. Scale: x ← x / sqrt(m_batch + eps)
        4. Update running scalar: m_run ← (1-β)*m_run + β*m_batch with β=0.1

    Supports dual eval modes:
        - use_running_stats=False: use batch stats at eval (eval-as-train)
        - use_running_stats=True: use running stats at eval (eval-as-inference)
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        momentum: float = 0.1,
        use_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.momentum = momentum
        self.use_running_stats = use_running_stats

        self.register_buffer("running_mean_energy", torch.ones(1))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        energy = x.pow(2).mean(dim=-1)  # (B, T)
        m_batch = energy.mean()  # scalar over B and T

        if self.training:
            scale = (m_batch + self.eps).rsqrt()
            # update running stats
            self.num_batches_tracked += 1
            self.running_mean_energy.mul_(1 - self.momentum).add_(
                m_batch.detach() * self.momentum
            )
        else:
            if self.use_running_stats and self.num_batches_tracked > 0:
                scale = (self.running_mean_energy + self.eps).rsqrt()
            else:
                scale = (m_batch + self.eps).rsqrt()

        return x * scale


class SeqPowerMeanBatch(nn.Module):
    """Power-mean (p=2) over batch+time, BN-style running scalar at eval.

    Train:
        1. Compute per-token energy: e_{b,t} = mean_d(x_{b,t,d}^2) → (B, T)
        2. Power-mean over b,t with p=2: m_batch = (1/(BT) * sum_{b,t}(e_{b,t}^2))^{1/2} → scalar
        3. Scale: x ← x / sqrt(m_batch + eps)
        4. Update running scalar: m_run ← (1-β)*m_run + β*m_batch with β=0.1

    Supports dual eval modes:
        - use_running_stats=False: use batch stats at eval (eval-as-train)
        - use_running_stats=True: use running stats at eval (eval-as-inference)
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        momentum: float = 0.1,
        p: float = 2.0,
        use_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.momentum = momentum
        self.p = p
        self.use_running_stats = use_running_stats

        self.register_buffer("running_mean_energy", torch.ones(1))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        energy = x.pow(2).mean(dim=-1)  # (B, T)
        # power-mean over all elements: (mean(e^p))^{1/p}
        m_batch = energy.pow(self.p).mean().pow(1.0 / self.p)  # scalar

        if self.training:
            scale = (m_batch + self.eps).rsqrt()
            # update running stats
            self.num_batches_tracked += 1
            self.running_mean_energy.mul_(1 - self.momentum).add_(
                m_batch.detach() * self.momentum
            )
        else:
            if self.use_running_stats and self.num_batches_tracked > 0:
                scale = (self.running_mean_energy + self.eps).rsqrt()
            else:
                scale = (m_batch + self.eps).rsqrt()

        return x * scale


# ── Registry ──────────────────────────────────────────────────────────────────

NORM_SWEEP_REGISTRY = {
    # Set 1 (original)
    "seq_max": SeqMax,
    "causal_seq_max": CausalSeqMax,
    "stochastic_seq_max": StochasticSeqMax,
    "tok1": Tok1,
    "tok1_batch": Tok1Batch,
    "seq_max_batch": SeqMaxBatch,
    "tok1_ghost": Tok1Ghost,
    "tok1_bn": Tok1BN,
    "tok1_bn_ghost": Tok1BNGhost,
    "seq_mean": SeqMean,
    # Set 2 (new)
    "tok190": Tok190,
    "tok190_clamp": Tok190Clamp,
    "seq_max_mean_batch": SeqMaxMeanBatch,
    "seq_max_median_batch": SeqMaxMedianBatch,
    "seq_power_mean": SeqPowerMean,
    "seq_mean_batch": SeqMeanBatch,
    "seq_power_mean_batch": SeqPowerMeanBatch,
    # standard norms for baseline comparison
    "rmsnorm": None,       # handled via nn.RMSNorm
    "layernorm": None,     # handled via nn.LayerNorm
    "none": None,          # handled via nn.Identity
}


def make_norm(norm_type: str, d_model: int, **kwargs) -> nn.Module:
    """Factory: create a norm module from its concise name."""
    if norm_type == "rmsnorm":
        return nn.RMSNorm(d_model)
    elif norm_type == "layernorm":
        return nn.LayerNorm(d_model)
    elif norm_type == "none":
        return nn.Identity()
    elif norm_type in NORM_SWEEP_REGISTRY:
        cls = NORM_SWEEP_REGISTRY[norm_type]
        return cls(d_model, **kwargs)
    else:
        raise ValueError(
            f"Unknown norm_type {norm_type!r}. "
            f"Choose from: {list(NORM_SWEEP_REGISTRY.keys())}"
        )
