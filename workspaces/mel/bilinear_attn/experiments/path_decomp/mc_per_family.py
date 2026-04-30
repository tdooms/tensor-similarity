"""Per-family Monte-Carlo similarity for 2-layer quadratic attention.

Mirrors the TN family-pair decomposition in ``moments.py`` / ``run_big_experiment.py``
but estimates every entry

    <F_rho, F_sigma> ~= E_x[ sum_{t,v} unembed(F_rho(x))_{t,v}
                                   * unembed(F_sigma(x))_{t,v} ]

by Monte-Carlo over Gaussian residual-stream inputs ``x ~ N(0, I_{d_model})``
with per-batch vectorisation across families.

The forward path decomposition here targets ``AttentionLM`` with
``attn_type='quadratic'`` (``models.attention_kernels.bilinear.QuadraticAttention``).
The 34 families are those described in ``README.md``:

    - 'direct'                (layer-1 residual then layer-2 residual)
    - 'layer1'                (layer-1 active then layer-2 residual)
    - ('layer2', rho) for rho in 0..31  (layer-2 active, per-slot source bits)

The source-bit ordering matches ``forward.py``:
    bit 0 = alpha = Q1, bit 1 = beta = K1, bit 2 = gamma = Q2,
    bit 3 = delta = K2, bit 4 = eta = V.

Norm handling matches the TN script: ``embed_norm`` is skipped (x already lives
at the residual stream), ``final_norm`` is treated as identity, and the head
is the model's ``unembed``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange, einsum


N_LAYER2_FAMILIES = 32
N_FAMILIES = 1 + 1 + N_LAYER2_FAMILIES


def enumerate_families():
    yield 'direct'
    yield 'layer1'
    for rho in range(N_LAYER2_FAMILIES):
        yield ('layer2', rho)


FAMILY_LIST = list(enumerate_families())
FAMILY_INDEX = {f: i for i, f in enumerate(FAMILY_LIST)}


# ---------------------------------------------------------------------------
# Quadratic-attention per-family pieces
# ---------------------------------------------------------------------------

def _split(linear: nn.Linear, x: torch.Tensor, n_head: int) -> torch.Tensor:
    """Linear then split last dim into (n_head, d_head)."""
    return rearrange(linear(x), "... s (n h) -> ... s n h", n=n_head)


def _l1_per_head_outputs(attn, x: torch.Tensor) -> torch.Tensor:
    """Per-head layer-1 active outputs, unscaled, no residual.

    Returns shape (..., n_head, seq, d_model). Sum over n_head times attn.scale
    reproduces the active component of ``attn(x) - (1 - scale) * x``.

    This works for ``QuadraticAttention``: q == k1 and k == k2 in the bilinear
    sense, but the slot structure is preserved by just reusing ``attn.q``,
    ``attn.k`` twice.
    """
    n, d = attn.n_head, attn.d_head
    q = _split(attn.q, x, n)
    k = _split(attn.k, x, n)
    v = _split(attn.v, x, n)
    q = attn.rotary(attn.norm_qk(q))
    k = attn.rotary(attn.norm_qk(k))
    scores = einsum(q, k, "... sq n h, ... sk n h -> ... n sq sk")
    pattern = (scores / d).square()
    T = pattern.shape[-1]
    pattern = pattern * attn.causal_mask[None, None, :T, :T]
    z = einsum(pattern, v, "... n sq sk, ... sk n h -> ... n sq h")
    # Per-head O: o.weight is (d_model, n*d_head); reshape to (n, d_model, d_head).
    o_w = rearrange(attn.o.weight, "d (n h) -> n d h", n=n)
    return einsum(z, o_w, "... n s h, n d h -> ... n s d")


def _l2_active_with_slots(attn, q1_in, k1_in, q2_in, k2_in, v_in):
    """Layer-2 active attention with per-slot inputs, no scale/residual.

    For quadratic attention the two QK circuits share parameters
    (``attn.q``, ``attn.k``), but we still route independent inputs through
    each slot so the four (K1, Q1, K2, Q2, V) source bits are treated as
    distinct. Inputs are (..., seq, d_model); returns (..., seq, d_model).
    """
    n, d = attn.n_head, attn.d_head
    q1 = attn.rotary(attn.norm_qk(_split(attn.q, q1_in, n)))
    k1 = attn.rotary(attn.norm_qk(_split(attn.k, k1_in, n)))
    q2 = attn.rotary(attn.norm_qk(_split(attn.q, q2_in, n)))
    k2 = attn.rotary(attn.norm_qk(_split(attn.k, k2_in, n)))
    v = _split(attn.v, v_in, n)
    s1 = einsum(q1, k1, "... sq n h, ... sk n h -> ... n sq sk")
    s2 = einsum(q2, k2, "... sq n h, ... sk n h -> ... n sq sk")
    pattern = (s1 * s2) / (d ** 2)
    T = pattern.shape[-1]
    pattern = pattern * attn.causal_mask[None, None, :T, :T]
    z = einsum(pattern, v, "... n sq sk, ... sk n h -> ... sq n h")
    return attn.o(rearrange(z, "... s n h -> ... s (n h)"))


def _rho_to_slot_inputs(rho: int, r0: torch.Tensor, r_active: torch.Tensor):
    """Slot inputs (q1, k1, q2, k2, v) for an l2-active family with bits rho.

    Slot bit layout (README): bits = (alpha, beta, gamma, delta, eta) =
    (Q1, K1, Q2, K2, V). Return order matches ``_l2_active_with_slots``'s
    positional args (q1, k1, q2, k2, v).
    """
    alpha = (rho >> 0) & 1
    beta  = (rho >> 1) & 1
    gamma = (rho >> 2) & 1
    delta = (rho >> 3) & 1
    eta   = (rho >> 4) & 1
    pick = lambda b: r_active if b else r0
    return pick(alpha), pick(beta), pick(gamma), pick(delta), pick(eta)


@torch.no_grad()
def family_outputs_attnlm(model, x: torch.Tensor) -> dict:
    """Compute each family's contribution at the layer-2 residual-stream output.

    ``model`` is an ``AttentionLM`` with 2 quadratic-attention layers.
    ``x`` is the residual-stream input, shape (B, n_ctx, d_model). Norms are
    assumed to have been stripped / set to identity by the caller (matching
    the TN path-decomp setup).

    Returns dict {family: tensor (B, n_ctx, d_model)} whose values sum to
    ``model.layers[1](model.layers[0](x))`` (before final_norm / unembed).
    """
    assert len(model.layers) == 2, "path decomposition is 2-layer only"
    attn1, attn2 = model.layers[0], model.layers[1]
    s1, s2 = attn1.scale, attn2.scale

    l1_per_head = _l1_per_head_outputs(attn1, x)        # (B, H1, T, d_model)
    r0 = (1.0 - s1) * x                                  # direct part (layer-1 residual)
    r_active = s1 * l1_per_head.sum(dim=-3)              # active part (sum over H1 heads)

    out = {
        'direct': (1.0 - s2) * r0,
        'layer1': (1.0 - s2) * r_active,
    }
    for rho in range(N_LAYER2_FAMILIES):
        qi, ki, qj, kj, vi = _rho_to_slot_inputs(rho, r0, r_active)
        out[('layer2', rho)] = s2 * _l2_active_with_slots(attn2, qi, ki, qj, kj, vi)
    return out


# ---------------------------------------------------------------------------
# MC family-pair accumulator
# ---------------------------------------------------------------------------

def _strip_norms(model):
    """In-place: drop embed_norm and set final_norm to Identity.

    Matches the TN similarity / path-decomp convention (norm_type='tok0' has
    no learnable params and is effectively a per-sample scale — removing it
    changes the cosine only by a constant factor that cancels).
    """
    model.embed_norm = None
    model.final_norm = nn.Identity()
    model.layer_norms = None
    return model


# ---------------------------------------------------------------------------
# Residual-stream samplers
# ---------------------------------------------------------------------------

def make_gaussian_resid_sampler(model_A, model_B, device, dtype):
    """x_A = x_B = N(0, I_{d_model}) at every (batch, position)."""
    n_ctx, d_model = model_A.n_ctx, model_A.d_model

    def sampler(bs: int):
        x = torch.randn(bs, n_ctx, d_model, device=device, dtype=dtype)
        return x, x
    return sampler


def make_gaussian_onehot_sampler(model_A, model_B, device, dtype):
    """z ~ N(0, I_V) at every position; x_{A/B} = z @ W_E^{A/B}.

    Covers the span of each model's embedding matrix with the same latent
    direction on both sides (variance-reduced coupling).
    """
    n_ctx = model_A.n_ctx
    V = model_A.unembed.out_features

    def sampler(bs: int):
        z = torch.randn(bs, n_ctx, V, device=device, dtype=dtype)
        x_A = z @ model_A.embed.weight.to(dtype)
        x_B = z @ model_B.embed.weight.to(dtype)
        return x_A, x_B
    return sampler


def make_token_sampler(model_A, model_B, device, dtype,
                       vocab_size: int, bos_token_id: int | None,
                       pool_size: int = 50_000, seed: int = 0):
    """Sample sequences from the induction-heads repeated-token generator,
    embed with each model, and return (x_A, x_B) at the residual stream.

    Pre-generates ``pool_size`` sequences up-front (matching the training
    distribution in ``experiments/induction_heads/data.py``); each call
    draws ``bs`` of them with replacement.
    """
    # Local import so this module stays usable without the induction_heads pkg.
    from experiments.induction_heads.data import RepeatedTokenDataset

    n_ctx = model_A.n_ctx
    ds = RepeatedTokenDataset(
        vocab_size=vocab_size, n_ctx=n_ctx,
        n_samples=pool_size, seed=seed, bos_token_id=bos_token_id,
    )
    pool_ids = ds.data.to(device=device)  # (pool_size, n_ctx) long
    N = pool_ids.shape[0]
    gen = torch.Generator(device=device if device.type == "cpu" else "cpu")
    gen.manual_seed(seed + 1)

    def sampler(bs: int):
        idx = torch.randint(0, N, (bs,), generator=gen).to(device)
        ids = pool_ids[idx]  # (bs, n_ctx)
        x_A = model_A.embed(ids).to(dtype)
        x_B = model_B.embed(ids).to(dtype)
        return x_A, x_B
    return sampler


SAMPLER_FACTORIES = {
    'gaussian_resid':  make_gaussian_resid_sampler,
    'gaussian_onehot': make_gaussian_onehot_sampler,
    # 'tokens' requires extra args (vocab_size, bos_token_id) – handle in driver.
}


@torch.no_grad()
def mc_family_pairs(
    model_A,
    model_B,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    n_samples: int = 200_000,
    batch_size: int = 8192,
    sampler=None,
):
    """Estimate the 34x34 family-pair matrices (M_AB, M_AA, M_BB).

    Args:
        sampler: callable ``bs -> (x_A, x_B)`` returning residual-stream
            inputs of shape ``(bs, n_ctx, d_model)`` for each side. Defaults
            to the Gaussian-over-residual sampler (legacy behaviour).

    Returns:
        dict with keys 'AB', 'AA', 'BB' each mapping to a (34, 34) tensor on
        ``device`` with ``dtype``. Entry ``[i, j]`` is an estimate of

            <F_i^A, F_j^B> = (1/N) sum_{sample, t, v} logits_A_i[t,v] * logits_B_j[t,v]

        normalised by the total count of (sample, t, v) triples so that the
        diagonal sums reproduce the MC cosine numerator used by
        ``mc_similarity`` in ``tn_sim.mc_similarity``.

    Notes:
        - ``model_A`` / ``model_B`` are ``AttentionLM``s already cast to
          ``dtype`` and on ``device`` with norms stripped (use
          ``_strip_norms`` beforehand).
    """
    n_ctx = model_A.n_ctx
    d_model = model_A.d_model
    V = model_A.unembed.out_features
    F = N_FAMILIES

    # Sanity: both models must share the geometry we're contracting over.
    assert model_B.n_ctx == n_ctx and model_B.d_model == d_model
    assert model_B.unembed.out_features == V

    if sampler is None:
        sampler = make_gaussian_resid_sampler(model_A, model_B, device, dtype)

    M_AB = torch.zeros(F, F, device=device, dtype=dtype)
    M_AA = torch.zeros(F, F, device=device, dtype=dtype)
    M_BB = torch.zeros(F, F, device=device, dtype=dtype)
    total = 0

    W_A = model_A.unembed.weight  # (V, d_model)
    W_B = model_B.unembed.weight
    b_A = model_A.unembed.bias    # usually None
    b_B = model_B.unembed.bias

    done = 0
    while done < n_samples:
        bs = min(batch_size, n_samples - done)
        x_A, x_B = sampler(bs)

        fams_A = family_outputs_attnlm(model_A, x_A)
        fams_B = family_outputs_attnlm(model_B, x_B)

        # Stack (F, B, T, d_model) -> (F, B*T, d_model). Apply unembed once.
        stack_A = torch.stack([fams_A[f] for f in FAMILY_LIST], dim=0)
        stack_B = torch.stack([fams_B[f] for f in FAMILY_LIST], dim=0)
        flat_A = stack_A.reshape(F, bs * n_ctx, d_model)
        flat_B = stack_B.reshape(F, bs * n_ctx, d_model)
        del stack_A, stack_B, fams_A, fams_B

        # logits_rho = flat @ W.T (+ b). Shape (F, B*T, V).
        logits_A = flat_A @ W_A.T
        if b_A is not None:
            logits_A = logits_A + b_A
        logits_B = flat_B @ W_B.T
        if b_B is not None:
            logits_B = logits_B + b_B
        del flat_A, flat_B

        # Reshape to (F, B*T*V) and do 34x34 matmuls in one shot.
        la = logits_A.reshape(F, -1)
        lb = logits_B.reshape(F, -1)
        del logits_A, logits_B
        M_AB += la @ lb.T
        M_AA += la @ la.T
        M_BB += lb @ lb.T
        del la, lb

        total += bs * n_ctx
        done += bs

    # Divide by (total samples * n_ctx) to match mc_similarity's normalisation.
    M_AB /= max(1, total)
    M_AA /= max(1, total)
    M_BB /= max(1, total)

    return {'AB': M_AB, 'AA': M_AA, 'BB': M_BB}
