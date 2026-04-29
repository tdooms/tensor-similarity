"""Forward path decomposition for 2-layer attention-only models.

Implements F_rho(X) directly in PyTorch (no tensor networks). Used by the
forward-decomposition (test 1) and family-aggregation (test 2) tests.

Path families (see README.md):
  - 'direct'        :  l1-residual o l2-residual
  - 'layer1'        :  l1-active   o l2-residual    (sum over l1 heads)
  - ('layer2', rho) :  per-slot (l1-residual or l1-active) o l2-active
                       for rho in {0,1}^5; bit i = source of slot i
                       (0 = direct/residual, 1 = composed/active).

Slot bit ordering matches README's (alpha, beta, gamma, delta, eta):
    bit 0 = alpha  = Q1 (query side of (q1.k1))
    bit 1 = beta   = K1 (key   side of (q1.k1))
    bit 2 = gamma  = Q2 (query side of (q2.k2))
    bit 3 = delta  = K2 (key   side of (q2.k2))
    bit 4 = eta    = V  (value side)

All forwards run under torch.no_grad() and assume scale=0 on the MLPs of the
2-layer Transformer (i.e., attention-only).
"""
from itertools import product

import torch
from einops import rearrange, einsum


N_LAYER2_FAMILIES = 32
N_FAMILIES = 1 + 1 + N_LAYER2_FAMILIES  # direct + layer1 + 32 layer-2 families


def enumerate_families():
    """Yield all 34 family identifiers."""
    yield 'direct'
    yield 'layer1'
    for rho in range(N_LAYER2_FAMILIES):
        yield ('layer2', rho)


def _split_heads(linear, x, n_head):
    """Apply a Linear and split its output into per-head shape (..., seq, n_head, d_head)."""
    return rearrange(linear(x), '... (n h) -> ... n h', n=n_head)


def _l1_per_head_outputs(attn, x):
    """Return per-head contributions to o(z), unscaled (no residual, no scale).

    Shape: (..., n_head, seq, d_model). Sum over the n_head axis times
    `attn.scale` reproduces the active-attention output of layer 1.
    """
    n, d = attn.n_head, attn.d_head
    q1, k1, q2, k2, v = (_split_heads(p, x, n) for p in (attn.q1, attn.k1, attn.q2, attn.k2, attn.v))
    q1, k1, q2, k2 = (attn.rotary(t) for t in (q1, k1, q2, k2))
    s1 = einsum(q1, k1, '... sq n h, ... sk n h -> ... n sq sk')
    s2 = einsum(q2, k2, '... sq n h, ... sk n h -> ... n sq sk')
    pattern = attn.mask((s1 * s2) / (d ** 2))
    z = einsum(pattern, v, '... n sq sk, ... sk n h -> ... n sq h')
    # Per-head O: o.weight is (d_model, n*d_head); reshape to (n, d_model, d_head).
    o_w = rearrange(attn.o.weight, 'd (n h) -> n d h', n=n)
    return einsum(z, o_w, '... n s h, n d h -> ... n s d')


def _l2_head_with_slots(attn, q1_in, k1_in, q2_in, k2_in, v_in):
    """Layer-2 attention output (sum over heads) with slot-specific inputs.

    Inputs: each (..., seq, d_model). Returns (..., seq, d_model). No scale,
    no residual.
    """
    n, d = attn.n_head, attn.d_head
    q1 = attn.rotary(_split_heads(attn.q1, q1_in, n))
    k1 = attn.rotary(_split_heads(attn.k1, k1_in, n))
    q2 = attn.rotary(_split_heads(attn.q2, q2_in, n))
    k2 = attn.rotary(_split_heads(attn.k2, k2_in, n))
    v = _split_heads(attn.v, v_in, n)
    s1 = einsum(q1, k1, '... sq n h, ... sk n h -> ... n sq sk')
    s2 = einsum(q2, k2, '... sq n h, ... sk n h -> ... n sq sk')
    pattern = attn.mask((s1 * s2) / (d ** 2))
    z = einsum(pattern, v, '... n sq sk, ... sk n h -> ... sq n h')
    return attn.o(rearrange(z, '... s n h -> ... s (n h)'))


def _rho_to_slot_inputs(rho, r0, r_active, l1_per_head=None, fine_combo=None):
    """Build the 5 layer-2 slot inputs for an l2-active family with bits rho.

    rho: int in [0, 32). bits = [alpha, beta, gamma, delta, eta] in slot
        positions [Q1, K1, Q2, K2, V] = [in:d3, in:d1, in:d4, in:d2, in:d0].
    Slot order returned: (q1, k1, q2, k2, v) matching attn.{q1,k1,q2,k2,v}.

    If fine_combo is None: composed slot reads `r_active` (sum over l1 heads).
    Otherwise: composed slot reads `l1_per_head[..., h, :, :]` for the h
    given by fine_combo at the slot's composed-slot index.
    """
    bits = [(rho >> i) & 1 for i in range(5)]
    inputs = []
    ci = 0
    for b in bits:
        if b == 0:
            inputs.append(r0)
        elif fine_combo is None:
            inputs.append(r_active)
        else:
            inputs.append(l1_per_head[..., fine_combo[ci], :, :])
            ci += 1
    return inputs


@torch.no_grad()
def family_outputs(transformer, x):
    """Compute each family's contribution at the layer-2 output (pre-head).

    Returns dict {family_id: tensor (..., seq, d_model)}. Sum over all values
    equals the layer-2 output before head linear.
    """
    embed = transformer.embed
    attn1 = transformer.body[0].attn
    attn2 = transformer.body[1].attn
    s1, s2 = attn1.scale, attn2.scale

    h0 = embed(x)
    l1_per_head = _l1_per_head_outputs(attn1, h0)
    r0 = (1 - s1) * h0
    r_active = s1 * l1_per_head.sum(dim=-3)

    out = {
        'direct': (1 - s2) * r0,
        'layer1': (1 - s2) * r_active,
    }
    for rho in range(N_LAYER2_FAMILIES):
        slot_in = _rho_to_slot_inputs(rho, r0, r_active)
        out[('layer2', rho)] = s2 * _l2_head_with_slots(attn2, *slot_in)
    return out


@torch.no_grad()
def family_outputs_fine(transformer, x):
    """Compute each family by SUMMING fine-grained per-head terms.

    For 'direct' and 'layer1' the result is identical to family_outputs() (no
    composed slots). For each ('layer2', rho), composed slots are enumerated
    over all H1^k_rho head choices and accumulated, giving the same family
    value as family_outputs() but via a different aggregation order.
    """
    embed = transformer.embed
    attn1 = transformer.body[0].attn
    attn2 = transformer.body[1].attn
    s1, s2 = attn1.scale, attn2.scale
    H1 = attn1.n_head

    h0 = embed(x)
    l1_per_head = _l1_per_head_outputs(attn1, h0)
    r0 = (1 - s1) * h0
    r_per_head = s1 * l1_per_head  # composed-slot read of an individual head

    out = {
        'direct': (1 - s2) * r0,
        'layer1': (1 - s2) * r_per_head.sum(dim=-3),
    }
    for rho in range(N_LAYER2_FAMILIES):
        bits = [(rho >> i) & 1 for i in range(5)]
        n_composed = sum(bits)
        accum = torch.zeros_like(r0)
        for combo in product(range(H1), repeat=n_composed):
            slot_in = _rho_to_slot_inputs(rho, r0, None,
                                          l1_per_head=r_per_head, fine_combo=combo)
            accum = accum + s2 * _l2_head_with_slots(attn2, *slot_in)
        out[('layer2', rho)] = accum
    return out


@torch.no_grad()
def forward_via_decomposition(transformer, x):
    """Sum every family's contribution and apply the head linear."""
    fams = family_outputs(transformer, x)
    layer2_out = sum(fams.values())
    return transformer.head(layer2_out)
