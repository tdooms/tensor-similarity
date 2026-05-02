from __future__ import annotations

import pytest
import torch

from src.components.base import Term
from src.components.mlp import MLP
from src.components.similarity import State, _initial_state, _moment, _step
from src.models.transformer import Transformer

from workspaces.mel.bilinear_attn.experiments.path_decomp.family_diagonal_tn_heatmaps import (
    family_diagonal_inner_products_component,
)
from workspaces.mel.bilinear_attn.experiments.path_decomp.moments import (
    _family_to_tt_and_src,
    _master_moment as _master_moment_ref,
)
from workspaces.mel.bilinear_attn.experiments.path_decomp_fast._orbit_master import (
    _master_moment as _master_moment_orbit,
)
from workspaces.mel.bilinear_attn.experiments.path_decomp_fast._pair_engine import (
    CANONICAL_FAMILIES,
    CANONICAL_LABELS,
    build_step_artifacts,
    compute_pair,
)


LIKE = dict(device="cpu", dtype=torch.float64)


def _make_attn_only(seed=42, d_in=2, d_model=4, n_head=1, n_ctx=2, d_h=8, d_out=2, scale=0.5, mask="none"):
    torch.manual_seed(seed)
    m = Transformer(d_in, d_model, n_head, n_ctx, d_h, d_out, n_layer=2, mask=mask, scale=scale).double()
    for layer in m.body:
        layer.mlp = MLP(d_model, d_h, scale=0.0).double()
    return m


def _strip_sym(terms):
    return [Term(t.tn, t.legs, symmetries=()) for t in terms]


def _stack_s_split(s_split, n_ctx, d_padded, like):
    S = torch.empty(2, 2, 2, 2, n_ctx, d_padded, n_ctx, d_padded, **like)
    for (ml, mr, sl, sr), v in s_split.items():
        S[ml, mr, sl, sr] = v
    return S


def _build_reference_master_inputs(model_a, model_b):
    state = _initial_state(model_a)
    n_ctx = state.s_aa.shape[0]
    like = dict(device=state.s_aa.device, dtype=state.s_aa.dtype)

    state = _step(state, model_a.embed.terms(n_ctx, **like), model_b.embed.terms(n_ctx, **like))
    d_padded = state.s_aa.shape[1]

    ta1 = _strip_sym(model_a.layers[0].terms(n_ctx, **like))
    tb1 = _strip_sym(model_b.layers[0].terms(n_ctx, **like))
    sides = {0: ta1, 1: tb1}

    s_split = {}
    for ml in (0, 1):
        for mr in (0, 1):
            for sl in (0, 1):
                for sr in (0, 1):
                    s_split[(ml, mr, sl, sr)] = _moment(sides[ml][sl], sides[mr][sr], ml, mr, state)

    S = _stack_s_split(s_split, n_ctx, d_padded, like)
    ta2 = model_a.layers[1].terms(n_ctx, **like)
    tb2 = model_b.layers[1].terms(n_ctx, **like)
    return ta2, tb2, S


def test_orbit_master_matches_no_sym():
    a = _make_attn_only(seed=42)
    b = _make_attn_only(seed=7)
    ta2, tb2, S = _build_reference_master_inputs(a, b)

    got = _master_moment_orbit(ta2[1], tb2[1], 0, 1, S, bridges_to_f32=False)
    ref = _master_moment_ref(ta2[1], tb2[1], 0, 1, S)
    diff = (got - ref).abs().max().item()
    assert diff < 1e-12


def test_pair_matches_reference():
    a = _make_attn_only(seed=42)
    b = _make_attn_only(seed=7)
    comp_a = a.component(ignore_bn=True) if hasattr(a, "component") else None
    comp_b = b.component(ignore_bn=True) if hasattr(b, "component") else None
    if comp_a is None or comp_b is None:
        from models.components.model import AttentionLMComponent

        comp_a = AttentionLMComponent.from_trained_model(a, ignore_norms=True).to(**LIKE)
        comp_b = AttentionLMComponent.from_trained_model(b, ignore_norms=True).to(**LIKE)

    ref = family_diagonal_inner_products_component(comp_a, comp_b)
    ref_vals = torch.tensor([ref[f] for f in CANONICAL_FAMILIES], dtype=torch.float64)

    art_a = build_step_artifacts(comp_a)
    art_b = build_step_artifacts(comp_b)
    got = compute_pair(art_a, art_b, families="all", use_orbit_master=True, bridges_to_f32=False)

    rel = (got - ref_vals).abs() / torch.clamp(ref_vals.abs(), min=1e-30)
    assert torch.max(rel).item() < 1e-10


def test_bf16_close_to_f32():
    try:
        from models.components.model import AttentionLMComponent
    except Exception as exc:
        pytest.skip(str(exc))

    a = _make_attn_only(seed=42)
    b = _make_attn_only(seed=7)
    comp_a_f32 = AttentionLMComponent.from_trained_model(a, ignore_norms=True).to(device="cpu", dtype=torch.float32)
    comp_b_f32 = AttentionLMComponent.from_trained_model(b, ignore_norms=True).to(device="cpu", dtype=torch.float32)
    comp_a_bf16 = AttentionLMComponent.from_trained_model(a, ignore_norms=True).to(device="cpu", dtype=torch.bfloat16)
    comp_b_bf16 = AttentionLMComponent.from_trained_model(b, ignore_norms=True).to(device="cpu", dtype=torch.bfloat16)

    ref = compute_pair(build_step_artifacts(comp_a_f32), build_step_artifacts(comp_b_f32), families="all", use_orbit_master=True, bridges_to_f32=True)
    got = compute_pair(build_step_artifacts(comp_a_bf16), build_step_artifacts(comp_b_bf16), families="all", use_orbit_master=True, bridges_to_f32=True)
    rel = (got - ref).abs() / torch.clamp(ref.abs(), min=1e-30)
    assert torch.nanmax(rel).item() < 5e-3


def test_single_family_consistency():
    from models.components.model import AttentionLMComponent

    a = _make_attn_only(seed=42)
    b = _make_attn_only(seed=7)
    comp_a = AttentionLMComponent.from_trained_model(a, ignore_norms=True).to(**LIKE)
    comp_b = AttentionLMComponent.from_trained_model(b, ignore_norms=True).to(**LIKE)

    art_a = build_step_artifacts(comp_a)
    art_b = build_step_artifacts(comp_b)
    all_vals = compute_pair(art_a, art_b, families="all", use_orbit_master=True, bridges_to_f32=False)
    direct = compute_pair(art_a, art_b, families=["direct"], use_orbit_master=True, bridges_to_f32=False)
    direct_idx = CANONICAL_LABELS.index("direct")
    assert torch.allclose(all_vals[direct_idx], direct[direct_idx], atol=1e-10, rtol=1e-10)
