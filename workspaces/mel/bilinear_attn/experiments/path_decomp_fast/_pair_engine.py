"""Pair engine for fast family-diagonal TN similarity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from src.components.base import Term  # noqa: E402
from src.components.similarity import State, _initial_state, _moment  # noqa: E402

from experiments.path_decomp.moments import (  # noqa: E402
    _family_to_tt_and_src,
)
from experiments.path_decomp_fast._orbit_master import _master_moment as _master_moment_orbit
from experiments.path_decomp.moments import _master_moment as _master_moment_ref


# ---------------------------------------------------------------------------
# Canonical 22 families (kept in sync with path_decomp/family_diagonal_tn_heatmaps.py)
# ---------------------------------------------------------------------------

LAYER2_GROUPS = [
    ("00000", ["00000"]),
    ("00001", ["00001", "00100"]),
    ("00010", ["00010", "01000"]),
    ("00011", ["00011", "01100"]),
    ("00101", ["00101"]),
    ("00110", ["00110", "01001"]),
    ("00111", ["00111", "01101"]),
    ("01010", ["01010"]),
    ("01011", ["01011", "01110"]),
    ("01111", ["01111"]),
    ("10000", ["10000"]),
    ("10001", ["10001", "10100"]),
    ("10010", ["10010", "11000"]),
    ("10011", ["10011", "11100"]),
    ("10101", ["10101"]),
    ("10110", ["10110", "11001"]),
    ("10111", ["10111", "11101"]),
    ("11010", ["11010"]),
    ("11011", ["11011", "11110"]),
    ("11111", ["11111"]),
]
CANONICAL_LABELS: list[str] = [
    "direct", "layer1", *[label for label, _members in LAYER2_GROUPS],
]


def _family_from_label(label: str):
    if label in ("direct", "layer1"):
        return label
    return ("layer2", int(label, 2))


CANONICAL_FAMILIES = [_family_from_label(label) for label in CANONICAL_LABELS]
LABEL_TO_INDEX = {label: i for i, label in enumerate(CANONICAL_LABELS)}


def _no_sym(terms: Sequence[Term]) -> list[Term]:
    return [Term(t.tn, t.legs, symmetries=()) for t in terms]


# ---------------------------------------------------------------------------
# Per-step artifacts
# ---------------------------------------------------------------------------


@dataclass
class StepArtifacts:
    s_self_embed: torch.Tensor
    embed_terms: list
    ta1_no_sym: list
    ta2: list
    th: list
    S_self: dict


@torch.no_grad()
def build_step_artifacts(model) -> StepArtifacts:
    assert model.n_layers == 2, "Only 2-layer models supported."

    state0 = _initial_state(model)
    n_ctx = state0.s_aa.shape[0]
    like = dict(device=state0.s_aa.device, dtype=state0.s_aa.dtype)

    # --- Embed: self s_aa = sum_{x, y in ta} _moment(x, y, 0, 0, state0) ---
    embed_terms = model.embed.terms(n_ctx, **like)
    s_self = sum(
        _moment(x, y, 0, 0, state0)
        for x in embed_terms for y in embed_terms
    )

    # --- Layer-1 self blocks (ml=mr=0, all sl, sr). Strip symmetries to match
    # the no-sym plan used by _master_moment. ---
    ta1_ns = _no_sym(model.layers[0].terms(n_ctx, **like))
    self_state = State(s_self, s_self, s_self)  # ml=mr=0 only reads s_aa
    S_self = {
        (sl, sr): _moment(ta1_ns[sl], ta1_ns[sr], 0, 0, self_state)
        for sl in (0, 1) for sr in (0, 1)
    }

    ta2 = model.layers[1].terms(n_ctx, **like)
    th = model.unembed.terms(n_ctx, **like)

    return StepArtifacts(
        s_self_embed=s_self,
        embed_terms=list(embed_terms),
        ta1_no_sym=ta1_ns,
        ta2=list(ta2),
        th=list(th),
        S_self=S_self,
    )


# ---------------------------------------------------------------------------
# Per-pair compute
# ---------------------------------------------------------------------------


def _pack_S(S_self_a: dict, S_self_b: dict, S_cross: dict,
            n_ctx: int, d_padded: int, like: dict) -> torch.Tensor:
    S = torch.zeros(2, 2, 2, 2, n_ctx, d_padded, n_ctx, d_padded, **like)
    for sl in (0, 1):
        for sr in (0, 1):
            S[0, 0, sl, sr] = S_self_a[(sl, sr)]
            S[1, 1, sl, sr] = S_self_b[(sl, sr)]
            S[0, 1, sl, sr] = S_cross[(sl, sr)]
    return S


def parse_families(spec: str | Sequence[str] | None) -> list[str]:
    if spec is None or spec == "all":
        return list(CANONICAL_LABELS)
    if isinstance(spec, str):
        labels = [p.strip() for p in spec.split(",") if p.strip()]
    else:
        labels = [str(p).strip() for p in spec if str(p).strip()]
    bad = [x for x in labels if x not in LABEL_TO_INDEX]
    if bad:
        raise ValueError(f"Unknown family labels: {bad}")
    return labels


def family_indices(spec: str | Sequence[str] | None) -> list[int]:
    return [LABEL_TO_INDEX[label] for label in parse_families(spec)]


@torch.no_grad()
def compute_pair(
    art_a: StepArtifacts,
    art_b: StepArtifacts,
    families: str | Sequence[str] | None = "all",
    use_orbit_master: bool = True,
    bridges_to_f32: bool = False,
    is_self: bool = False,
) -> torch.Tensor:
    s_self_a = art_a.s_self_embed
    s_self_b = art_b.s_self_embed
    device = s_self_a.device
    dtype = s_self_a.dtype
    like = dict(device=device, dtype=dtype)
    n_ctx, d_padded = s_self_a.shape[0], s_self_a.shape[1]
    selected_labels = parse_families(families)
    selected_indices = [LABEL_TO_INDEX[label] for label in selected_labels]
    selected_families = [CANONICAL_FAMILIES[idx] for idx in selected_indices]

    # 1. Embed cross: s_ab only. state0 identity is model-agnostic so reuse from a.
    # Build a minimal initial state on-the-fly (same shape as art_a.s_self's pre-embed).
    # We actually need initial-state s_aa = I_n ⊗ I_{d_in+1}, of the *input* dim.
    # The embed_terms' input-leg dim is d_in+1 (vocab_size+1) which differs from
    # d_padded (d_model+1) post-embed; so we rebuild state0 from the embed
    # network's input-leg size.
    d_in = art_a.embed_terms[0].tn.ind_size('in:d0')
    eye_n = torch.eye(n_ctx, **like)
    eye_d = torch.eye(d_in, **like)
    s0 = torch.einsum('ij,kl->ikjl', eye_n, eye_d)
    state0 = State(s0, s0, s0)
    if is_self:
        s_ab_embed = s_self_a
    else:
        s_ab_embed = sum(
            _moment(x, y, 0, 1, state0)
            for x in art_a.embed_terms for y in art_b.embed_terms
        )
    state_post_embed = State(s_self_a, s_ab_embed, s_self_b)

    # 2. Layer-1 cross s_split: only (ml=0, mr=1, sl, sr).
    S_cross = {
        (sl, sr): _moment(art_a.ta1_no_sym[sl], art_b.ta1_no_sym[sr],
                          0, 1, state_post_embed)
        for sl in (0, 1) for sr in (0, 1)
    }

    # 3. Pack S and build the two needed layer-2 masters.
    S = _pack_S(art_a.S_self, art_b.S_self, S_cross, n_ctx, d_padded, like)
    ta2_a, ta2_b = art_a.ta2, art_b.ta2
    needed = {
        (_family_to_tt_and_src(fam)[0], _family_to_tt_and_src(fam)[0])
        for fam in selected_families
    }
    master_fn = _master_moment_orbit if use_orbit_master else _master_moment_ref
    masters = {
        (tta, ttb): master_fn(
            ta2_a[tta],
            ta2_b[ttb],
            0,
            1,
            S,
            bridges_to_f32=bridges_to_f32,
        ) if master_fn is _master_moment_orbit else master_fn(ta2_a[tta], ta2_b[ttb], 0, 1, S)
        for tta, ttb in sorted(needed)
    }

    # 4. Head + trace for every canonical family; stack before host transfer.
    th_a, th_b = art_a.th[0], art_b.th[0]
    out_scalars = []
    for fam in selected_families:
        tta, src = _family_to_tt_and_src(fam)
        master = masters[(tta, tta)]  # same-family diagonal
        idx = (slice(None),) * 4 + tuple(src) + tuple(src)
        s_ab_l2 = master[idx]
        proxy = State(s_ab_l2, s_ab_l2, s_ab_l2)
        s_ab_out = _moment(th_a, th_b, 0, 1, proxy)
        out_scalars.append(torch.einsum('ijij->', s_ab_out[:, 1:, :, 1:]))

    stacked = torch.stack(out_scalars)
    out = torch.full((len(CANONICAL_LABELS),), torch.nan, dtype=torch.float64)
    out[selected_indices] = stacked.detach().to(dtype=torch.float64, device="cpu")
    return out


# ---------------------------------------------------------------------------
# Small convenience wrappers
# ---------------------------------------------------------------------------


def family_labels() -> list[str]:
    return list(CANONICAL_LABELS)


def family_count() -> int:
    return len(CANONICAL_FAMILIES)
