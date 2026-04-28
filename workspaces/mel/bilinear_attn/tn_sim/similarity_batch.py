"""Batched TN similarity using true contraction batching."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Iterable, Sequence

import torch
from quimb.tensor import Tensor, TensorNetwork

from src.components.base import Term
from models.components.model import AttentionLMComponent, _validate_model_for_tn_similarity
from tn_sim.similarity import _validate_model_compatibility


_EXPR_CACHE_BATCH: dict = {}


@dataclass
class StateBatched:
    """Batched second-moment state with leading batch dimension."""

    S_aa: torch.Tensor  # (B, n_ctx, d+1, n_ctx, d+1)
    S_bb: torch.Tensor
    S_ab: torch.Tensor

    @staticmethod
    def from_models(models: Sequence[AttentionLMComponent]) -> "StateBatched":
        comps = models[0].components()
        n_ctx = next((c.n_ctx for c in comps if hasattr(c, "n_ctx")), 1)
        d = comps[0].network().ind_size("in:d0") - 1
        like = dict(
            device=next(models[0].parameters()).device,
            dtype=next(models[0].parameters()).dtype,
        )
        S = torch.eye(n_ctx * (d + 1), **like).reshape(n_ctx, d + 1, n_ctx, d + 1)
        S = S.unsqueeze(0).repeat(len(models), 1, 1, 1, 1)
        return StateBatched(S, S.clone(), S.clone())


def _as_component(model) -> AttentionLMComponent:
    if isinstance(model, AttentionLMComponent):
        return model
    _validate_model_for_tn_similarity(model)
    return AttentionLMComponent.from_trained_model(model)


def _validate_batch_compatibility(
    models_a: Sequence[AttentionLMComponent],
    models_b: Sequence[AttentionLMComponent],
) -> None:
    if len(models_a) != len(models_b):
        raise ValueError("Batch sizes must match for TN similarity batching.")
    if not models_a:
        raise ValueError("Batch must contain at least one model pair.")

    base_a = models_a[0]
    base_b = models_b[0]
    _validate_model_compatibility(base_a, base_b)

    for model in models_a[1:]:
        _validate_model_compatibility(base_a, model)
    for model in models_b[1:]:
        _validate_model_compatibility(base_b, model)


def _stack_tensors(tensors: Sequence[Tensor]) -> Tensor:
    base = tensors[0]
    for t in tensors[1:]:
        if t.inds != base.inds:
            raise ValueError("Tensor indices do not match for batching.")
        if t.tags != base.tags:
            raise ValueError("Tensor tags do not match for batching.")
    data = torch.stack([t.data for t in tensors], dim=0)
    return Tensor(data, inds=("b",) + tuple(base.inds), tags=base.tags)


def _batch_terms(
    components: Sequence[AttentionLMComponent],
    n_ctx: int,
    like: dict,
) -> list[Term]:
    terms_by_component = [component.terms(n_ctx, **like) for component in components]
    n_terms = len(terms_by_component[0])
    for terms in terms_by_component[1:]:
        if len(terms) != n_terms:
            raise ValueError("Component terms do not align for batching.")

    batched_terms: list[Term] = []
    for term_idx in range(n_terms):
        term_group = [terms[term_idx] for terms in terms_by_component]
        base_legs = term_group[0].legs
        for term in term_group[1:]:
            if term.legs != base_legs:
                raise ValueError("Term leg mappings do not match across batch.")

        tensor_lists = [list(term.tn) for term in term_group]
        n_tensors = len(tensor_lists[0])
        for tensor_list in tensor_lists[1:]:
            if len(tensor_list) != n_tensors:
                raise ValueError("Term tensor counts do not match across batch.")

        stacked_tensors = [
            _stack_tensors([tensor_list[idx] for tensor_list in tensor_lists])
            for idx in range(n_tensors)
        ]
        batched_terms.append(
            Term(TensorNetwork(stacked_tensors, check_collisions=False), base_legs)
        )

    return batched_terms


def _matchings(legs: list[tuple[str, str, str]]):
    """All perfect matchings (Wick pairings) of a list of legs."""
    if not legs:
        return [()]
    return [
        ((legs[0], legs[i]),) + rest
        for i in range(1, len(legs))
        for rest in _matchings(legs[1:i] + legs[i + 1 :])
    ]


def _contract_batch(tn: TensorNetwork, bridges: Iterable[Tensor], output_inds: tuple[str, ...]):
    """Contract TN with batched bridge tensors, caching by index structure."""
    batch_size = tn.ind_size("b")
    key = (
        batch_size,
        tuple(t.inds for t in tn),
        tuple(b.inds for b in bridges),
    )
    if key not in _EXPR_CACHE_BATCH:
        full = tn.copy()
        for b in bridges:
            full &= b
        _EXPR_CACHE_BATCH[key] = full.contract(
            output_inds=output_inds,
            optimize="greedy",
            get="expression",
        )
    return _EXPR_CACHE_BATCH[key](
        *(t.data.detach() for t in tn),
        *(b.data.detach() for b in bridges),
    )


def _isserlis_batch(tn, legs, S_for_pair, mu_for_leg, output_inds):
    """Batched Isserlis contraction with overcounting correction."""

    def bridge(l1, l2):
        return Tensor(S_for_pair(l1, l2), inds=("b",) + l1[:2] + l2[:2])

    result = sum(
        _contract_batch(tn, [bridge(l1, l2) for l1, l2 in matching], output_inds)
        for matching in _matchings(legs)
    )

    mu_bridges = [Tensor(mu_for_leg(leg), inds=("b",) + leg[:2]) for leg in legs]
    n_matchings = prod(range(1, len(legs), 2))
    return result - (n_matchings - 1) * _contract_batch(tn, mu_bridges, output_inds)


def _second_moment_batch(term_a: Term, term_b: Term, state: StateBatched):
    """E[term_a(x) term_b(x)^T] via doubled TN with batched Isserlis bridges."""

    def prefix(tn, p):
        mapping = {i: f"{p}:{i}" for i in tn.ind_map if i != "b"}
        return tn.reindex(mapping)

    tn = TensorNetwork(
        list(prefix(term_a.tn, "a")) + list(prefix(term_b.tn, "b")),
        check_collisions=False,
    )

    legs = [
        (f"{p}:{pos}", f"{p}:{data}", p)
        for term, p in [(term_a, "a"), (term_b, "b")]
        for data, pos in sorted(term.legs.items())
    ]

    S_map = {
        "aa": state.S_aa,
        "ab": state.S_ab,
        "ba": state.S_ab,
        "bb": state.S_bb,
    }
    mu_map = {
        "a": state.S_aa[:, :, 0, 0],
        "b": state.S_bb[:, :, 0, 0],
    }

    return _isserlis_batch(
        tn,
        legs,
        S_for_pair=lambda l1, l2: S_map[l1[2] + l2[2]],
        mu_for_leg=lambda l: mu_map[l[2]],
        output_inds=("b", "a:out:s", "a:out:d", "b:out:s", "b:out:d"),
    )


def propagate_batch(state: StateBatched, terms_a: list[Term], terms_b: list[Term]) -> StateBatched:
    """Propagate second-moment state through one layer for batched model pairs."""

    def second_moments(terms_left, terms_right, s):
        return sum(_second_moment_batch(a, b, s) for a in terms_left for b in terms_right)

    return StateBatched(
        second_moments(terms_a, terms_a, StateBatched(state.S_aa, state.S_aa, state.S_aa)),
        second_moments(terms_b, terms_b, StateBatched(state.S_bb, state.S_bb, state.S_bb)),
        second_moments(terms_a, terms_b, state),
    )


def similarity_batch(
    models_a: Sequence[AttentionLMComponent],
    models_b: Sequence[AttentionLMComponent],
) -> StateBatched:
    """Compute batched exact Gaussian functional similarity."""
    state = StateBatched.from_models(models_a)
    n_ctx = state.S_aa.shape[1]
    like = dict(device=state.S_aa.device, dtype=state.S_aa.dtype)

    components_a = [model.components() for model in models_a]
    components_b = [model.components() for model in models_b]
    n_components = len(components_a[0])

    for comps in components_a[1:]:
        if len(comps) != n_components:
            raise ValueError("Model components do not align for batching.")
    for comps in components_b:
        if len(comps) != n_components:
            raise ValueError("Model components do not align for batching.")

    for idx in range(n_components):
        terms_a = _batch_terms([comp[idx] for comp in components_a], n_ctx, like)
        terms_b = _batch_terms([comp[idx] for comp in components_b], n_ctx, like)
        state = propagate_batch(state, terms_a, terms_b)

    return state


def compute_tn_similarity_batch(
    models_a: Sequence,
    models_b: Sequence,
    device: str | None = None,
) -> StateBatched:
    """Compute batched TN similarity for lists of model pairs. Runs in the
    models' native dtype — no conversion."""
    comp_a = [_as_component(model) for model in models_a]
    comp_b = [_as_component(model) for model in models_b]

    _validate_batch_compatibility(comp_a, comp_b)

    if device is not None:
        comp_a = [model.to(device=device) for model in comp_a]
        comp_b = [model.to(device=device) for model in comp_b]

    return similarity_batch(comp_a, comp_b)


def _cosine_from_state_batch(state: StateBatched) -> torch.Tensor:
    def trace(S):
        return torch.einsum("bijij->b", S[:, :, 1:, :, 1:])

    tr_aa = trace(state.S_aa)
    tr_bb = trace(state.S_bb)
    tr_ab = trace(state.S_ab)

    denom = (tr_aa * tr_bb).sqrt()
    sims = torch.zeros_like(denom)
    mask = denom >= 1e-30
    sims[mask] = tr_ab[mask] / denom[mask]
    return sims


def cosine_similarity_batch(
    models_a: Sequence,
    models_b: Sequence,
    device: str | None = None,
) -> list[float]:
    """Compute cosine similarity for batched model pairs."""
    state = compute_tn_similarity_batch(models_a, models_b, device=device)
    return _cosine_from_state_batch(state).cpu().tolist()
