import torch
from math import prod
from dataclasses import dataclass
from quimb.tensor import Tensor, TensorNetwork

_EXPR_CACHE = {}


@dataclass
class State:
    """Second-moment state S = E[ff^T] in (s, d, s, d) layout for a pair of models."""
    S_aa: torch.Tensor  # (n_ctx, d+1, n_ctx, d+1)
    S_bb: torch.Tensor
    S_ab: torch.Tensor

    @staticmethod
    def from_model(model):
        comps = model.components()
        n_ctx = next((c.n_ctx for c in comps if hasattr(c, 'n_ctx')), 1)
        d = comps[0].network().ind_size('in:d0') - 1
        like = dict(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
        S = torch.eye(n_ctx * (d + 1), **like).reshape(n_ctx, d + 1, n_ctx, d + 1)
        return State(S, S.clone(), S.clone())



def _matchings(legs):
    """All perfect matchings (Wick pairings) of a list of legs."""
    if not legs:
        return [()]
    return [
        ((legs[0], legs[i]),) + rest
        for i in range(1, len(legs))
        for rest in _matchings(legs[1:i] + legs[i+1:])
    ]


def _contract(tn, bridges, output_inds):
    """Contract TN with bridge tensors. Caches the contraction expression by index structure."""
    key = tuple(t.inds for t in tn) + tuple(b.inds for b in bridges)
    if key not in _EXPR_CACHE:
        full = tn.copy()
        for b in bridges:
            full &= b
        _EXPR_CACHE[key] = full.contract(output_inds=output_inds, optimize='greedy', get='expression')
    return _EXPR_CACHE[key](*(t.data for t in tn), *(b.data for b in bridges))


def _isserlis(tn, legs, S_for_pair, mu_for_leg, output_inds):
    """Corrected Isserlis: Σ_matchings Π S[paired legs] - (n-1) × μ product.

    Each leg is (pos_index, data_index, model_tag).
    TN bridge indices are leg[:2], model selection uses leg[2].
    """
    def bridge(l1, l2):
        return Tensor(S_for_pair(l1, l2), inds=l1[:2] + l2[:2])

    result = sum(
        _contract(tn, [bridge(l1, l2) for l1, l2 in matching], output_inds)
        for matching in _matchings(legs))

    mu_bridges = [Tensor(mu_for_leg(leg), inds=leg[:2]) for leg in legs]
    n_matchings = prod(range(1, len(legs), 2))
    return result - (n_matchings - 1) * _contract(tn, mu_bridges, output_inds)


def _second_moment(term_a, term_b, state):
    """E[term_a(x) term_b(x)^T] via doubled TN with Isserlis bridges."""
    prefix = lambda tn, p: tn.reindex({i: f'{p}:{i}' for i in tn.ind_map})
    tn = TensorNetwork(
        list(prefix(term_a.tn, 'a')) + list(prefix(term_b.tn, 'b')),
        check_collisions=False)

    legs = [(f'{p}:{pos}', f'{p}:{data}', p)
            for term, p in [(term_a, 'a'), (term_b, 'b')]
            for data, pos in sorted(term.legs.items())]

    S_map = {'aa': state.S_aa, 'ab': state.S_ab, 'ba': state.S_ab, 'bb': state.S_bb}
    mu_map = {'a': state.S_aa[:, :, 0, 0], 'b': state.S_bb[:, :, 0, 0]}

    return _isserlis(
        tn, legs,
        S_for_pair=lambda l1, l2: S_map[l1[2] + l2[2]],
        mu_for_leg=lambda l: mu_map[l[2]],
        output_inds=('a:out:s', 'a:out:d', 'b:out:s', 'b:out:d'))


def propagate(state, ca, cb):
    """Propagate second-moment state through one layer for both models."""
    n_ctx = state.S_aa.shape[0]
    like = dict(device=state.S_aa.device, dtype=state.S_aa.dtype)
    terms_a = ca.terms(n_ctx, **like)
    terms_b = cb.terms(n_ctx, **like)

    def second_moments(terms_left, terms_right, s):
        return sum(_second_moment(a, b, s) for a in terms_left for b in terms_right)

    return State(
        second_moments(terms_a, terms_a, State(state.S_aa, state.S_aa, state.S_aa)),
        second_moments(terms_b, terms_b, State(state.S_bb, state.S_bb, state.S_bb)),
        second_moments(terms_a, terms_b, state))


def similarity(model_a, model_b):
    """Compute exact Gaussian functional similarity between two models."""
    state = State.from_model(model_a)
    for ca, cb in zip(model_a.components(), model_b.components()):
        state = propagate(state, ca, cb)
    return state
