import torch
from dataclasses import dataclass
from itertools import combinations
from quimb.tensor import Tensor, TensorNetwork

_EXPR_CACHE = {}


@dataclass
class State:
    """Gaussian moment state: (μ, Σ) for a pair of models."""
    mu_a: torch.Tensor      # (S, d+1)
    mu_b: torch.Tensor
    sigma_aa: torch.Tensor  # (S, S, d+1, d+1)
    sigma_bb: torch.Tensor
    sigma_ab: torch.Tensor

    @staticmethod
    def default(d, n_ctx, **like):
        mu = torch.zeros(n_ctx, d + 1, **like)
        mu[:, 0] = 1
        diag = torch.diag(torch.cat([torch.zeros(1, **like), torch.ones(d, **like)]))
        sigma = torch.zeros(n_ctx, n_ctx, d + 1, d + 1, **like)
        sigma[range(n_ctx), range(n_ctx)] = diag
        return State(mu, mu.clone(), sigma, sigma.clone(), sigma.clone())

    def self_a(self):
        return State(self.mu_a, self.mu_a, self.sigma_aa, self.sigma_aa, self.sigma_aa)

    def self_b(self):
        return State(self.mu_b, self.mu_b, self.sigma_bb, self.sigma_bb, self.sigma_bb)

    def inner_product(self):
        s = torch.arange(self.sigma_ab.shape[0])
        return ((self.mu_a[:, 1:] * self.mu_b[:, 1:]).sum() +
                self.sigma_ab[s, s, 1:, 1:].diagonal(dim1=-2, dim2=-1).sum()).item()

    def cosine(self):
        ip = self.inner_product()
        return ip / (self.self_a().inner_product() * self.self_b().inner_product()) ** 0.5


@dataclass(frozen=True)
class Leg:
    data: str
    pos: str

    @property
    def inds(self):
        return (self.pos, self.data)

    @property
    def model(self):
        return self.data.split(':')[0]


# --- Utilities ---

def _matchings(legs):
    if not legs:
        return [()]
    return [
        ((legs[0], legs[i]),) + rest
        for i in range(1, len(legs))
        for rest in _matchings(legs[1:i] + legs[i+1:])
    ]


def _prefix(tn, p):
    return tn.reindex({idx: f'{p}:{idx}' for idx in tn.ind_map})


def _double(tn_a, tn_b):
    return TensorNetwork(
        list(_prefix(tn_a, 'a')) + list(_prefix(tn_b, 'b')),
        check_collisions=False,
    )


def _bridge(sigma, l1, l2):
    """Bridge tensor: diagonal when same position, full when different."""
    if l1.pos == l2.pos:
        return (l1.pos, l1.data, l2.data), sigma[range(sigma.shape[0]), range(sigma.shape[0])]
    return (l1.pos, l1.data, l2.pos, l2.data), sigma.permute(0, 2, 1, 3)


# --- Core ---

def _partitions(legs, evaluable=None):
    """All (mu_legs, sigma_matching) partitions. Only evaluable legs can be μ-evaluated."""
    N = len(legs)
    ev = evaluable if evaluable is not None else set(range(N))
    for k in range(0, len(ev) + 1):
        if (N - k) % 2:
            continue
        for mu_idx in combinations(sorted(ev), k):
            sigma_legs = [legs[i] for i in range(N) if i not in mu_idx]
            for matching in _matchings(sigma_legs):
                yield [legs[i] for i in mu_idx], matching


def _isserlis(tn, legs, mu_fn, sigma_fn, output_inds, evaluable=None):
    """Generalized Isserlis: sum over all μ/Σ partitions of legs.

    Caches compiled contraction expressions by index structure for reuse
    across calls with different tensor data but identical topology.
    """
    base_inds = tuple(tensor.inds for tensor in tn)
    base_data = tuple(tensor.data for tensor in tn)

    jobs = []
    for mu_legs, matching in _partitions(legs, evaluable):
        mus = [(leg, mu_fn(leg)) for leg in mu_legs]
        if any(not mu.any() for _, mu in mus):
            continue

        extra_inds = tuple(leg.inds for leg, _ in mus)
        extra_data = tuple(mu for _, mu in mus)
        for l1, l2 in matching:
            b_inds, b_data = _bridge(sigma_fn(l1, l2), l1, l2)
            extra_inds += (b_inds,)
            extra_data += (b_data,)

        key = base_inds + extra_inds
        if key not in _EXPR_CACHE:
            t = tn.copy()
            for inds, data in zip(extra_inds, extra_data):
                t &= Tensor(data, inds=inds)
            _EXPR_CACHE[key] = t.contract(output_inds=output_inds, optimize='greedy', get='expression')
        jobs.append((_EXPR_CACHE[key], base_data + extra_data))

    return sum(expr(*arrays) for expr, arrays in jobs)


def _build_legs(term, p):
    input_data = sorted(idx for idx in term.tn.outer_inds() if idx.startswith('in:d'))
    return [Leg(f'{p}:{idx}', pos=f'{p}:{term.legs[idx]}') for idx in input_data]


def _second_moment(term_a, term_b, state):
    tn = _double(term_a.tn, term_b.tn)
    legs_a = _build_legs(term_a, 'a')
    legs_b = _build_legs(term_b, 'b')
    legs = legs_a + legs_b

    evaluable = {i for i in range(len(legs_a)) if not term_a.zero_mean}
    evaluable |= {len(legs_a) + i for i in range(len(legs_b)) if not term_b.zero_mean}

    result = _isserlis(
        tn, legs,
        mu_fn=lambda leg: state.mu_b if leg.model == 'b' else state.mu_a,
        sigma_fn=lambda l1, l2: (
            state.sigma_aa if l1.model == 'a' and l2.model == 'a' else
            state.sigma_bb if l1.model == 'b' and l2.model == 'b' else
            state.sigma_ab
        ),
        output_inds=('a:out:s', 'a:out:d', 'b:out:s', 'b:out:d'),
        evaluable=evaluable,
    )
    return result.permute(0, 2, 1, 3)


def _mean(term, mu, sigma):
    legs = [Leg(idx, pos=term.legs[idx])
            for idx in sorted(term.tn.outer_inds()) if idx.startswith('in:d')]
    return _isserlis(
        term.tn, legs,
        mu_fn=lambda _: mu,
        sigma_fn=lambda _, __: sigma,
        output_inds=('out:s', 'out:d'),
    )


def _outer(mu_a, mu_b):
    return torch.einsum('si,tj->stij', mu_a, mu_b)


# --- Propagation ---

def propagate(state, ca, cb, n_ctx, **like):
    terms_a = ca.terms(n_ctx, **like)
    terms_b = cb.terms(n_ctx, **like)

    sm_aa = sum(_second_moment(a, a2, state.self_a()) for a in terms_a for a2 in terms_a)
    sm_bb = sm_aa if ca is cb else sum(_second_moment(b, b2, state.self_b()) for b in terms_b for b2 in terms_b)
    sm_ab = sum(_second_moment(a, b, state) for a in terms_a for b in terms_b)

    mu_a = sum(_mean(t, state.mu_a, state.sigma_aa) for t in terms_a)
    mu_b = mu_a if ca is cb else sum(_mean(t, state.mu_b, state.sigma_bb) for t in terms_b)

    return State(mu_a, mu_b,
                 sm_aa - _outer(mu_a, mu_a),
                 sm_bb - _outer(mu_b, mu_b),
                 sm_ab - _outer(mu_a, mu_b))


def similarity(model_a, model_b):
    comps_a = model_a.components()
    comps_b = model_b.components()

    n_ctx = next((c.n_ctx for c in comps_a if hasattr(c, 'n_ctx')), 1)
    d = comps_a[0].network().ind_size('in:d0') - 1
    t0 = next(iter(comps_a[0].network())).data
    like = dict(device=t0.device, dtype=t0.dtype)

    state = State.default(d, n_ctx, **like)
    for ca, cb in zip(comps_a, comps_b):
        state = propagate(state, ca, cb, n_ctx, **like)
    return state
