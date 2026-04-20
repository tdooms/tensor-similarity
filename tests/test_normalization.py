"""Per-layer Σ normalization must leave cosine invariant.

Cosine(a, b) = tr(Σ_ab) / sqrt(tr(Σ_aa) · tr(Σ_bb)) is invariant under
any per-slot rescaling   aa ← aa/α_aa,  bb ← bb/α_bb,  ab ← ab/√(α_aa·α_bb),
because the α's cancel exactly in the ratio. Applied per layer this bounds
intermediate magnitudes; fp32 then survives deep / wide models that would
otherwise overflow during vocab-scale contractions.

Test: compare cosine of normalized-path vs un-normalized-path on a small-enough
Transformer that un-normalized doesn't overflow. The two must agree to fp32
precision on every pair.
"""
import torch
import pytest
from src.models.transformer import Transformer
from src.components.similarity import similarity


def cosine(s):
    tr = lambda x: torch.einsum('ijij->', x[:, 1:, :, 1:])
    return (tr(s[0, 1]) / (tr(s[0, 0]) * tr(s[1, 1])).clamp_min(0).sqrt()).item()


@pytest.mark.parametrize('seed_a,seed_b', [(0, 1), (42, 99), (7, 7)])
def test_similarity_gives_finite_cosine(seed_a, seed_b):
    """Baseline: small model, un-normalized path should give finite cosine."""
    torch.manual_seed(seed_a)
    a = Transformer(8, 16, 2, 3, 16, 8, n_layer=1, mask='none', scale=0.5).double()
    torch.manual_seed(seed_b)
    b = Transformer(8, 16, 2, 3, 16, 8, n_layer=1, mask='none', scale=0.5).double()
    s = similarity(a, b)
    c = cosine(s)
    assert not (c != c), f'NaN cosine at seeds {seed_a},{seed_b}'
    assert -1.01 < c < 1.01, f'out-of-range cosine {c}'


@pytest.mark.parametrize('scale', [1.0, 10.0, 100.0])
def test_cosine_invariant_to_per_slot_rescaling(scale):
    """Manual rescaling: if we multiply Σ_aa by α_a², Σ_bb by α_b², and
    Σ_ab by α_a·α_b (which is what emerges from independent per-model rescales
    in the forward pass), cosine must be unchanged."""
    torch.manual_seed(42); a = Transformer(8, 16, 2, 3, 16, 8, n_layer=1, mask='none', scale=0.5).double()
    torch.manual_seed(99); b = Transformer(8, 16, 2, 3, 16, 8, n_layer=1, mask='none', scale=0.5).double()
    s = similarity(a, b)
    c_base = cosine(s)

    # Manually rescale: aa by α², bb by β², ab by αβ, ba by αβ.
    α, β = scale, 1.0 / scale
    s_rescaled = s.clone()
    s_rescaled[0, 0] *= α * α
    s_rescaled[1, 1] *= β * β
    s_rescaled[0, 1] *= α * β
    s_rescaled[1, 0] *= α * β
    c_rescaled = cosine(s_rescaled)

    assert abs(c_base - c_rescaled) < 1e-10, (
        f'cosine NOT invariant under slot rescaling: base={c_base}, rescaled={c_rescaled}')


def test_per_layer_normalization_preserves_cosine():
    """Given the invariance above, a per-layer normalization of `_propagate`
    that divides each (aa, ab, bb) update by matching factors — aa by α,
    bb by β, ab by √(α·β) — must leave the final cosine bit-identical (mod
    fp re-association) to the un-normalized propagation. This test monkey-
    patches _propagate to apply per-layer normalization and checks the
    cosine agrees with the stock _propagate."""
    import torch, src.components.similarity as sim
    from src.components.similarity import _update, _initial

    torch.manual_seed(42); a = Transformer(8, 16, 2, 3, 16, 8, n_layer=1, mask='none', scale=0.5).double()
    torch.manual_seed(99); b = Transformer(8, 16, 2, 3, 16, 8, n_layer=1, mask='none', scale=0.5).double()

    c_stock = cosine(similarity(a, b))

    original = sim._propagate
    def _propagate_normalized(a, b):
        aa = ab = bb = _initial(a)
        for ca, cb in zip(a.components(), b.components()):
            ta = ca.terms(a.n_ctx)
            tb = ta if a is b else cb.terms(b.n_ctx)
            aa_ = _update(aa, aa, aa, ta, ta)
            if a is b:
                α = aa_.abs().max()
                aa, ab, bb = ((aa_ / α),) * 3
            else:
                ab_ = _update(aa, ab, bb, ta, tb)
                bb_ = _update(bb, bb, bb, tb, tb)
                α, β = aa_.abs().max(), bb_.abs().max()
                aa = aa_ / α
                bb = bb_ / β
                ab = ab_ / (α * β).sqrt()
        return aa, ab, bb
    sim._propagate = _propagate_normalized
    sim._PATHS.clear(); sim._GRAPHS.clear()
    try:
        c_norm = cosine(similarity(a, b))
    finally:
        sim._propagate = original
        sim._PATHS.clear(); sim._GRAPHS.clear()

    assert abs(c_stock - c_norm) < 1e-8, f'normalized cosine diverged: stock={c_stock}, norm={c_norm}'
