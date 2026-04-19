"""Pin the public API documented in ``tn_sim/README.md``.

If any import listed below breaks, either the README, ``__init__.py``, or
the underlying implementation has drifted. Fix whichever is wrong; do not
weaken the assertion.
"""
import tn_sim


PUBLIC_API = (
    "compute_tn_similarity",
    "cosine_similarity",
    "inner_product",
    "self_similarity",
    "mc_similarity",
    "mc_similarity_gaussian_tokens",
    "random_sim",
)


def test_public_names_importable():
    for name in PUBLIC_API:
        assert hasattr(tn_sim, name), f"tn_sim.{name} missing"


def test_all_matches_public_api():
    assert set(tn_sim.__all__) == set(PUBLIC_API), (
        f"tn_sim.__all__ = {sorted(tn_sim.__all__)!r} drifted from documented "
        f"public API {sorted(PUBLIC_API)!r}"
    )
