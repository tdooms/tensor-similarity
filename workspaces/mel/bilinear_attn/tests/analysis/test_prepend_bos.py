"""Tests for the ``prepend_bos`` flag added to ICL / N-gram metrics.

These guard the fix for Pile-DSIR contiguous training, where the model
never sees a BOS at position 0 and BOS-prepending inflates short-context
losses.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from analysis.behaviour.icl import compute_icl_score, compute_token_k_losses
from analysis.behaviour.ngram import NgramAnalyzer
from tests.analysis.conftest import V, DictDataset, _collate_fn


K1, K2 = 3, 10


@pytest.fixture
def long_dataloader():
    torch.manual_seed(0)
    input_ids = torch.randint(2, V, (16, 20))
    return DataLoader(
        DictDataset(input_ids), batch_size=4, shuffle=False, collate_fn=_collate_fn
    )


@pytest.fixture
def icl_model():
    from models import AttentionLM
    torch.manual_seed(42)
    # Large init so logits are non-trivial; default init yields near-uniform
    # outputs on a random model, which makes two different inputs produce
    # identical CE down to float64 precision.
    return AttentionLM.from_config({
        "model": {
            "vocab_size": V, "n_ctx": 32, "d_model": 32, "n_head": 4,
            "n_layers": 2, "attn_scale": 0.2, "rope_base": 10000,
            "use_rmsnorm_qk": False, "use_bias_qk": True,
        },
        "init": {"std_embed": 0.5, "std_qkv": 0.5, "std_o": 0.5},
    }).eval()


def test_icl_prepend_bos_changes_losses(icl_model, long_dataloader):
    """Turning off BOS-prepending must produce a different loss trajectory."""
    l_with = compute_token_k_losses(
        icl_model, long_dataloader, k1=K1, k2=K2, bos_token_id=0,
        prepend_bos=True,
    )
    l_without = compute_token_k_losses(
        icl_model, long_dataloader, k1=K1, k2=K2, bos_token_id=0,
        prepend_bos=False,
    )
    assert all(math.isfinite(x) for x in (*l_with, *l_without))
    # At least one of (l_k1, l_k2) must differ: feeding a BOS shifts the
    # distribution even for a randomly initialised model.
    assert (abs(l_with[0] - l_without[0]) > 1e-6
            or abs(l_with[1] - l_without[1]) > 1e-6)


def test_icl_no_bos_matches_manual_indexing(icl_model, long_dataloader):
    """When prepend_bos=False the loss at position k must match a manual
    forward pass on ``x_1..x_{k-1}``."""
    l_k1, l_k2 = compute_token_k_losses(
        icl_model, long_dataloader, k1=K1, k2=K2, bos_token_id=0,
        prepend_bos=False,
    )

    # Manual recomputation on the full dataset (same order, same batches).
    total1, total2, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in long_dataloader:
            input_ids = batch["input_ids"]
            inp = input_ids[:, : K2 - 1]
            logits = icl_model(inp)
            t1 = input_ids[:, K1 - 1]
            t2 = input_ids[:, K2 - 1]
            total1 += F.cross_entropy(logits[:, K1 - 2], t1, reduction="sum").item()
            total2 += F.cross_entropy(logits[:, K2 - 2], t2, reduction="sum").item()
            n += input_ids.shape[0]
    assert math.isclose(l_k1, total1 / n, rel_tol=1e-5, abs_tol=1e-5)
    assert math.isclose(l_k2, total2 / n, rel_tol=1e-5, abs_tol=1e-5)


def test_icl_no_bos_rejects_k1_lt_2(icl_model, long_dataloader):
    with pytest.raises(AssertionError):
        compute_token_k_losses(
            icl_model, long_dataloader, k1=1, k2=K2, bos_token_id=0,
            prepend_bos=False,
        )


def test_icl_score_keys_preserved(icl_model, long_dataloader):
    result = compute_icl_score(
        icl_model, long_dataloader, bos_token_id=0, k1=K1, k2=K2,
        prepend_bos=False,
    )
    assert f"loss_{K1}" in result and f"loss_{K2}" in result
    assert f"icl_{K1}_{K2}" in result
    assert math.isclose(
        result[f"icl_{K1}_{K2}"],
        result[f"loss_{K2}"] - result[f"loss_{K1}"],
        abs_tol=1e-6,
    )


# ---------------------------------------------------------------------------
# N-gram
# ---------------------------------------------------------------------------

@pytest.fixture
def ngram_analyzer(long_dataloader):
    ana = NgramAnalyzer(vocab_size=V, device="cpu", max_common_ngrams=20)
    ana.extract_common_ngrams_from_data(
        long_dataloader, tokenizer=None, max_n=3, max_samples=64,
    )
    return ana


def test_ngram_test_loss_no_bos_matches_manual(icl_model, long_dataloader, ngram_analyzer):
    n = 3
    loss = ngram_analyzer.compute_test_loss(
        icl_model, long_dataloader, n=n, bos_token_id=0, prepend_bos=False,
    )

    total, count = 0.0, 0
    with torch.no_grad():
        for batch in long_dataloader:
            input_ids = batch["input_ids"]
            first_n = input_ids[:, :n]
            inp = first_n[:, :-1]
            logits = icl_model(inp)
            total += F.cross_entropy(
                logits[:, -1], first_n[:, -1], reduction="sum"
            ).item()
            count += input_ids.shape[0]
    assert math.isclose(loss, total / count, rel_tol=1e-5, abs_tol=1e-5)


def test_ngram_test_loss_no_bos_rejects_n1(icl_model, long_dataloader, ngram_analyzer):
    with pytest.raises(ValueError):
        ngram_analyzer.compute_test_loss(
            icl_model, long_dataloader, n=1, bos_token_id=0, prepend_bos=False,
        )


def test_ngram_loss_differs_with_and_without_bos(icl_model, ngram_analyzer):
    with_bos = ngram_analyzer.compute_ngram_loss(
        icl_model, n=3, bos_token_id=0, prepend_bos=True,
    )
    without = ngram_analyzer.compute_ngram_loss(
        icl_model, n=3, bos_token_id=0, prepend_bos=False,
    )
    assert math.isfinite(with_bos) and math.isfinite(without)
    assert abs(with_bos - without) > 1e-6
