"""Tests for ICL score computation."""
import pytest
import torch
from torch.utils.data import DataLoader

from analysis.behaviour.icl import compute_token_k_losses, compute_icl_score
from analysis.behaviour.tracker import BehaviourTracker, TrackerConfig
from tests.analysis.conftest import V, DictDataset, _collate_fn


# ---------------------------------------------------------------------------
# Fixtures – sequences long enough for the test k values
# ---------------------------------------------------------------------------

K1, K2 = 3, 10  # small values that fit in tiny models


@pytest.fixture
def long_dataloader():
    """Dataloader with sequences of length 20 (enough for k2=10)."""
    input_ids = torch.randint(0, V, (40, 20))
    dataset = DictDataset(input_ids)
    return DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=_collate_fn)


@pytest.fixture
def padded_long_dataloader():
    """Dataloader with padding; only 12 real tokens per row."""
    seq_len = 20
    real_len = 12
    input_ids = torch.full((40, seq_len), 99, dtype=torch.long)  # pad=99
    attention_mask = torch.zeros(40, seq_len, dtype=torch.long)
    input_ids[:, :real_len] = torch.randint(2, V, (40, real_len))
    attention_mask[:, :real_len] = 1
    dataset = DictDataset(input_ids, attention_mask)
    return DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=_collate_fn)


@pytest.fixture
def short_dataloader():
    """Dataloader where sequences are too short for k2=10."""
    input_ids = torch.randint(0, V, (20, 5))
    dataset = DictDataset(input_ids)
    return DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=_collate_fn)


@pytest.fixture
def icl_model():
    """Small model with n_ctx >= 20."""
    from models import AttentionLM
    torch.manual_seed(42)
    cfg = {
        "model": {
            "vocab_size": V,
            "n_ctx": 32,
            "d_model": 32,
            "n_head": 4,
            "n_layers": 2,
            "attn_scale": 0.2,
            "rope_base": 10000,
            "use_rmsnorm_qk": False,
            "use_bias_qk": True,
        }
    }
    return AttentionLM.from_config(cfg)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_compute_token_k_losses_returns_finite(icl_model, long_dataloader):
    l_k1, l_k2 = compute_token_k_losses(
        icl_model, long_dataloader, k1=K1, k2=K2, bos_token_id=0,
    )
    assert isinstance(l_k1, float)
    assert isinstance(l_k2, float)
    assert l_k1 > 0
    assert l_k2 > 0


def test_compute_icl_score_keys(icl_model, long_dataloader):
    result = compute_icl_score(
        icl_model, long_dataloader, bos_token_id=0, k1=K1, k2=K2,
    )
    assert f"loss_{K1}" in result
    assert f"loss_{K2}" in result
    assert f"icl_{K1}_{K2}" in result
    # icl = l_k2 - l_k1
    assert abs(result[f"icl_{K1}_{K2}"] - (result[f"loss_{K2}"] - result[f"loss_{K1}"])) < 1e-6


def test_icl_score_with_padding(icl_model, padded_long_dataloader):
    """Padded sequences with 12 real tokens should still work for k2=10."""
    result = compute_icl_score(
        icl_model, padded_long_dataloader, bos_token_id=0, k1=K1, k2=K2,
    )
    assert result[f"loss_{K1}"] > 0
    assert result[f"loss_{K2}"] > 0


def test_icl_score_short_sequences_returns_nan(icl_model, short_dataloader):
    """When no sequence has >= k2 tokens, losses should be nan."""
    result = compute_icl_score(
        icl_model, short_dataloader, bos_token_id=0, k1=K1, k2=K2,
    )
    import math
    assert math.isnan(result[f"loss_{K1}"])
    assert math.isnan(result[f"loss_{K2}"])
    assert math.isnan(result[f"icl_{K1}_{K2}"])


def test_icl_max_batches(icl_model, long_dataloader):
    """max_batches limits how many batches are processed."""
    r_all = compute_icl_score(
        icl_model, long_dataloader, bos_token_id=0, k1=K1, k2=K2,
    )
    r_one = compute_icl_score(
        icl_model, long_dataloader, bos_token_id=0, k1=K1, k2=K2,
        max_batches=1,
    )
    # Both should be valid but may differ
    assert r_one[f"loss_{K1}"] > 0
    assert r_one[f"loss_{K2}"] > 0


def test_k2_must_be_greater_than_k1(icl_model, long_dataloader):
    with pytest.raises(AssertionError):
        compute_token_k_losses(
            icl_model, long_dataloader, k1=10, k2=3, bos_token_id=0,
        )


# ---------------------------------------------------------------------------
# Tracker integration
# ---------------------------------------------------------------------------

def test_tracker_compute_metrics_includes_icl(icl_model, long_dataloader):
    """ICL metrics should appear in tracker output when enabled."""
    config = TrackerConfig(
        bigram_enabled=False,
        ngram_enabled=False,
        ablation_enabled=False,
        icl_enabled=True,
        icl_compute_every=1,
        icl_k1=K1,
        icl_k2=K2,
    )
    tracker = BehaviourTracker(
        model=icl_model,
        train_dataloader=long_dataloader,
        val_dataloader=long_dataloader,
        vocab_size=V,
        config=config,
    )
    tracker._is_fitted = True

    metrics = tracker.compute_metrics(step=1)
    assert f"loss_{K1}" in metrics
    assert f"loss_{K2}" in metrics
    assert f"icl_{K1}_{K2}" in metrics


def test_tracker_should_compute_icl():
    config = TrackerConfig(
        bigram_enabled=False,
        ngram_enabled=False,
        ablation_enabled=False,
        icl_enabled=True,
        icl_compute_every=10,
    )
    tracker = BehaviourTracker.__new__(BehaviourTracker)
    tracker.config = config
    assert tracker.should_compute(10)
    assert not tracker.should_compute(7)
