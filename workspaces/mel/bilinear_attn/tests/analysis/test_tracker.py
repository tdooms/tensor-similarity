"""Tests for behaviour tracker."""
from pathlib import Path

import pytest
import torch

from analysis.behaviour.tracker import BehaviourTracker, TrackerConfig
from tests.analysis.conftest import V, save_fake_checkpoint


_CHEAP_FIT = dict(
    bigram_compute_every=10, ngram_compute_every=10,
    bigram_n_samples=10, ngram_max_val_batches=5,
)


def test_tracker_config_defaults():
    c = TrackerConfig()
    assert c.bigram_enabled and c.ngram_enabled
    assert c.bigram_compute_every > 0 and c.ngram_compute_every > 0


def test_tracker_init(make_tracker):
    t = make_tracker()
    assert t.model is not None
    assert not t._is_fitted


def test_tracker_fit(make_tracker):
    t = make_tracker()
    t.fit(max_fit_samples=20)
    assert t._is_fitted
    assert t.bigram_analyzer is not None
    assert t.ngram_analyzer is not None


def test_tracker_compute_metrics(make_tracker):
    t = make_tracker(config=TrackerConfig(**_CHEAP_FIT))
    t.fit(max_fit_samples=20)
    metrics = t.compute_metrics(step=10)
    assert "step" in metrics
    assert "bigram_score" in metrics


def test_tracker_should_compute(make_tracker):
    t = make_tracker(config=TrackerConfig(
        bigram_compute_every=100, ngram_compute_every=200,
    ))
    assert t.should_compute(100)
    assert t.should_compute(200)
    assert not t.should_compute(50)


def test_tracker_log_metrics(make_tracker):
    t = make_tracker(config=TrackerConfig(**_CHEAP_FIT))
    t.fit(max_fit_samples=20)
    metrics = t.log_metrics(step=10, additional_metrics={"train_loss": 5.0})
    assert len(t.metrics_history) > 0
    assert "train_loss" in metrics


def test_tracker_toggle(make_tracker):
    t = make_tracker()
    assert t.config.bigram_enabled
    t.toggle_bigram(False)
    assert not t.config.bigram_enabled


def test_tracker_set_interval(make_tracker):
    t = make_tracker()
    t.set_compute_interval("bigram", 1000)
    assert t.config.bigram_compute_every == 1000


def test_tracker_get_metric_series(make_tracker):
    t = make_tracker(config=TrackerConfig(
        bigram_compute_every=10, ngram_enabled=False, bigram_n_samples=10,
    ))
    t.fit(max_fit_samples=20)
    t.log_metrics(step=10)
    t.log_metrics(step=20)
    steps, values = t.get_metric_series("bigram_score")
    assert len(steps) == 2 == len(values)


def test_tracker_save_load_history(make_tracker, tmp_path):
    t = make_tracker(config=TrackerConfig(
        bigram_compute_every=10, ngram_enabled=False, bigram_n_samples=10,
    ))
    t.fit(max_fit_samples=20)
    t.log_metrics(step=10)

    save_path = tmp_path / "test_history.json"
    t.save_history(str(save_path))
    assert save_path.exists()

    t2 = make_tracker()
    t2.load_history(str(save_path))
    assert len(t2.metrics_history) == len(t.metrics_history)


def test_tracker_cache_save_and_load(make_tracker, tmp_path):
    cache_dir = tmp_path / "cache"
    cfg = TrackerConfig(
        bigram_compute_every=10, ngram_compute_every=10,
        bigram_n_samples=10, ngram_max_n=4, ngram_max_val_batches=5,
    )
    t1 = make_tracker(config=cfg, run_dir=tmp_path / "run1", cache_dir=cache_dir)
    t1.fit(max_fit_samples=20)
    assert (cache_dir / "bigram.pt").exists()
    assert (cache_dir / "ngram.pt").exists()

    orig_bigram_total = t1.bigram_analyzer.total_bigrams
    orig_ngram_ns = sorted(t1.ngram_analyzer.common_ngrams.keys())

    t2 = make_tracker(config=cfg, run_dir=tmp_path / "run2", cache_dir=cache_dir)
    t2.fit(max_fit_samples=20)
    assert t2._is_fitted
    assert t2.bigram_analyzer.total_bigrams == orig_bigram_total
    assert sorted(t2.ngram_analyzer.common_ngrams.keys()) == orig_ngram_ns


def test_evaluate_checkpoint(make_tracker, model, tmp_path):
    ckpt = tmp_path / "step_100.pt"
    save_fake_checkpoint(model, ckpt, step=100)

    t = make_tracker(config=TrackerConfig(
        bigram_compute_every=10, ngram_compute_every=10,
        bigram_n_samples=10, ngram_max_n=3, ngram_max_val_batches=5,
    ))
    t.fit(max_fit_samples=20)

    metrics = t.evaluate_checkpoint(ckpt)
    assert metrics["step"] == 100
    assert {"checkpoint", "bigram_score", "bigram_entropy", "bigram_gap"} <= metrics.keys()
    assert isinstance(metrics["bigram_score"], float)


def test_evaluate_checkpoint_unfitted_raises(make_tracker, model, tmp_path):
    ckpt = tmp_path / "step_0.pt"
    save_fake_checkpoint(model, ckpt, step=0)
    t = make_tracker()
    with pytest.raises(RuntimeError, match="fit"):
        t.evaluate_checkpoint(ckpt)


def test_evaluate_checkpoints_from_dir(make_tracker, model, tmp_path):
    run_dir = tmp_path
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir()
    save_fake_checkpoint(model, ckpt_dir / "step_500.pt", step=500)
    save_fake_checkpoint(model, ckpt_dir / "step_1000.pt", step=1000)

    t = make_tracker(
        config=TrackerConfig(
            bigram_compute_every=10, ngram_enabled=False, bigram_n_samples=10,
        ),
        run_dir=run_dir,
    )
    t.fit(max_fit_samples=20)

    all_metrics = t.evaluate_checkpoints(run_dir=run_dir)
    assert [m["step"] for m in all_metrics] == [500, 1000]
    assert all("bigram_score" in m for m in all_metrics)
    assert len(t.metrics_history) == 2
    assert (run_dir / "behaviour_metrics.jsonl").exists()


def test_evaluate_checkpoints_explicit_paths(make_tracker, model, tmp_path):
    ckpt1 = tmp_path / "a.pt"
    ckpt2 = tmp_path / "b.pt"
    save_fake_checkpoint(model, ckpt1, step=200)
    save_fake_checkpoint(model, ckpt2, step=100)

    t = make_tracker(config=TrackerConfig(
        bigram_compute_every=10, ngram_enabled=False, bigram_n_samples=10,
    ))
    t.fit(max_fit_samples=20)

    all_metrics = t.evaluate_checkpoints(checkpoint_paths=[ckpt1, ckpt2], save=False)
    assert [m["step"] for m in all_metrics] == [100, 200]  # sorted


def test_evaluate_checkpoints_no_dir_raises(make_tracker):
    # Tracker with no run_dir: evaluate_checkpoints() with no args must raise
    # a ValueError about the missing source.
    t = make_tracker(
        config=TrackerConfig(ngram_enabled=False, bigram_n_samples=10),
        run_dir=None,
    )
    t.fit(max_fit_samples=20)
    with pytest.raises(ValueError, match="checkpoint_paths or run_dir"):
        t.evaluate_checkpoints()
