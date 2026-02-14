"""Tests for behaviour tracker."""
import pytest
import torch
import tempfile
from pathlib import Path

from analysis.behaviour.tracker import BehaviourTracker, TrackerConfig
from tests.analysis_tests.conftest import V


def test_tracker_config_defaults():
    """Test TrackerConfig has sensible defaults."""
    config = TrackerConfig()
    
    assert config.bigram_enabled == True
    assert config.ngram_enabled == True
    assert config.bigram_compute_every > 0
    assert config.ngram_compute_every > 0


def test_tracker_init(model, dummy_dataloader):
    """Test BehaviourTracker initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = BehaviourTracker(
            model=model,
            train_dataloader=dummy_dataloader,
            val_dataloader=dummy_dataloader,
            vocab_size=V,
            run_dir=tmpdir,
        )
        
        assert tracker.model is not None
        assert not tracker._is_fitted


def test_tracker_fit(model, dummy_dataloader):
    """Test fitting the tracker."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = BehaviourTracker(
            model=model,
            train_dataloader=dummy_dataloader,
            val_dataloader=dummy_dataloader,
            vocab_size=V,
            run_dir=tmpdir,
        )
        
        tracker.fit(max_fit_samples=20)
        
        assert tracker._is_fitted
        assert tracker.bigram_analyzer is not None
        assert tracker.ngram_analyzer is not None


def test_tracker_compute_metrics(model, dummy_dataloader):
    """Test computing metrics."""
    config = TrackerConfig(
        bigram_compute_every=10,
        ngram_compute_every=10,
        bigram_n_samples=10,
        ngram_max_val_batches=5,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = BehaviourTracker(
            model=model,
            train_dataloader=dummy_dataloader,
            val_dataloader=dummy_dataloader,
            vocab_size=V,
            run_dir=tmpdir,
            config=config,
        )
        
        tracker.fit(max_fit_samples=20)
        
        metrics = tracker.compute_metrics(step=10)
        
        assert "step" in metrics
        assert "bigram_score" in metrics


def test_tracker_should_compute(model, dummy_dataloader):
    """Test should_compute method."""
    config = TrackerConfig(
        bigram_compute_every=100,
        ngram_compute_every=200,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = BehaviourTracker(
            model=model,
            train_dataloader=dummy_dataloader,
            val_dataloader=dummy_dataloader,
            vocab_size=V,
            run_dir=tmpdir,
            config=config,
        )
        
        assert tracker.should_compute(100) == True
        assert tracker.should_compute(200) == True
        assert tracker.should_compute(50) == False


def test_tracker_log_metrics(model, dummy_dataloader):
    """Test logging metrics."""
    config = TrackerConfig(
        bigram_compute_every=10,
        ngram_compute_every=10,
        bigram_n_samples=10,
        ngram_max_val_batches=5,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = BehaviourTracker(
            model=model,
            train_dataloader=dummy_dataloader,
            val_dataloader=dummy_dataloader,
            vocab_size=V,
            run_dir=tmpdir,
            config=config,
        )
        
        tracker.fit(max_fit_samples=20)
        
        metrics = tracker.log_metrics(step=10, additional_metrics={"train_loss": 5.0})
        
        assert len(tracker.metrics_history) > 0
        assert "train_loss" in metrics


def test_tracker_toggle(model, dummy_dataloader):
    """Test toggling metrics on/off."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = BehaviourTracker(
            model=model,
            train_dataloader=dummy_dataloader,
            val_dataloader=dummy_dataloader,
            vocab_size=V,
            run_dir=tmpdir,
        )
        
        assert tracker.config.bigram_enabled == True
        
        tracker.toggle_bigram(False)
        
        assert tracker.config.bigram_enabled == False


def test_tracker_set_interval(model, dummy_dataloader):
    """Test setting compute intervals."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = BehaviourTracker(
            model=model,
            train_dataloader=dummy_dataloader,
            val_dataloader=dummy_dataloader,
            vocab_size=V,
            run_dir=tmpdir,
        )
        
        tracker.set_compute_interval("bigram", 1000)
        
        assert tracker.config.bigram_compute_every == 1000


def test_tracker_get_metric_series(model, dummy_dataloader):
    """Test extracting metric series."""
    config = TrackerConfig(
        bigram_compute_every=10,
        ngram_enabled=False,
        bigram_n_samples=10,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = BehaviourTracker(
            model=model,
            train_dataloader=dummy_dataloader,
            val_dataloader=dummy_dataloader,
            vocab_size=V,
            run_dir=tmpdir,
            config=config,
        )
        
        tracker.fit(max_fit_samples=20)
        
        tracker.log_metrics(step=10)
        tracker.log_metrics(step=20)
        
        steps, values = tracker.get_metric_series("bigram_score")
        
        assert len(steps) == 2
        assert len(values) == 2


def test_tracker_save_load_history(model, dummy_dataloader):
    """Test saving and loading history."""
    config = TrackerConfig(
        bigram_compute_every=10,
        ngram_enabled=False,
        bigram_n_samples=10,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = BehaviourTracker(
            model=model,
            train_dataloader=dummy_dataloader,
            val_dataloader=dummy_dataloader,
            vocab_size=V,
            run_dir=tmpdir,
            config=config,
        )
        
        tracker.fit(max_fit_samples=20)
        tracker.log_metrics(step=10)
        
        save_path = Path(tmpdir) / "test_history.json"
        tracker.save_history(str(save_path))
        
        assert save_path.exists()
        
        # Load into new tracker
        tracker2 = BehaviourTracker(
            model=model,
            train_dataloader=dummy_dataloader,
            val_dataloader=dummy_dataloader,
            vocab_size=V,
        )
        tracker2.load_history(str(save_path))
        
        assert len(tracker2.metrics_history) == len(tracker.metrics_history)


def test_tracker_cache_save_and_load(model, dummy_dataloader):
    """Test that fit() saves to cache and subsequent fit() loads from cache."""
    config = TrackerConfig(
        bigram_compute_every=10,
        ngram_compute_every=10,
        bigram_n_samples=10,
        ngram_max_n=4,
        ngram_max_val_batches=5,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        
        # First tracker: fits from data, saves to cache
        tracker1 = BehaviourTracker(
            model=model,
            train_dataloader=dummy_dataloader,
            val_dataloader=dummy_dataloader,
            vocab_size=V,
            run_dir=str(Path(tmpdir) / "run1"),
            cache_dir=str(cache_dir),
            config=config,
        )
        tracker1.fit(max_fit_samples=20)
        
        assert (cache_dir / "bigram.pt").exists()
        assert (cache_dir / "ngram.pt").exists()
        
        orig_bigram_total = tracker1.bigram_analyzer.total_bigrams
        orig_ngram_ns = sorted(tracker1.ngram_analyzer.common_ngrams.keys())
        
        # Second tracker: loads from cache (no dataloader iteration needed)
        tracker2 = BehaviourTracker(
            model=model,
            train_dataloader=dummy_dataloader,
            val_dataloader=dummy_dataloader,
            vocab_size=V,
            run_dir=str(Path(tmpdir) / "run2"),
            cache_dir=str(cache_dir),
            config=config,
        )
        tracker2.fit(max_fit_samples=20)
        
        assert tracker2._is_fitted
        assert tracker2.bigram_analyzer.total_bigrams == orig_bigram_total
        assert sorted(tracker2.ngram_analyzer.common_ngrams.keys()) == orig_ngram_ns


def _save_fake_checkpoint(model, path, step):
    """Helper: save a checkpoint in the same format as Trainer.save_checkpoint."""
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
    }, path)


def test_evaluate_checkpoint(model, dummy_dataloader):
    """Test evaluating a single checkpoint."""
    config = TrackerConfig(
        bigram_compute_every=10,
        ngram_compute_every=10,
        bigram_n_samples=10,
        ngram_max_n=3,
        ngram_max_val_batches=5,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "step_100.pt"
        _save_fake_checkpoint(model, ckpt_path, step=100)
        
        tracker = BehaviourTracker(
            model=model,
            train_dataloader=dummy_dataloader,
            val_dataloader=dummy_dataloader,
            vocab_size=V,
            run_dir=tmpdir,
            config=config,
        )
        tracker.fit(max_fit_samples=20)
        
        metrics = tracker.evaluate_checkpoint(ckpt_path)
        
        assert metrics["step"] == 100
        assert "checkpoint" in metrics
        assert "bigram_score" in metrics
        assert "bigram_entropy" in metrics
        assert "bigram_gap" in metrics
        assert isinstance(metrics["bigram_score"], float)


def test_evaluate_checkpoint_unfitted_raises(model, dummy_dataloader):
    """Test that evaluate_checkpoint raises if tracker is not fitted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "step_0.pt"
        _save_fake_checkpoint(model, ckpt_path, step=0)
        
        tracker = BehaviourTracker(
            model=model,
            train_dataloader=dummy_dataloader,
            val_dataloader=dummy_dataloader,
            vocab_size=V,
            run_dir=tmpdir,
        )
        
        with pytest.raises(RuntimeError, match="fit"):
            tracker.evaluate_checkpoint(ckpt_path)


def test_evaluate_checkpoints_from_dir(model, dummy_dataloader):
    """Test evaluating all checkpoints found in a run directory."""
    config = TrackerConfig(
        bigram_compute_every=10,
        ngram_enabled=False,
        bigram_n_samples=10,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir()
        
        # Save two checkpoints at different steps
        _save_fake_checkpoint(model, ckpt_dir / "step_500.pt", step=500)
        _save_fake_checkpoint(model, ckpt_dir / "step_1000.pt", step=1000)
        
        tracker = BehaviourTracker(
            model=model,
            train_dataloader=dummy_dataloader,
            val_dataloader=dummy_dataloader,
            vocab_size=V,
            run_dir=str(run_dir),
            config=config,
        )
        tracker.fit(max_fit_samples=20)
        
        all_metrics = tracker.evaluate_checkpoints(run_dir=run_dir)
        
        assert len(all_metrics) == 2
        assert all_metrics[0]["step"] == 500
        assert all_metrics[1]["step"] == 1000
        assert "bigram_score" in all_metrics[0]
        assert "bigram_score" in all_metrics[1]
        
        # Check that history was updated
        assert len(tracker.metrics_history) == 2
        
        # Check that file was written
        assert (run_dir / "behaviour_metrics.jsonl").exists()


def test_evaluate_checkpoints_explicit_paths(model, dummy_dataloader):
    """Test evaluating an explicit list of checkpoint paths."""
    config = TrackerConfig(
        bigram_compute_every=10,
        ngram_enabled=False,
        bigram_n_samples=10,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt1 = Path(tmpdir) / "a.pt"
        ckpt2 = Path(tmpdir) / "b.pt"
        _save_fake_checkpoint(model, ckpt1, step=200)
        _save_fake_checkpoint(model, ckpt2, step=100)
        
        tracker = BehaviourTracker(
            model=model,
            train_dataloader=dummy_dataloader,
            val_dataloader=dummy_dataloader,
            vocab_size=V,
            run_dir=tmpdir,
            config=config,
        )
        tracker.fit(max_fit_samples=20)
        
        all_metrics = tracker.evaluate_checkpoints(
            checkpoint_paths=[ckpt1, ckpt2],
            save=False,
        )
        
        # Should be sorted by step
        assert all_metrics[0]["step"] == 100
        assert all_metrics[1]["step"] == 200


def test_evaluate_checkpoints_no_dir_raises(model, dummy_dataloader):
    """Test that evaluate_checkpoints raises without run_dir or paths."""
    config = TrackerConfig(ngram_enabled=False, bigram_n_samples=10)
    
    tracker = BehaviourTracker(
        model=model,
        train_dataloader=dummy_dataloader,
        val_dataloader=dummy_dataloader,
        vocab_size=V,
        config=config,
    )
    tracker.fit(max_fit_samples=20)
    
    with pytest.raises(ValueError, match="checkpoint_paths or run_dir"):
        tracker.evaluate_checkpoints()
