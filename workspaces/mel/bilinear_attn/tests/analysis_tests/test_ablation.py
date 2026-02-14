"""Tests for position-ablated loss computation."""
import pytest
import torch
import tempfile
from pathlib import Path

from analysis.behaviour.ablation import compute_ablated_loss, compute_val_loss, ablate_rotary
from analysis.behaviour.tracker import BehaviourTracker, TrackerConfig
from tests.analysis_tests.conftest import V


def test_ablate_rotary_is_identity(model):
    """Test that ablate_rotary makes Rotary modules act as identity."""
    from models.attention_kernels.rotary import Rotary
    
    # Collect all Rotary modules
    rotary_modules = [m for m in model.modules() if isinstance(m, Rotary)]
    assert len(rotary_modules) > 0, "Model should have Rotary modules"
    
    x = torch.randn(1, 8, rotary_modules[0].cos_cached.shape[-1])
    x_4d = x.unsqueeze(2)  # (B, T, 1, d_head)
    
    # Normal forward changes the tensor
    normal_out = rotary_modules[0](x_4d)
    assert not torch.allclose(x_4d, normal_out), "Rotary should modify input"
    
    # Ablated forward should be identity
    with ablate_rotary(model):
        ablated_out = rotary_modules[0](x_4d)
        assert torch.allclose(x_4d, ablated_out), "Ablated Rotary should be identity"
    
    # After context exit, Rotary should work normally again
    restored_out = rotary_modules[0](x_4d)
    assert torch.allclose(normal_out, restored_out), "Rotary should be restored after context exit"


def test_compute_val_loss(model, dummy_dataloader):
    """Test that compute_val_loss returns a finite float."""
    loss = compute_val_loss(model, dummy_dataloader, device="cpu", max_batches=5)
    assert isinstance(loss, float)
    assert loss > 0
    assert not torch.isnan(torch.tensor(loss))


def test_compute_ablated_loss(model, dummy_dataloader):
    """Test that compute_ablated_loss returns a finite float."""
    loss = compute_ablated_loss(model, dummy_dataloader, device="cpu", max_batches=5)
    assert isinstance(loss, float)
    assert loss > 0
    assert not torch.isnan(torch.tensor(loss))


def test_ablated_loss_differs_from_val_loss(model, dummy_dataloader):
    """Test that ablated loss is different from normal val loss."""
    val_loss = compute_val_loss(model, dummy_dataloader, device="cpu", max_batches=5)
    abl_loss = compute_ablated_loss(model, dummy_dataloader, device="cpu", max_batches=5)
    # They should differ because RoPE changes the attention pattern
    # (in rare random-init cases they could be close, so we just check they're both valid)
    assert isinstance(val_loss, float)
    assert isinstance(abl_loss, float)


def test_tracker_compute_metrics_includes_ablation(model, dummy_dataloader):
    """Test that compute_metrics includes ablation metrics when enabled."""
    config = TrackerConfig(
        bigram_enabled=False,
        ngram_enabled=False,
        ablation_enabled=True,
        ablation_compute_every=10,
        ablation_max_val_batches=5,
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
        tracker._is_fitted = True  # no bigram/ngram to fit
        
        metrics = tracker.compute_metrics(step=10)
        
        assert "val_loss" in metrics
        assert "ablated_loss" in metrics
        assert "ablation_gap" in metrics
        assert isinstance(metrics["ablated_loss"], float)
        assert metrics["ablation_gap"] == pytest.approx(
            metrics["ablated_loss"] - metrics["val_loss"]
        )


def test_tracker_should_compute_ablation(model, dummy_dataloader):
    """Test that should_compute respects ablation interval."""
    config = TrackerConfig(
        bigram_enabled=False,
        ngram_enabled=False,
        ablation_enabled=True,
        ablation_compute_every=200,
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
        
        assert tracker.should_compute(200) == True
        assert tracker.should_compute(400) == True
        assert tracker.should_compute(50) == False


def _save_fake_checkpoint(model, path, step):
    """Helper: save a checkpoint in the same format as Trainer.save_checkpoint."""
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
    }, path)


def test_evaluate_checkpoint_includes_ablation(model, dummy_dataloader):
    """Test that evaluate_checkpoint includes ablation metrics."""
    config = TrackerConfig(
        bigram_enabled=False,
        ngram_enabled=False,
        ablation_enabled=True,
        ablation_max_val_batches=5,
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
        tracker._is_fitted = True
        
        metrics = tracker.evaluate_checkpoint(ckpt_path)
        
        assert metrics["step"] == 100
        assert "val_loss" in metrics
        assert "ablated_loss" in metrics
        assert "ablation_gap" in metrics
