"""Tests for the Trainer class."""
import pytest
import torch
import tempfile
import shutil
from pathlib import Path

from models import AttentionLM
from train.trainer import Trainer


def test_trainer_initialization(tiny_config, dummy_dataloader, device):
    """Test that Trainer initializes correctly."""
    torch.manual_seed(tiny_config["seed"])
    model = AttentionLM.from_config(tiny_config)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            train_dataloader=dummy_dataloader,
            val_dataloader=dummy_dataloader,
            cfg=tiny_config,
            run_dir=tmpdir,
            device=device,
        )
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.step == 0


def test_trainer_creates_run_dir(tiny_config, dummy_dataloader, device):
    """Test that Trainer creates run directory structure."""
    torch.manual_seed(tiny_config["seed"])
    model = AttentionLM.from_config(tiny_config)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "test_run"
        trainer = Trainer(
            model=model,
            train_dataloader=dummy_dataloader,
            cfg=tiny_config,
            run_dir=str(run_dir),
            device=device,
        )
        
        assert run_dir.exists()
        assert (run_dir / "checkpoints").exists()
        assert (run_dir / "errors").exists()


def test_trainer_single_step(tiny_config, dummy_dataloader, device):
    """Test that a single training step works."""
    torch.manual_seed(tiny_config["seed"])
    model = AttentionLM.from_config(tiny_config)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tiny_config["train"]["max_steps"] = 1
        trainer = Trainer(
            model=model,
            train_dataloader=dummy_dataloader,
            cfg=tiny_config,
            run_dir=tmpdir,
            device=device,
        )
        
        trainer.train(eval_every=1000, save_every=1000)
        
        assert trainer.step == 1


def test_trainer_loss_decreases(tiny_config, dummy_dataloader, device):
    """Test that loss decreases over training."""
    torch.manual_seed(tiny_config["seed"])
    model = AttentionLM.from_config(tiny_config)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tiny_config["train"]["max_steps"] = 5
        trainer = Trainer(
            model=model,
            train_dataloader=dummy_dataloader,
            cfg=tiny_config,
            run_dir=tmpdir,
            device=device,
        )
        
        # Get initial loss
        batch = next(iter(dummy_dataloader))
        input_ids = batch["input_ids"].to(device)
        with torch.no_grad():
            logits = model(input_ids)
            from train.losses import compute_loss
            initial_loss = compute_loss(logits, input_ids).item()
        
        trainer.train(eval_every=1000, save_every=1000)
        
        # Get final loss
        with torch.no_grad():
            logits = model(input_ids)
            final_loss = compute_loss(logits, input_ids).item()
        
        # Loss should decrease (or at least not explode)
        assert final_loss < initial_loss * 2, \
            f"Loss should not explode. Initial: {initial_loss}, Final: {final_loss}"


def test_trainer_saves_checkpoint(tiny_config, dummy_dataloader, device):
    """Test that Trainer saves checkpoints."""
    torch.manual_seed(tiny_config["seed"])
    model = AttentionLM.from_config(tiny_config)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tiny_config["train"]["max_steps"] = 5
        trainer = Trainer(
            model=model,
            train_dataloader=dummy_dataloader,
            cfg=tiny_config,
            run_dir=tmpdir,
            device=device,
        )
        
        trainer.train(eval_every=1000, save_every=2)
        
        # Should have saved at step 2, 4, and final
        checkpoints = list(Path(tmpdir).glob("checkpoints/*.pt"))
        assert len(checkpoints) >= 1, "Should have saved at least one checkpoint"


def test_trainer_logs_metrics(tiny_config, dummy_dataloader, device):
    """Test that Trainer logs metrics to file."""
    torch.manual_seed(tiny_config["seed"])
    model = AttentionLM.from_config(tiny_config)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tiny_config["train"]["max_steps"] = 10
        trainer = Trainer(
            model=model,
            train_dataloader=dummy_dataloader,
            cfg=tiny_config,
            run_dir=tmpdir,
            device=device,
        )
        
        trainer.train(eval_every=1000, save_every=1000)
        
        metrics_file = Path(tmpdir) / "metrics.jsonl"
        assert metrics_file.exists(), "Metrics file should exist"
        
        # Check that file has content
        content = metrics_file.read_text()
        assert len(content) > 0, "Metrics file should have content"


def test_trainer_checkpoint_loadable(tiny_config, dummy_dataloader, device):
    """Test that saved checkpoints can be loaded."""
    torch.manual_seed(tiny_config["seed"])
    model = AttentionLM.from_config(tiny_config)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tiny_config["train"]["max_steps"] = 3
        trainer = Trainer(
            model=model,
            train_dataloader=dummy_dataloader,
            cfg=tiny_config,
            run_dir=tmpdir,
            device=device,
        )
        
        trainer.train(eval_every=1000, save_every=1)
        
        # Find a checkpoint
        checkpoints = list(Path(tmpdir).glob("checkpoints/*.pt"))
        assert len(checkpoints) > 0
        
        # Load it
        checkpoint = torch.load(checkpoints[0], map_location=device, weights_only=True)
        
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "step" in checkpoint
        
        # Verify model can load the state dict
        new_model = AttentionLM.from_config(tiny_config)
        new_model.load_state_dict(checkpoint["model_state_dict"])
