"""Tests for the Trainer class."""
from pathlib import Path

import pytest
import torch

from models import AttentionLM
from train.trainer import Trainer


@pytest.fixture
def make_trainer(tiny_config, dummy_dataloader, device, tmp_path):
    """Factory producing a Trainer wired to a fresh tmp run dir.

    Optional kwargs: ``max_steps`` (sets cfg.train.max_steps),
    ``run_dir`` (override output path).
    """
    def _factory(*, max_steps=None, run_dir=None):
        torch.manual_seed(tiny_config["seed"])
        model = AttentionLM.from_config(tiny_config)
        if max_steps is not None:
            tiny_config["train"]["max_steps"] = max_steps
        return Trainer(
            model=model,
            train_dataloader=dummy_dataloader,
            val_dataloader=dummy_dataloader,
            cfg=tiny_config,
            run_dir=str(run_dir if run_dir is not None else tmp_path),
            device=device,
        )
    return _factory


def test_trainer_initialization(make_trainer):
    t = make_trainer()
    assert t.model is not None
    assert t.optimizer is not None
    assert t.scheduler is not None
    assert t.step == 0


def test_trainer_creates_run_dir(make_trainer, tmp_path):
    run_dir = tmp_path / "test_run"
    make_trainer(run_dir=run_dir)
    assert run_dir.exists()
    assert (run_dir / "checkpoints").exists()
    assert (run_dir / "errors").exists()


def test_trainer_single_step(make_trainer):
    t = make_trainer(max_steps=1)
    t.train(eval_every=1000, save_every=1000)
    assert t.step == 1


def test_trainer_loss_does_not_explode(make_trainer, dummy_dataloader, device):
    """Loss after a few steps should not be wildly bigger than the initial."""
    from train.losses import compute_loss

    t = make_trainer(max_steps=5)
    input_ids = next(iter(dummy_dataloader))["input_ids"].to(device)
    with torch.no_grad():
        initial = compute_loss(t.model(input_ids), input_ids).item()
    t.train(eval_every=1000, save_every=1000)
    with torch.no_grad():
        final = compute_loss(t.model(input_ids), input_ids).item()
    assert final < initial * 2, f"Loss exploded: {initial} -> {final}"


def test_trainer_saves_checkpoint(make_trainer, tmp_path):
    t = make_trainer(max_steps=5)
    t.train(eval_every=1000, save_every=2)
    assert list(tmp_path.glob("checkpoints/*.pt")), "No checkpoints saved"


def test_trainer_logs_metrics(make_trainer, tmp_path):
    t = make_trainer(max_steps=10)
    t.train(eval_every=1000, save_every=1000)
    metrics_file = tmp_path / "metrics.jsonl"
    assert metrics_file.exists()
    assert metrics_file.read_text().strip()


def test_trainer_checkpoint_loadable(make_trainer, tiny_config, tmp_path, device):
    t = make_trainer(max_steps=3)
    t.train(eval_every=1000, save_every=1)
    checkpoints = list(tmp_path.glob("checkpoints/*.pt"))
    assert checkpoints

    ckpt = torch.load(checkpoints[0], map_location=device, weights_only=True)
    assert {"model_state_dict", "optimizer_state_dict", "step"} <= ckpt.keys()

    new_model = AttentionLM.from_config(tiny_config)
    new_model.load_state_dict(ckpt["model_state_dict"])
