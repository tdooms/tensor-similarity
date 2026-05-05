from pathlib import Path

import pytest
import torch
import yaml

import bilinear_icl.eval.behavioral as behavioral
import bilinear_icl.train.trainer as trainer_mod
from bilinear_icl.train.sanity import NonFiniteError
from bilinear_icl.train.trainer import train


def _smoke_cfg():
    with Path("configs/smoke.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["wandb"]["enabled"] = False
    cfg["hf"]["enabled"] = False
    cfg["figures"]["run_at_end"] = False
    return cfg


def test_nan_loss_raises_and_writes_error(tmp_path, monkeypatch):
    cfg = _smoke_cfg()

    def bad_loss(*args, **kwargs):
        return torch.tensor(float("nan"), requires_grad=True)

    monkeypatch.setattr(trainer_mod, "mean_mse", bad_loss)

    run_dir = tmp_path / "nan_loss"
    with pytest.raises(NonFiniteError):
        train(cfg, run_dir=str(run_dir))

    assert (run_dir / "errors" / "nan_loss_step1.txt").exists()


def test_nan_eval_metrics_raises_and_writes_error(tmp_path, monkeypatch):
    cfg = _smoke_cfg()

    original_compute = behavioral.compute

    def bad_compute(*args, **kwargs):
        out = original_compute(*args, **kwargs)
        out["test_loss"] = float("nan")
        return out

    monkeypatch.setattr(behavioral, "compute", bad_compute)

    run_dir = tmp_path / "nan_eval"
    with pytest.raises(NonFiniteError):
        train(cfg, run_dir=str(run_dir))

    assert any((run_dir / "errors").glob("nan_eval_metrics_step*.txt"))


def test_nan_params_raises_and_writes_error(tmp_path, monkeypatch):
    cfg = _smoke_cfg()

    class NaNInjectingOptimizer(torch.optim.SGD):
        def __init__(self, params):
            super().__init__(params, lr=1e-3)
            self._did_inject = False

        def step(self, closure=None):
            out = super().step(closure)
            if not self._did_inject:
                self.param_groups[0]["params"][0].data.fill_(float("nan"))
                self._did_inject = True
            return out

    def fake_build_optimizer(model, **kwargs):
        return NaNInjectingOptimizer(model.parameters())

    monkeypatch.setattr(trainer_mod, "build_optimizer", fake_build_optimizer)

    run_dir = tmp_path / "nan_params"
    with pytest.raises(NonFiniteError):
        train(cfg, run_dir=str(run_dir))

    assert (run_dir / "errors" / "nan_params_step1.txt").exists()
