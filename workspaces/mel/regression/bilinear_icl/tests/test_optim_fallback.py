import pytest
from torch.optim import AdamW

from bilinear_icl.models import RegressionTransformer
from bilinear_icl.train import optim


def _tiny_model():
    return RegressionTransformer(D=4, K=8, d_model=32, n_head=4, n_layers=2, d_mlp=32)


def test_build_optimizer_raises_without_muon(monkeypatch):
    monkeypatch.setattr(optim, "SingleDeviceMuonWithAuxAdam", None)
    model = _tiny_model()

    with pytest.raises(RuntimeError, match="muon is required"):
        optim.build_optimizer(
            model,
            muon_lr=0.02,
            adamw_lr=1e-3,
            weight_decay=0.02,
            betas=(0.9, 0.95),
        )


def test_build_optimizer_adamw_fallback_uses_adamw_lr(monkeypatch):
    monkeypatch.setattr(optim, "SingleDeviceMuonWithAuxAdam", None)
    model = _tiny_model()

    opt = optim.build_optimizer(
        model,
        muon_lr=0.02,
        adamw_lr=1e-3,
        weight_decay=0.02,
        betas=(0.9, 0.95),
        allow_adamw_fallback=True,
    )

    assert isinstance(opt, AdamW)
    assert all(pg["lr"] == 1e-3 for pg in opt.param_groups)
