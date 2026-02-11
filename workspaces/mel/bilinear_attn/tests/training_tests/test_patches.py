"""Tests for P0–P4 patches: Muon optimizer, norm_type, cosine schedule, betas, dtype."""
import math
import pytest
import torch
import torch.nn as nn

from models.transformer import AttentionLM, NORM_TYPES, NORM_PLACES
from train.optim import (
    create_optimizer,
    create_scheduler,
    _is_muon_param,
    Optimizers,
)

# ---------------------------------------------------------------------------
# Tiny model helpers
# ---------------------------------------------------------------------------

V = 97
N_CTX = 16
D_MODEL = 32
N_HEAD = 4
N_LAYERS = 2


def _make_model(**overrides):
    defaults = dict(
        vocab_size=V, n_ctx=N_CTX, d_model=D_MODEL,
        n_head=N_HEAD, n_layers=N_LAYERS, attn_type="quadratic",
        norm_type="rmsnorm", norm_place="pre_unembed",
    )
    defaults.update(overrides)
    return AttentionLM(**defaults)


# ===================================================================
# P0: Muon param grouping
# ===================================================================

class TestMuonParamGrouping:
    """Verify _is_muon_param correctly classifies parameters."""

    def test_attention_weights_are_muon(self):
        model = _make_model()
        muon_names = [n for n, p in model.named_parameters() if _is_muon_param(n, p)]
        # Should include q, k, v, o weight for each layer
        for i in range(N_LAYERS):
            for proj in ("q", "k", "v", "o"):
                expected = f"layers.{i}.{proj}.weight"
                assert expected in muon_names, f"{expected} should be a Muon param"

    def test_embed_unembed_not_muon(self):
        model = _make_model()
        muon_names = [n for n, p in model.named_parameters() if _is_muon_param(n, p)]
        assert "embed.weight" not in muon_names
        assert "unembed.weight" not in muon_names

    def test_biases_not_muon(self):
        model = _make_model(use_bias_qkv=True, use_bias_o=True)
        muon_names = [n for n, p in model.named_parameters() if _is_muon_param(n, p)]
        bias_names = [n for n in muon_names if "bias" in n]
        assert len(bias_names) == 0, f"Biases should not be Muon params: {bias_names}"

    def test_norm_weights_not_muon(self):
        model = _make_model(norm_place="pre_layer", use_rmsnorm_qk=True)
        muon_names = [n for n, p in model.named_parameters() if _is_muon_param(n, p)]
        norm_names = [n for n in muon_names if "norm" in n]
        assert len(norm_names) == 0, f"Norm weights should not be Muon params: {norm_names}"

    def test_muon_param_count(self):
        model = _make_model(use_bias_qkv=False, use_bias_o=False)
        muon_names = [n for n, p in model.named_parameters() if _is_muon_param(n, p)]
        # 4 projections * N_LAYERS
        assert len(muon_names) == 4 * N_LAYERS


class TestCreateOptimizerMuon:
    """Test create_optimizer with use_muon=True."""

    def test_returns_optimizers_namedtuple(self):
        model = _make_model()
        result = create_optimizer(model, use_muon=True)
        assert isinstance(result, Optimizers)

    def test_muon_optimizer_has_three_param_groups(self):
        model = _make_model(use_bias_qkv=True, use_bias_o=True)
        result = create_optimizer(model, use_muon=True)
        # 3 groups: muon, adam_decay, adam_nodecay
        assert len(result.muon.param_groups) == 3

    def test_all_params_accounted_for(self):
        model = _make_model()
        result = create_optimizer(model, use_muon=True)
        opt_param_count = sum(
            p.numel() for g in result.muon.param_groups for p in g["params"]
        )
        model_param_count = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        assert opt_param_count == model_param_count, (
            f"Optimizer covers {opt_param_count} params but model has {model_param_count}"
        )


class TestCreateOptimizerAdamOnly:
    """Test create_optimizer with use_muon=False (legacy path)."""

    def test_returns_adamw(self):
        model = _make_model()
        result = create_optimizer(model, use_muon=False)
        assert isinstance(result, torch.optim.AdamW)

    def test_two_param_groups(self):
        model = _make_model()
        result = create_optimizer(model, use_muon=False)
        assert len(result.param_groups) == 2

    def test_betas_default(self):
        model = _make_model()
        result = create_optimizer(model, use_muon=False)
        for g in result.param_groups:
            assert g["betas"] == (0.9, 0.95)


# ===================================================================
# P1: norm_type and norm_place
# ===================================================================

class TestNormPlace:
    """Test all norm_place modes produce valid outputs."""

    @pytest.mark.parametrize("norm_place", NORM_PLACES)
    def test_forward_runs(self, norm_place):
        model = _make_model(norm_place=norm_place)
        x = torch.randint(0, V, (2, 8))
        logits = model(x)
        assert logits.shape == (2, 8, V)
        assert torch.isfinite(logits).all()

    def test_none_has_no_norms(self):
        model = _make_model(norm_place="none")
        assert isinstance(model.final_norm, nn.Identity)
        assert model.layer_norms is None
        assert model.embed_norm is None

    def test_post_embed_has_embed_norm(self):
        model = _make_model(norm_place="post_embed")
        assert model.embed_norm is not None
        assert not isinstance(model.embed_norm, nn.Identity)
        assert isinstance(model.final_norm, nn.Identity)
        assert model.layer_norms is None

    def test_pre_unembed_has_final_norm(self):
        model = _make_model(norm_place="pre_unembed")
        assert not isinstance(model.final_norm, nn.Identity)
        assert model.layer_norms is None
        assert model.embed_norm is None

    def test_pre_layer_has_all_norms(self):
        model = _make_model(norm_place="pre_layer")
        assert not isinstance(model.final_norm, nn.Identity)
        assert model.layer_norms is not None
        assert len(model.layer_norms) == N_LAYERS

    def test_invalid_norm_place_raises(self):
        with pytest.raises(AssertionError):
            _make_model(norm_place="invalid")

    @pytest.mark.parametrize("norm_place", NORM_PLACES)
    def test_backward_runs(self, norm_place):
        model = _make_model(norm_place=norm_place)
        x = torch.randint(0, V, (2, 8))
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_pre_layer_residual_correct(self):
        """Verify pre_layer mode applies norm before attention, not to residual."""
        model = _make_model(norm_place="pre_layer")
        x = torch.randint(0, V, (1, 4))
        model_none = _make_model(norm_place="none")
        model_none.load_state_dict(
            {k: v for k, v in model.state_dict().items() if k in model_none.state_dict()},
            strict=False,
        )
        out_pre = model(x)
        out_none = model_none(x)
        assert not torch.allclose(out_pre, out_none, atol=1e-6)


class TestNormType:
    """Test norm_type selects the correct normalisation function."""

    def test_rmsnorm_uses_rmsnorm(self):
        model = _make_model(norm_type="rmsnorm", norm_place="pre_unembed")
        assert isinstance(model.final_norm, nn.RMSNorm)

    def test_layernorm_uses_layernorm(self):
        model = _make_model(norm_type="layernorm", norm_place="pre_unembed")
        assert isinstance(model.final_norm, nn.LayerNorm)

    def test_none_uses_identity(self):
        model = _make_model(norm_type="none", norm_place="pre_unembed")
        assert isinstance(model.final_norm, nn.Identity)

    def test_invalid_norm_type_raises(self):
        with pytest.raises(AssertionError):
            _make_model(norm_type="invalid")

    @pytest.mark.parametrize("norm_type", NORM_TYPES)
    def test_from_config_norm_type(self, norm_type):
        cfg = {
            "model": {
                "vocab_size": V, "n_ctx": N_CTX, "d_model": D_MODEL,
                "n_head": N_HEAD, "n_layers": N_LAYERS,
                "attn_type": "quadratic", "norm_type": norm_type,
            },
            "init": {},
        }
        model = AttentionLM.from_config(cfg)
        assert model.norm_type == norm_type

    @pytest.mark.parametrize("norm_place", NORM_PLACES)
    def test_from_config_norm_place(self, norm_place):
        cfg = {
            "model": {
                "vocab_size": V, "n_ctx": N_CTX, "d_model": D_MODEL,
                "n_head": N_HEAD, "n_layers": N_LAYERS,
                "attn_type": "quadratic", "norm_place": norm_place,
            },
            "init": {},
        }
        model = AttentionLM.from_config(cfg)
        assert model.norm_place == norm_place


# ===================================================================
# P2: Cosine LR schedule with lr_decay_frac
# ===================================================================

class TestCosineScheduler:
    """Test the cosine warmup+decay scheduler."""

    def _step_scheduler(self, scheduler, optimizer, n):
        lrs = []
        for _ in range(n):
            lrs.append(scheduler.get_last_lr()[0])
            optimizer.step()
            scheduler.step()
        return lrs

    def test_warmup_increases(self):
        model = _make_model()
        opt = create_optimizer(model, use_muon=False)
        sched = create_scheduler(opt, warmup_steps=10, max_steps=100)
        lrs = self._step_scheduler(sched, opt, 10)
        # LR should be monotonically increasing during warmup
        for i in range(1, len(lrs)):
            assert lrs[i] > lrs[i - 1], f"LR should increase during warmup: {lrs}"

    def test_decay_is_cosine(self):
        model = _make_model()
        opt = create_optimizer(model, use_muon=False, lr=1e-3)
        sched = create_scheduler(opt, warmup_steps=0, max_steps=100, lr_decay_frac=0.1)
        lrs = self._step_scheduler(sched, opt, 100)
        # At step 0 (no warmup), lr_lambda = lr_decay_frac + 0.5*(1+cos(0))*(1-lr_decay_frac)
        # = 0.1 + 0.5*2*0.9 = 0.1 + 0.9 = 1.0 → lr = 1e-3
        assert abs(lrs[0] - 1e-3) < 1e-7
        # At midpoint, cosine should be ~0.5*(1+cos(pi/2)) = 0.5
        mid = lrs[50]
        expected_mid = 1e-3 * (0.1 + 0.5 * (1.0 + math.cos(math.pi * 50 / 100)) * 0.9)
        assert abs(mid - expected_mid) < 1e-7

    def test_decays_to_floor(self):
        model = _make_model()
        opt = create_optimizer(model, use_muon=False, lr=1e-3)
        sched = create_scheduler(opt, warmup_steps=0, max_steps=100, lr_decay_frac=0.1)
        self._step_scheduler(sched, opt, 100)
        final_lr = sched.get_last_lr()[0]
        # Should be at lr_decay_frac * lr = 0.1 * 1e-3 = 1e-4
        assert abs(final_lr - 1e-4) < 1e-7

    def test_does_not_decay_below_floor(self):
        model = _make_model()
        opt = create_optimizer(model, use_muon=False, lr=1e-3)
        sched = create_scheduler(opt, warmup_steps=0, max_steps=50, lr_decay_frac=0.2)
        # Step well past max_steps
        self._step_scheduler(sched, opt, 100)
        final_lr = sched.get_last_lr()[0]
        assert final_lr >= 1e-3 * 0.2 - 1e-8

    def test_warmup_then_decay(self):
        model = _make_model()
        opt = create_optimizer(model, use_muon=False, lr=1e-3)
        sched = create_scheduler(opt, warmup_steps=10, max_steps=100, lr_decay_frac=0.1)
        lrs = self._step_scheduler(sched, opt, 100)
        # Peak should be near step 10
        peak_lr = max(lrs[:15])
        final_lr = lrs[-1]
        assert peak_lr > final_lr


# ===================================================================
# P3: Adam betas default
# ===================================================================

class TestBetasDefault:
    def test_default_betas_095(self):
        model = _make_model()
        opt = create_optimizer(model, use_muon=False)
        for g in opt.param_groups:
            assert g["betas"] == (0.9, 0.95)

    def test_custom_betas(self):
        model = _make_model()
        opt = create_optimizer(model, use_muon=False, betas=(0.9, 0.999))
        for g in opt.param_groups:
            assert g["betas"] == (0.9, 0.999)


# ===================================================================
# P4: dtype config
# ===================================================================

class TestDtypeConfig:
    """Test that dtype config is parsed correctly in the trainer."""

    def test_dtype_map_keys(self):
        from train.trainer import _DTYPE_MAP
        assert "float32" in _DTYPE_MAP
        assert "float16" in _DTYPE_MAP
        assert "bfloat16" in _DTYPE_MAP

    def test_float32_no_scaler(self):
        from train.trainer import _DTYPE_MAP
        dtype, use_scaler = _DTYPE_MAP["float32"]
        assert dtype == torch.float32
        assert use_scaler is False

    def test_float16_uses_scaler(self):
        from train.trainer import _DTYPE_MAP
        dtype, use_scaler = _DTYPE_MAP["float16"]
        assert dtype == torch.float16
        assert use_scaler is True

    def test_bfloat16_no_scaler(self):
        from train.trainer import _DTYPE_MAP
        dtype, use_scaler = _DTYPE_MAP["bfloat16"]
        assert dtype == torch.bfloat16
        assert use_scaler is False


# ===================================================================
# Integration: model + optimizer + scheduler
# ===================================================================

class TestIntegration:
    """End-to-end: create model, optimizer, scheduler, run a few steps."""

    @pytest.mark.parametrize("use_muon", [True, False])
    def test_training_step_runs(self, use_muon):
        model = _make_model()
        opt_result = create_optimizer(model, use_muon=use_muon, lr=1e-3, muon_lr=0.02)
        if isinstance(opt_result, Optimizers):
            optimizer = opt_result.muon
        else:
            optimizer = opt_result
        scheduler = create_scheduler(optimizer, warmup_steps=2, max_steps=10)

        x = torch.randint(0, V, (2, 8))
        for _ in range(3):
            optimizer.zero_grad()
            logits = model(x)
            loss = logits.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

    @pytest.mark.parametrize("norm_place", NORM_PLACES)
    @pytest.mark.parametrize("attn_type", ["quadratic", "softmax"])
    def test_all_norm_attn_combos(self, norm_place, attn_type):
        model = _make_model(norm_place=norm_place, attn_type=attn_type)
        x = torch.randint(0, V, (2, 8))
        logits = model(x)
        assert logits.shape == (2, 8, V)
        assert torch.isfinite(logits).all()
