import pytest
import torch
from src.components.paths import (
    residual_tn, mlp_active_tn, get_active_tn, get_residual_tn,
    contract_tn_pair, pad_gram, contract_path, order_stratified_similarity,
)
from src.components.mlp import MLP
from src.components.attention import Attention
from src.models.transformer import Transformer


class TestResidualTN:
    def test_identity_self_contraction(self):
        """Residual TN with scale=0 is identity; self-contraction gives I."""
        d = 5
        tn = residual_tn(d, scale=0)
        gram, exp = contract_tn_pair(tn, tn)
        torch.testing.assert_close(gram * 10**exp, torch.eye(d))

    def test_scale_half(self):
        """Residual TN with scale=0.5 gives 0.25 * I."""
        d = 4
        tn = residual_tn(d, scale=0.5)
        gram, exp = contract_tn_pair(tn, tn)
        expected = 0.25 * torch.eye(d)
        torch.testing.assert_close(gram * 10**exp, expected, rtol=1e-4, atol=1e-4)

    def test_with_inner_gram(self):
        """Residual TN propagates inner Gram linearly: (1-s)^2 * G."""
        d = 4
        tn = residual_tn(d, scale=0.3)
        inner = torch.randn(d, d)
        gram, exp = contract_tn_pair(tn, tn, inner=inner)
        expected = 0.7**2 * inner
        torch.testing.assert_close(gram * 10**exp, expected, rtol=1e-4, atol=1e-4)


class TestMlpActiveTN:
    def test_has_two_inputs(self):
        """Active MLP TN should have 2 input indices."""
        torch.manual_seed(0)
        mlp = MLP(d_model=4, d_hidden=6, scale=1)
        tn = mlp_active_tn(mlp)
        n_inputs = sum(1 for idx in tn.ind_map if idx.startswith('in:d'))
        assert n_inputs == 2

    def test_output_dim_is_padded(self):
        """Active MLP TN output dimension should be d_model+1."""
        torch.manual_seed(0)
        mlp = MLP(d_model=4, d_hidden=6, scale=1)
        tn = mlp_active_tn(mlp)
        assert tn.ind_size('out:d') == 5  # d_model + 1


class TestPadGram:
    def test_no_pad_needed(self):
        gram = torch.eye(5)
        result = pad_gram(gram, 5)
        torch.testing.assert_close(result, gram)

    def test_pad_embeds_bottom_right(self):
        gram = torch.ones(3, 3)
        result = pad_gram(gram, 5)
        assert result.shape == (5, 5)
        assert result[0, 0] == 0.0
        torch.testing.assert_close(result[2:, 2:], gram)


class TestContractPath:
    def test_all_residual_single_mlp(self):
        """All-residual path for a single MLP."""
        torch.manual_seed(0)
        mlp = MLP(d_model=4, d_hidden=6, scale=0.5)
        val = contract_path([mlp], [mlp], mask=0)
        # (1-0.5)^2 * trace(I_{5}) = 0.25 * 5 = 1.25
        expected = torch.tensor(1.25).log10().item()
        assert abs(val - expected) < 1e-4, f"got {val}, expected {expected}"

    def test_all_active_single_mlp(self):
        """Single active MLP path gives finite result."""
        torch.manual_seed(0)
        mlp = MLP(d_model=4, d_hidden=6, scale=1)
        val = contract_path([mlp], [mlp], mask=1)
        assert val != float('-inf')
        assert val > 0


class TestOrderStratified:
    def test_transformer_has_all_orders(self):
        """A 4-component transformer should produce orders 0..4."""
        torch.manual_seed(42)
        t = Transformer(d_model=4, n_head=2, n_ctx=3, d_hidden=6, scale=0.5)
        strat = order_stratified_similarity(t, t)
        assert set(strat.keys()) == {0, 1, 2, 3, 4}

    def test_scale_one_residual_is_zero(self):
        """With scale=1, residual path (mask=0) should give order-0 = 0 (log = -inf)."""
        torch.manual_seed(42)
        mlp_a = MLP(d_model=4, d_hidden=6, scale=1.0)
        mlp_b = MLP(d_model=4, d_hidden=6, scale=1.0)

        from src.models.base import Model

        class SingleMLP(Model):
            def __init__(self, mlp):
                super().__init__(None)
                self.mlp = mlp

            def components(self):
                return [self.mlp]

        strat = order_stratified_similarity(SingleMLP(mlp_a), SingleMLP(mlp_b))
        # scale=1 means (1-scale)=0, so residual contribution is zero
        assert strat[0] == float('-inf')
        assert strat[1] != float('-inf')

    def test_single_mlp_full_vs_active_plus_residual(self):
        """For scale=1 MLP, the full network() equals the active-only TN."""
        torch.manual_seed(42)
        mlp = MLP(d_model=4, d_hidden=6, scale=1.0)

        # Full network contraction (from Component.contract)
        full_gram, full_exp = mlp.contract()

        # Active-only TN contraction
        active = mlp_active_tn(mlp)
        active_gram, active_exp = contract_tn_pair(active, active)

        full_trace = (full_gram * 10**full_exp).trace()
        active_trace = (active_gram * 10**active_exp).trace()
        torch.testing.assert_close(full_trace, active_trace, rtol=1e-3, atol=1e-3)
