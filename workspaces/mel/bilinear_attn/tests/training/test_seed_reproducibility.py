"""Seed reproducibility tests for the pieces not already covered by
``tests.models.test_determinism`` (model init / forward pass):

* induction-head data generation is seed-stable
* a Muon training step produces identical loss + post-step params across runs
"""
import torch

from models import AttentionLM
from train.optim import Optimizers, create_optimizer


_CFG = {
    "model": {
        "vocab_size": 256, "n_ctx": 16, "d_model": 16, "n_head": 4, "n_layers": 2,
        "attn_type": "bilinear", "attn_scale": 0.35, "rope_base": 10000,
        "norm_type": "none", "norm_places": [],
        "use_rmsnorm_qk": False, "use_bias_qk": True,
    },
    "init": {"std_embed": 0.02, "std_qkv": 0.02, "std_o": 0.01},
}


def test_data_generation_seed_stable():
    """Dataset generation is seed-stable.

    Note: ``create_repeated_token_dataloaders`` uses ``shuffle=True`` on the
    DataLoader, which consumes torch's global RNG, so ``seed=`` alone does
    not pin the batch order. We seed torch globally before each call to
    verify end-to-end reproducibility given a pinned environment.
    """
    from experiments.induction_heads.data import create_repeated_token_dataloaders

    def first_batch():
        torch.manual_seed(0)
        train_dl, _ = create_repeated_token_dataloaders(
            vocab_size=256, n_ctx=8, batch_size=64,
            n_train=1000, n_val=100, seed=42,
        )
        return next(iter(train_dl))["input_ids"].clone()

    a, b = first_batch(), first_batch()
    assert torch.equal(a, b)


def _one_muon_step():
    torch.manual_seed(42)
    model = AttentionLM.from_config(_CFG)
    model.train()
    result = create_optimizer(
        model, lr=3e-4, muon_lr=0.02, weight_decay=0.1,
        betas=(0.9, 0.95), use_muon=True,
    )
    optimizer = result.muon if isinstance(result, Optimizers) else result

    torch.manual_seed(99)
    input_ids = torch.randint(0, 256, (2, 16))
    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, 256), input_ids[:, 1:].reshape(-1),
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), sum(p.sum().item() for p in model.parameters())


def test_muon_training_step_reproducible():
    loss_a, sum_a = _one_muon_step()
    loss_b, sum_b = _one_muon_step()
    assert abs(loss_a - loss_b) < 1e-6
    assert abs(sum_a - sum_b) < 1e-4
