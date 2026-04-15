"""Cheap parity tests for batched TN similarity."""

import torch
import pytest

from models import AttentionLM
from tn_sim import cosine_similarity, cosine_similarity_batch, compute_tn_similarity_batch
from tn_sim.test_similarity import make_tn_compatible_config, DTYPE
from models.components import AttentionLMComponent
from src.components.similarity import similarity as src_similarity


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    pytest.skip("CUDA is required for batched TN similarity tests.")


@pytest.mark.parametrize("attn_type", ["bilinear", "quadratic"])
def test_cosine_similarity_batch_matches_pairwise(attn_type):
    torch.manual_seed(0)
    device = _device()
    cfg = make_tn_compatible_config(
        n_ctx=2,
        d_model=4,
        n_head=1,
        n_layers=1,
        attn_type=attn_type,
    )

    model_a1 = AttentionLM.from_config(cfg).to(device=device, dtype=DTYPE)
    torch.manual_seed(1)
    model_b1 = AttentionLM.from_config(cfg).to(device=device, dtype=DTYPE)
    torch.manual_seed(2)
    model_a2 = AttentionLM.from_config(cfg).to(device=device, dtype=DTYPE)
    torch.manual_seed(3)
    model_b2 = AttentionLM.from_config(cfg).to(device=device, dtype=DTYPE)

    sims_batch = cosine_similarity_batch(
        [model_a1, model_a2],
        [model_b1, model_b2],
        device=device,
        dtype=DTYPE,
    )
    sims_pair = [
        cosine_similarity(model_a1, model_b1, device=device, dtype=DTYPE),
        cosine_similarity(model_a2, model_b2, device=device, dtype=DTYPE),
    ]

    torch.testing.assert_close(
        torch.tensor(sims_batch, dtype=DTYPE),
        torch.tensor(sims_pair, dtype=DTYPE),
    )


def test_state_batch_shape():
    torch.manual_seed(4)
    device = _device()
    cfg = make_tn_compatible_config(n_ctx=2, d_model=4, n_head=1, n_layers=1)

    model_a = AttentionLM.from_config(cfg).to(device=device, dtype=DTYPE)
    torch.manual_seed(5)
    model_b = AttentionLM.from_config(cfg).to(device=device, dtype=DTYPE)

    state = compute_tn_similarity_batch([model_a], [model_b], device=device, dtype=DTYPE)
    assert state.S_aa.shape[0] == 1
    assert state.S_aa.shape == state.S_bb.shape == state.S_ab.shape


@pytest.mark.parametrize("attn_type", ["bilinear", "quadratic"])
def test_batched_state_matches_src(attn_type):
    torch.manual_seed(10)
    device = _device()
    cfg = make_tn_compatible_config(
        n_ctx=2,
        d_model=4,
        n_head=1,
        n_layers=1,
        attn_type=attn_type,
    )

    model_a1 = AttentionLM.from_config(cfg).to(device=device, dtype=DTYPE)
    torch.manual_seed(11)
    model_b1 = AttentionLM.from_config(cfg).to(device=device, dtype=DTYPE)
    torch.manual_seed(12)
    model_a2 = AttentionLM.from_config(cfg).to(device=device, dtype=DTYPE)
    torch.manual_seed(13)
    model_b2 = AttentionLM.from_config(cfg).to(device=device, dtype=DTYPE)

    comp_a1 = AttentionLMComponent.from_trained_model(model_a1).to(device=device, dtype=DTYPE)
    comp_b1 = AttentionLMComponent.from_trained_model(model_b1).to(device=device, dtype=DTYPE)
    comp_a2 = AttentionLMComponent.from_trained_model(model_a2).to(device=device, dtype=DTYPE)
    comp_b2 = AttentionLMComponent.from_trained_model(model_b2).to(device=device, dtype=DTYPE)

    batched = compute_tn_similarity_batch(
        [comp_a1, comp_a2],
        [comp_b1, comp_b2],
        device=device,
        dtype=DTYPE,
    )

    src_1 = src_similarity(comp_a1, comp_b1)
    src_2 = src_similarity(comp_a2, comp_b2)

    torch.testing.assert_close(batched.S_aa[0], src_1.S_aa)
    torch.testing.assert_close(batched.S_bb[0], src_1.S_bb)
    torch.testing.assert_close(batched.S_ab[0], src_1.S_ab)

    torch.testing.assert_close(batched.S_aa[1], src_2.S_aa)
    torch.testing.assert_close(batched.S_bb[1], src_2.S_bb)
    torch.testing.assert_close(batched.S_ab[1], src_2.S_ab)
