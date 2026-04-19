"""Adapter parity: wrapped mel vs. a reference model built from
``src.components`` primitives.

mel's attention now uses the same ``lerp(x, o(z), scale)`` residual
convention as ``src.components.attention.Attention``, so the reference is
a direct ``src``-primitive build with the mel weights copied in.

What this validates:

1. **State parity** — ``similarity(wrapper, wrapper)`` and
   ``similarity(reference, reference)`` produce element-wise identical
   State tensors (``s_aa``/``s_ab``/``s_bb``).
2. **Cross-model parity** — the cross state also matches.
3. **Forward-pass parity** — contracting the adapter TN with an explicit
   input (via sum over terms) reproduces mel's forward output.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn
from quimb.tensor import Tensor

# Import order matters: either of these sets up sys.path for ``src``.
from models import AttentionLM
from models.components import AttentionLMComponent
import tn_sim  # noqa: F401  (path side-effect)

from src.components.attention import Attention as SrcAttention
from src.components.linear import Linear as SrcLinear
from src.components.similarity import similarity as src_similarity
from src.models.base import Model as SrcModel


DTYPE = torch.float64
DEVICE = "cpu"


class _RefModel(SrcModel):
    """Reference Model built from ``src.components`` primitives."""

    def __init__(self, mel_model: AttentionLM) -> None:
        super().__init__(None)
        V, D = mel_model.vocab_size, mel_model.d_model
        n_head, n_ctx = mel_model.n_head, mel_model.n_ctx
        bias = mel_model.layers[0].q1.bias is not None
        scale = mel_model.layers[0].scale

        self.embed = SrcLinear(V, D, bias=False)
        self.layers = nn.ModuleList([
            SrcAttention(D, n_head, n_ctx, mask='causal', bias=bias, scale=scale)
            for _ in range(mel_model.n_layers)
        ])
        self.unembed = SrcLinear(D, V, bias=False)
        self._copy_from(mel_model)

    @torch.no_grad()
    def _copy_from(self, m: AttentionLM) -> None:
        # nn.Embedding.weight is (V, D); SrcLinear(V, D).weight is (D, V).
        self.embed.weight.copy_(m.embed.weight.T)
        for ref, src in zip(self.layers, m.layers):
            for name in ("q1", "k1", "q2", "k2", "v", "o"):
                getattr(ref, name).weight.copy_(getattr(src, name).weight)
                if getattr(ref, name).bias is not None:
                    getattr(ref, name).bias.copy_(getattr(src, name).bias)
        self.unembed.weight.copy_(m.unembed.weight)

    def components(self):
        return [self.embed] + list(self.layers) + [self.unembed]


def _make_mel_model(
    *,
    vocab_size: int = 8,
    n_ctx: int = 4,
    d_model: int = 8,
    n_head: int = 2,
    n_layers: int = 1,
    attn_scale: float = 0.5,
    use_bias_qk: bool = True,
    seed: int = 0,
) -> AttentionLM:
    torch.manual_seed(seed)
    cfg = {
        "model": {
            "vocab_size": vocab_size,
            "n_ctx": n_ctx,
            "d_model": d_model,
            "n_head": n_head,
            "n_layers": n_layers,
            "attn_scale": attn_scale,
            "attn_type": "bilinear",
            "use_bias_qk": use_bias_qk,
            "use_rmsnorm_qk": False,
            "norm_type": "none",
            "norm_places": [],
            "rope_base": 10000,
        },
        "init": {"std_embed": 0.1, "std_qkv": 0.1, "std_o": 0.1},
    }
    return AttentionLM.from_config(cfg).to(dtype=DTYPE)


def _state_trace(state) -> tuple[float, float, float]:
    tr = lambda s: torch.einsum('ijij->', s[:, 1:, :, 1:]).item()
    return tr(state.s_aa), tr(state.s_ab), tr(state.s_bb)


class TestAdapterParity:
    """Wrapped mel and a src-primitive reference (both residual-add) must
    produce numerically identical second-moment states."""

    @pytest.mark.parametrize("n_layers", [1, 2])
    @pytest.mark.parametrize("use_bias_qk", [False, True])
    def test_self_state_matches_reference(self, n_layers, use_bias_qk):
        mel = _make_mel_model(n_layers=n_layers, use_bias_qk=use_bias_qk, seed=0)

        wrapped = AttentionLMComponent.from_trained_model(mel).to(dtype=DTYPE)
        reference = _RefModel(mel).to(dtype=DTYPE)

        state_w = src_similarity(wrapped, wrapped)
        state_r = src_similarity(reference, reference)

        for name in ("s_aa", "s_ab", "s_bb"):
            w = getattr(state_w, name)
            r = getattr(state_r, name)
            max_abs_diff = (w - r).abs().max().item()
            max_rel_diff = (max_abs_diff / max(r.abs().max().item(), 1e-30))
            assert max_abs_diff < 1e-10 or max_rel_diff < 1e-8, (
                f"{name}: max |Δ| = {max_abs_diff:.3e}, "
                f"max |Δ|/|r| = {max_rel_diff:.3e}"
            )

    def test_cross_state_matches_reference(self):
        mel_a = _make_mel_model(n_layers=1, seed=0)
        mel_b = _make_mel_model(n_layers=1, seed=1)

        wrapped_a = AttentionLMComponent.from_trained_model(mel_a).to(dtype=DTYPE)
        wrapped_b = AttentionLMComponent.from_trained_model(mel_b).to(dtype=DTYPE)
        ref_a = _RefModel(mel_a).to(dtype=DTYPE)
        ref_b = _RefModel(mel_b).to(dtype=DTYPE)

        tr_w = _state_trace(src_similarity(wrapped_a, wrapped_b))
        tr_r = _state_trace(src_similarity(ref_a, ref_b))

        for name, w, r in zip(("s_aa", "s_ab", "s_bb"), tr_w, tr_r):
            tol = max(1e-10, 1e-8 * max(abs(w), abs(r)))
            assert abs(w - r) < tol, f"{name}: wrapper={w!r}, reference={r!r}"


# -----------------------------------------------------------------------------
# Forward-pass parity: the adapter's TN, when evaluated on a concrete input
# (summed across residual + active terms), must reproduce mel's forward.

def _eval_full(component, x_padded, n_ctx, *, like):
    """Sum ``component.terms()`` evaluated on ``x_padded`` (shape (T, D+1)).

    Each term's TN is contracted with a single input that's broadcast to
    every input-data leg. Sequence legs are kept as free axes: ``in:s`` for
    positions that receive the input (rows of x_padded) and ``out:s`` for
    the output positions.
    """
    T = x_padded.shape[0]
    total = None
    for term in component.terms(n_ctx, **like):
        tn = term.tn.copy()
        # Bind each data leg to the input, pinning its sequence position.
        for leg, pos in term.legs.items():
            # `leg` is an input-data index like 'in:d0'; `pos` is 'in:s' or 'out:s'.
            tn &= Tensor(x_padded, inds=(pos, leg))
        out = tn.contract(output_inds=('out:s', 'out:d')).data
        total = out if total is None else total + out
    return total  # shape (T, D+1)


class TestForwardParity:
    """Contracting the adapter's TN with an explicit input must reproduce
    mel's forward output (on a single attention layer, then on the full
    model)."""

    def test_single_attention_layer(self):
        mel = _make_mel_model(n_layers=1, seed=123, n_ctx=4, d_model=8)
        wrapped = AttentionLMComponent.from_trained_model(mel).to(dtype=DTYPE)
        attn_mel = mel.layers[0]
        attn_comp = wrapped.layers[0]

        torch.manual_seed(7)
        x = torch.randn(mel.n_ctx, mel.d_model, dtype=DTYPE)

        # Mel forward (add batch dim, run, drop batch).
        y_mel = attn_mel(x.unsqueeze(0)).squeeze(0)  # (T, D)

        # TN forward via sum of terms on padded input.
        x_padded = torch.cat([torch.ones(mel.n_ctx, 1, dtype=DTYPE), x], dim=-1)
        like = {"device": x.device, "dtype": x.dtype}
        y_tn = _eval_full(attn_comp, x_padded, mel.n_ctx, like=like)  # (T, D+1)

        # Constant axis must remain 1; data axes must match mel forward.
        assert torch.allclose(y_tn[:, 0], torch.ones(mel.n_ctx, dtype=DTYPE), atol=1e-10), \
            "TN output dropped the constant axis"
        diff = (y_tn[:, 1:] - y_mel).abs().max().item()
        assert diff < 1e-10, f"single-layer mismatch: max |Δ| = {diff:.3e}"
