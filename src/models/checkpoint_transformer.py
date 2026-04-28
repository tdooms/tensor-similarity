"""TN-flavoured mirror of melephant `AttentionLM` checkpoints.

`AttentionLM`'s inference-time `tok0_batch` norm is a single batch-wide scalar
(`1/√(running_mean_energy + ε)`) applied at `pre_unembed`. It is exactly
absorbable into `unembed.weight` at load time — see `_absorb_final_norm` —
which lets us reuse the existing TN graph (embed, attn ×N, unembed) without
adding a `Norm` component or any approximation.

`n_ctx` is an *analysis* parameter (the sequence length the similarity tensor
spans), not the model's native context length. The native length is 512 for
`melephant/2l-bilinear-attn-normalised-v2`; we cap it to a small value here
(default 4) because the Σ tensor is `(n_ctx, d+1, n_ctx, d+1)` — quadratic
in `n_ctx` — and contraction cost is super-quadratic. The state-dict is
`n_ctx`-invariant (rotary cos/sin and mask are non-persistent buffers).
"""
import json

import torch
from torch import nn
from quimb.tensor import Tensor, TensorNetwork

from src.components.attention import Attention
from src.components.base import Term
from src.components.linear import Linear


class CheckpointAttention(Attention):
    """Bilinear-attn block with a *full* residual (`out = x + scale·Wo(z)`).

    The base `Attention` uses `torch.lerp` semantics
    (`out = (1-s)·x + s·Wo(z)`), so its `.terms()` puts `(1-s)·I` on the
    residual term. The melephant model uses a plain additive residual, so we
    override `.terms()` to put a full identity there. The active term still
    carries `s` because `Attention.network()` already bakes `scale` into
    `Wo`.
    """

    def terms(self, n_ctx):
        d, like = self.d_model, self._like()
        active = self.network().reindex({"out:d": "mid:d"})
        lift = torch.zeros(d + 1, d, **like)
        lift[1:] = torch.eye(d, **like)
        active &= Tensor(lift, inds=("out:d", "mid:d"))
        legs = {"in:d0": "in:s", "in:d1": "in:s", "in:d2": "in:s",
                "in:d3": "out:s", "in:d4": "out:s"}
        return [
            Term(TensorNetwork([Tensor(torch.eye(d + 1, **like), inds=("out:d", "in:d0"))]), {"in:d0": "out:s"}),
            Term(active, legs, symmetries=((0, 2, 1, 4, 3),)),
        ]


class CheckpointTransformer(nn.Module):
    """Embed → N × CheckpointAttention → Unembed."""

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        n_head: int,
        n_layers: int,
        n_ctx: int = 4,
        attn_scale: float = 0.35,
        use_bias_qk: bool = True,
    ) -> None:
        super().__init__()
        self.n_ctx = n_ctx
        self.embed = Linear(vocab_size, d_model, bias=False)
        self.layers = nn.ModuleList([
            CheckpointAttention(d_model, n_head, n_ctx, mask="causal", bias=use_bias_qk, scale=attn_scale)
            for _ in range(n_layers)
        ])
        self.unembed = Linear(d_model, vocab_size, bias=False)

    def components(self):
        return [self.embed, *self.layers, self.unembed]

    @classmethod
    def from_hf_config(cls, config: dict, n_ctx: int = 4) -> "CheckpointTransformer":
        """Build from a melephant `config.json` — `config["model"]` block."""
        m = config["model"] if "model" in config else config
        assert m.get("attn_type", "bilinear") == "bilinear", f"only bilinear attn supported, got {m.get('attn_type')!r}"
        return cls(
            vocab_size=m["vocab_size"],
            d_model=m["d_model"],
            n_head=m["n_head"],
            n_layers=m["n_layers"],
            n_ctx=n_ctx,
            attn_scale=m.get("attn_scale", 0.35),
            use_bias_qk=m.get("use_bias_qk", True),
        )


def _absorb_final_norm(state: dict, eps: float = 1e-6) -> dict:
    """Fold the `Tok0Batch` `pre_unembed` scale into `unembed.weight`.

    `Tok0Batch` at eval scales by `1/√(running_mean_energy + ε)` — input-
    independent — so it absorbs *exactly* into `unembed`. The buffer's init
    value is `1.0`, so the rescale is a no-op for fresh checkpoints, which is
    why we apply it unconditionally. Returns `state` with `final_norm.*` keys
    dropped and `unembed.weight` rescaled.
    """
    out = {k: v for k, v in state.items() if not k.startswith("final_norm.")}
    energy = state.get("final_norm.running_mean_energy", torch.ones(1)).to(torch.float32)
    out["unembed.weight"] = (out["unembed.weight"].to(torch.float32) * (energy + eps).rsqrt()).to(out["unembed.weight"].dtype)
    return out


def load_state_into(model: CheckpointTransformer, state: dict) -> CheckpointTransformer:
    """Strict-load a melephant `model_state_dict` after norm absorption + embed transpose.

    Local `Linear.weight` is `(d_out, d_in)` like `nn.Linear`, while the HF
    `embed.weight` is stored as `(vocab, d_model)` (rows are token vectors).
    We transpose to `(d_model, vocab)` so it lines up with `Linear(vocab, d)`.
    """
    state = _absorb_final_norm(state)
    state = dict(state)
    state["embed.weight"] = state["embed.weight"].T.contiguous()
    model.load_state_dict({k: v.cuda() for k, v in state.items()})
    return model


@torch.no_grad()
def load_pt(path, config: dict, n_ctx: int = 4) -> CheckpointTransformer:
    """Build a `CheckpointTransformer` from a melephant `.pt` checkpoint, on GPU."""
    blob = torch.load(path, map_location="cuda", weights_only=False)
    state = blob["model_state_dict"] if "model_state_dict" in blob else blob
    model = CheckpointTransformer.from_hf_config(config, n_ctx=n_ctx).cuda().eval()
    return load_state_into(model, state)


def load_config(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
