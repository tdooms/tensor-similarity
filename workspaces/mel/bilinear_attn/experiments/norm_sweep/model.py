"""Model wrapper for the norm sweep experiment.

Uses the same architecture as AttentionLM but dispatches norm_type through
the norm_sweep registry so all custom norms are available.
"""

import math
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.attention_kernels.bilinear import BilinearAttention, QuadraticAttention
from models.attention_kernels.softmax import SoftmaxAttention
from experiments.norm_sweep.norms import make_norm, NORM_SWEEP_REGISTRY

ATTN_REGISTRY = {
    "bilinear": BilinearAttention,
    "quadratic": QuadraticAttention,
    "softmax": SoftmaxAttention,
}

VALID_NORM_PLACES = ("post_embed", "pre_layer", "pre_unembed")


class NormSweepLM(nn.Module):
    """Autoregressive LM with any norm from the norm_sweep registry.

    Identical architecture to AttentionLM; only the norm factory differs.
    """

    def __init__(
        self,
        vocab_size: int,
        n_ctx: int,
        d_model: int,
        n_head: int,
        n_layers: int,
        attn_scale: float = 0.2,
        rope_base: int = 10000,
        use_rmsnorm_qk: bool = False,
        use_bias_qk: bool = True,
        std_embed: float = 0.02,
        std_qkv: float = 0.02,
        std_o: float = 0.01,
        attn_type: str = "quadratic",
        norm_type: str = "rmsnorm",
        norm_places: list[str] | None = None,
        init_type: str = "normal",
        init_gain: float = 1.0,
        norm_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        if norm_places is None:
            norm_places = []
        for p in norm_places:
            assert p in VALID_NORM_PLACES, f"Invalid norm_place {p!r}"

        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.n_head = n_head
        self.n_layers = n_layers
        self.attn_type = attn_type
        self.norm_type = norm_type
        self.norm_places = norm_places
        self.init_type = init_type

        nk = norm_kwargs or {}
        attn_cls = ATTN_REGISTRY[attn_type]

        self.embed = nn.Embedding(vocab_size, d_model)

        if "post_embed" in norm_places:
            self.embed_norm = make_norm(norm_type, d_model, **nk)
        else:
            self.embed_norm = None

        if "pre_unembed" in norm_places or "pre_layer" in norm_places:
            self.final_norm = make_norm(norm_type, d_model, **nk)
        else:
            self.final_norm = nn.Identity()

        if "pre_layer" in norm_places:
            self.layer_norms = nn.ModuleList(
                [make_norm(norm_type, d_model, **nk) for _ in range(n_layers)]
            )
        else:
            self.layer_norms = None

        self.layers = nn.ModuleList([
            attn_cls(
                d_model=d_model,
                n_head=n_head,
                n_ctx=n_ctx,
                scale=attn_scale,
                use_rmsnorm_qk=use_rmsnorm_qk,
                use_bias_qk=use_bias_qk,
                rope_base=rope_base,
            )
            for _ in range(n_layers)
        ])

        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights(std_embed, std_qkv, std_o, init_type, init_gain)

    def _init_weights(self, std_embed, std_qkv, std_o, init_type="normal", init_gain=1.0):
        assert init_type in ("normal", "orthogonal", "mup")

        if init_type == "normal":
            nn.init.normal_(self.embed.weight, mean=0.0, std=std_embed)
            nn.init.normal_(self.unembed.weight, mean=0.0, std=std_embed)
            for layer in self.layers:
                if hasattr(layer, "q1"):
                    qk_projs = [layer.q1, layer.k1, layer.q2, layer.k2]
                else:
                    qk_projs = [layer.q, layer.k]
                for proj in qk_projs:
                    nn.init.normal_(proj.weight, mean=0.0, std=std_qkv)
                    if proj.bias is not None:
                        nn.init.zeros_(proj.bias)
                nn.init.normal_(layer.v.weight, mean=0.0, std=std_qkv)
                nn.init.normal_(layer.o.weight, mean=0.0, std=std_o)

        elif init_type == "orthogonal":
            nn.init.normal_(self.embed.weight, mean=0.0, std=std_embed)
            nn.init.normal_(self.unembed.weight, mean=0.0, std=std_embed)
            for layer in self.layers:
                if hasattr(layer, "q1"):
                    qk_projs = [layer.q1, layer.k1, layer.q2, layer.k2]
                else:
                    qk_projs = [layer.q, layer.k]
                for proj in qk_projs:
                    nn.init.orthogonal_(proj.weight, gain=init_gain)
                    if proj.bias is not None:
                        nn.init.zeros_(proj.bias)
                nn.init.orthogonal_(layer.v.weight, gain=init_gain)
                nn.init.orthogonal_(layer.o.weight, gain=init_gain)

        else:  # mup
            nn.init.normal_(self.embed.weight, mean=0.0, std=std_embed)
            nn.init.normal_(self.unembed.weight, mean=0.0, std=std_embed)
            d_head = self.d_model // self.n_head
            std_qk = 1.0 / math.sqrt(d_head)
            std_v = 1.0 / math.sqrt(self.d_model)
            std_o_mup = 1.0 / (math.sqrt(self.d_model) * math.sqrt(self.n_layers))
            for layer in self.layers:
                if hasattr(layer, "q1"):
                    qk_projs = [layer.q1, layer.k1, layer.q2, layer.k2]
                else:
                    qk_projs = [layer.q, layer.k]
                for proj in qk_projs:
                    nn.init.normal_(proj.weight, mean=0.0, std=std_qk)
                    if proj.bias is not None:
                        nn.init.zeros_(proj.bias)
                nn.init.normal_(layer.v.weight, mean=0.0, std=std_v)
                nn.init.normal_(layer.o.weight, mean=0.0, std=std_o_mup)

    def forward(self, input_ids: torch.Tensor, return_debug: bool = False):
        x = self.embed(input_ids)

        if self.embed_norm is not None:
            x = self.embed_norm(x)

        debug_list = [] if return_debug else None

        if self.layer_norms is not None:
            for norm, layer in zip(self.layer_norms, self.layers):
                x_normed = norm(x)
                if return_debug:
                    out, debug = layer(x_normed, return_debug=True)
                    debug_list.append(debug)
                else:
                    out = layer(x_normed)
                x = x + (out - x_normed)
        else:
            for layer in self.layers:
                if return_debug:
                    x, debug = layer(x, return_debug=True)
                    debug_list.append(debug)
                else:
                    x = layer(x)

        x = self.final_norm(x)
        logits = self.unembed(x)

        if return_debug:
            return logits, debug_list
        return logits

    @classmethod
    def from_config(cls, cfg: dict) -> "NormSweepLM":
        model_cfg = cfg["model"]
        init_cfg = cfg.get("init", {})
        return cls(
            vocab_size=model_cfg["vocab_size"],
            n_ctx=model_cfg["n_ctx"],
            d_model=model_cfg["d_model"],
            n_head=model_cfg["n_head"],
            n_layers=model_cfg["n_layers"],
            attn_scale=model_cfg.get("attn_scale", 0.2),
            rope_base=model_cfg.get("rope_base", 10000),
            use_rmsnorm_qk=model_cfg.get("use_rmsnorm_qk", False),
            use_bias_qk=model_cfg.get("use_bias_qk", True),
            std_embed=init_cfg.get("std_embed", 0.02),
            std_qkv=init_cfg.get("std_qkv", 0.02),
            std_o=init_cfg.get("std_o", 0.01),
            attn_type=model_cfg.get("attn_type", "quadratic"),
            norm_type=model_cfg.get("norm_type", "rmsnorm"),
            norm_places=model_cfg.get("norm_places", []),
            init_type=init_cfg.get("init_type", "normal"),
            init_gain=init_cfg.get("init_gain", 1.0),
            norm_kwargs=model_cfg.get("norm_kwargs", {}),
        )
