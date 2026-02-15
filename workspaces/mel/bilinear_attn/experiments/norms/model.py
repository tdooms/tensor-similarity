"""Model wrapper for normalization experiments."""
import math
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.transformer import MaxRMSNorm, CausalMaxRMSNorm
from experiments.norms.attention_kernels import (
    BilinearAttentionNorm,
    QuadraticAttentionNorm,
    SoftmaxAttentionNorm,
)


ATTN_REGISTRY_NORM = {
    "bilinear": BilinearAttentionNorm,
    "quadratic": QuadraticAttentionNorm,
    "softmax": SoftmaxAttentionNorm,
}

NORM_TYPES = ("none", "rmsnorm", "layernorm", "maxrmsnorm", "causal_maxrmsnorm")
VALID_NORM_PLACES = ("post_embed", "pre_layer", "pre_unembed")


class AttentionLMNorm(nn.Module):
    """Autoregressive LM with configurable Q/K normalization.
    
    This is a modified version of AttentionLM that supports:
    - qk_norm_type: Type of Q/K normalization ('none', 'rmsnorm', 'alpha_head')
    - alpha_init: Initial value for alpha parameters (if using alpha_head)
    
    All other parameters match the original AttentionLM.
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
        qk_norm_type: str = "none",
        alpha_init: float = 1.0,
        use_bias_qk: bool = True,
        std_embed: float = 0.02,
        std_qkv: float = 0.02,
        std_o: float = 0.01,
        attn_type: str = "quadratic",
        norm_type: str = "rmsnorm",
        norm_places: list[str] | None = None,
        init_type: str = "normal",
        init_gain: float = 1.0,
    ) -> None:
        super().__init__()
        if norm_places is None:
            norm_places = []
        assert norm_type in NORM_TYPES, f"norm_type must be one of {NORM_TYPES}, got {norm_type!r}"
        for p in norm_places:
            assert p in VALID_NORM_PLACES, f"each norm_place must be one of {VALID_NORM_PLACES}, got {p!r}"
        
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.n_head = n_head
        self.n_layers = n_layers
        self.attn_type = attn_type
        self.norm_type = norm_type
        self.norm_places = norm_places
        self.init_type = init_type
        self.qk_norm_type = qk_norm_type
        
        attn_cls = ATTN_REGISTRY_NORM[attn_type]
        
        def _make_norm():
            if norm_type == "rmsnorm":
                return nn.RMSNorm(d_model)
            elif norm_type == "layernorm":
                return nn.LayerNorm(d_model)
            elif norm_type == "maxrmsnorm":
                return MaxRMSNorm(d_model)
            elif norm_type == "causal_maxrmsnorm":
                return CausalMaxRMSNorm(d_model)
            else:
                return nn.Identity()
        
        self.embed = nn.Embedding(vocab_size, d_model)
        
        if "post_embed" in norm_places:
            self.embed_norm = _make_norm()
        else:
            self.embed_norm = None
        
        if "pre_unembed" in norm_places or "pre_layer" in norm_places:
            self.final_norm = _make_norm()
        else:
            self.final_norm = nn.Identity()
        
        if "pre_layer" in norm_places:
            self.layer_norms = nn.ModuleList([_make_norm() for _ in range(n_layers)])
        else:
            self.layer_norms = None
        
        self.layers = nn.ModuleList([
            attn_cls(
                d_model=d_model,
                n_head=n_head,
                n_ctx=n_ctx,
                scale=attn_scale,
                qk_norm_type=qk_norm_type,
                alpha_init=alpha_init,
                use_bias_qk=use_bias_qk,
                rope_base=rope_base,
            )
            for _ in range(n_layers)
        ])
        
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)
        
        self._init_weights(std_embed, std_qkv, std_o, init_type, init_gain)
    
    def _init_weights(
        self,
        std_embed: float,
        std_qkv: float,
        std_o: float,
        init_type: str = "normal",
        init_gain: float = 1.0,
    ) -> None:
        """Initialize weights."""
        assert init_type in ("normal", "orthogonal", "mup"), (
            f"init_type must be 'normal', 'orthogonal', or 'mup', got {init_type!r}"
        )
        
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
        
        else:
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
        """Forward pass."""
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
    def from_config(cls, cfg: dict) -> "AttentionLMNorm":
        """Create model from config dict."""
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
            qk_norm_type=model_cfg.get("qk_norm_type", "none"),
            alpha_init=model_cfg.get("alpha_init", 1.0),
            use_bias_qk=model_cfg.get("use_bias_qk", True),
            std_embed=init_cfg.get("std_embed", 0.02),
            std_qkv=init_cfg.get("std_qkv", 0.02),
            std_o=init_cfg.get("std_o", 0.01),
            attn_type=model_cfg.get("attn_type", "quadratic"),
            norm_type=model_cfg.get("norm_type", "rmsnorm"),
            norm_places=model_cfg.get("norm_places", []),
            init_type=init_cfg.get("init_type", "normal"),
            init_gain=init_cfg.get("init_gain", 1.0),
        )
