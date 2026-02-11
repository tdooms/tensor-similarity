import torch
import torch.nn as nn
from .attention_kernels.bilinear import QuadraticAttention
from .attention_kernels.softmax import SoftmaxAttention

ATTN_REGISTRY = {
    "quadratic": QuadraticAttention,
    "softmax": SoftmaxAttention,
}


NORM_TYPES = ("none", "rmsnorm", "layernorm")
NORM_PLACES = ("none", "post_embed", "pre_layer", "pre_unembed")


class AttentionLM(nn.Module):
    """Autoregressive language model with configurable attention.
    
    Architecture:
        - Token embedding E
        - n_layers x Attention blocks (attention + residual only, no MLP)
        - Separate unembedding U (not tied)
    
    Supports attn_type: 'quadratic' (default) or 'softmax'.
    
    norm_type selects the normalisation function:
        'none'      – no normalization
        'rmsnorm'   – RMSNorm (default)
        'layernorm' – LayerNorm
    
    norm_place controls where normalisation is applied:
        'none'       – no normalization anywhere
        'post_embed' – norm after embedding only
        'pre_layer'  – norm before each attention layer + before unembed
        'pre_unembed'– norm only before unembed (default)
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
        use_bias_qkv: bool = True,
        use_bias_o: bool = True,
        std_embed: float = 0.02,
        std_qkv: float = 0.02,
        std_o: float = 0.01,
        attn_type: str = "quadratic",
        norm_type: str = "rmsnorm",
        norm_place: str = "pre_unembed",
    ) -> None:
        super().__init__()
        assert norm_type in NORM_TYPES, f"norm_type must be one of {NORM_TYPES}, got {norm_type!r}"
        assert norm_place in NORM_PLACES, f"norm_place must be one of {NORM_PLACES}, got {norm_place!r}"
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.n_head = n_head
        self.n_layers = n_layers
        self.attn_type = attn_type
        self.norm_type = norm_type
        self.norm_place = norm_place
        
        attn_cls = ATTN_REGISTRY[attn_type]
        
        def _make_norm():
            if norm_type == "rmsnorm":
                return nn.RMSNorm(d_model)
            elif norm_type == "layernorm":
                return nn.LayerNorm(d_model)
            else:
                return nn.Identity()
        
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # Post-embedding norm – only used by 'post_embed'
        if norm_place == "post_embed":
            self.embed_norm = _make_norm()
        else:
            self.embed_norm = None
        
        # Final norm (before unembed) – used by 'pre_unembed' and 'pre_layer'
        if norm_place in ("pre_unembed", "pre_layer"):
            self.final_norm = _make_norm()
        else:
            self.final_norm = nn.Identity()
        
        # Per-layer pre-norms – only used by 'pre_layer'
        if norm_place == "pre_layer":
            self.layer_norms = nn.ModuleList([_make_norm() for _ in range(n_layers)])
        else:
            self.layer_norms = None
        
        self.layers = nn.ModuleList([
            attn_cls(
                d_model=d_model,
                n_head=n_head,
                n_ctx=n_ctx,
                scale=attn_scale,
                use_rmsnorm_qk=use_rmsnorm_qk,
                use_bias_qkv=use_bias_qkv,
                use_bias_o=use_bias_o,
                rope_base=rope_base,
            )
            for _ in range(n_layers)
        ])
        
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)
        
        self._init_weights(std_embed, std_qkv, std_o)
    
    def _init_weights(self, std_embed: float, std_qkv: float, std_o: float) -> None:
        """Initialize weights with specified standard deviations."""
        nn.init.normal_(self.embed.weight, mean=0.0, std=std_embed)
        nn.init.normal_(self.unembed.weight, mean=0.0, std=std_embed)
        
        for layer in self.layers:
            nn.init.normal_(layer.q.weight, mean=0.0, std=std_qkv)
            nn.init.normal_(layer.k.weight, mean=0.0, std=std_qkv)
            nn.init.normal_(layer.v.weight, mean=0.0, std=std_qkv)
            nn.init.normal_(layer.o.weight, mean=0.0, std=std_o)
            
            if layer.q.bias is not None:
                nn.init.zeros_(layer.q.bias)
                nn.init.zeros_(layer.k.bias)
                nn.init.zeros_(layer.v.bias)
            if layer.o.bias is not None:
                nn.init.zeros_(layer.o.bias)
    
    def forward(self, input_ids: torch.Tensor, return_debug: bool = False):
        """Forward pass.
        
        Args:
            input_ids: Token IDs of shape (B, T)
            return_debug: If True, return debug info from all layers
            
        Returns:
            logits: Shape (B, T, V)
            debug_list (optional): List of debug dicts per layer
        """
        x = self.embed(input_ids)
        
        if self.embed_norm is not None:
            x = self.embed_norm(x)
        
        debug_list = [] if return_debug else None
        
        if self.layer_norms is not None:
            # pre_layer mode: x_out = x + attn(norm(x))
            # Since attn layers compute inp + scale*O(z), the delta is layer(inp) - inp.
            for norm, layer in zip(self.layer_norms, self.layers):
                x_normed = norm(x)
                if return_debug:
                    out, debug = layer(x_normed, return_debug=True)
                    debug_list.append(debug)
                else:
                    out = layer(x_normed)
                x = x + (out - x_normed)  # extract delta, add to original x
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
    def from_config(cls, cfg: dict) -> "AttentionLM":
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
            use_rmsnorm_qk=model_cfg.get("use_rmsnorm_qk", False),
            use_bias_qkv=model_cfg.get("use_bias_qkv", True),
            use_bias_o=model_cfg.get("use_bias_o", True),
            std_embed=init_cfg.get("std_embed", 0.02),
            std_qkv=init_cfg.get("std_qkv", 0.02),
            std_o=init_cfg.get("std_o", 0.01),
            attn_type=model_cfg.get("attn_type", "quadratic"),
            norm_type=model_cfg.get("norm_type", "rmsnorm"),
            norm_place=model_cfg.get("norm_place", "pre_unembed"),
        )
