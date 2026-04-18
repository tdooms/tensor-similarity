import math
import torch
import torch.nn as nn
from .attention_kernels.bilinear import BilinearAttention, QuadraticAttention
from .attention_kernels.softmax import SoftmaxAttention


class MaxRMSNorm(nn.Module):
    """Normalize by the largest per-token RMS in the sequence.

    For input x of shape (B, T, D):
        e_t   = mean_d(x_{t,d}^2)          # (B, T, 1)
        m     = max_t(e_t)                  # (B, 1, 1)
        scale = 1 / sqrt(m + eps)           # (B, 1, 1)
        out   = scale * x                    # (B, T, D)
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.normalized_shape, (
            f"expected last dim {self.normalized_shape}, got {x.shape[-1]}"
        )
        energy = x.pow(2).mean(dim=-1, keepdim=True)        # (B, T, 1)
        max_energy = energy.max(dim=-2, keepdim=True).values # (B, 1, 1)
        scale = (max_energy + self.eps).rsqrt()              # (B, 1, 1)
        return x * scale


class CausalMaxRMSNorm(nn.Module):
    """Causal variant of MaxRMSNorm: normalize by the largest per-token RMS *up to* the current position.

    For input x of shape (B, T, D):
        e_t       = mean_d(x_{t,d}^2)              # (B, T, 1)
        m_t       = cummax_{s<=t}(e_s)              # (B, T, 1)
        scale_t   = 1 / sqrt(m_t + eps)             # (B, T, 1)
        out_t     = scale_t * x_t                   # (B, T, D)

    This mirrors inference behaviour where future tokens are unavailable.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.normalized_shape, (
            f"expected last dim {self.normalized_shape}, got {x.shape[-1]}"
        )
        energy = x.pow(2).mean(dim=-1, keepdim=True)              # (B, T, 1)
        causal_max_energy = energy.cummax(dim=-2).values           # (B, T, 1)
        scale = (causal_max_energy + self.eps).rsqrt()             # (B, T, 1)
        return x * scale


class Tok0Batch(nn.Module):
    """Normalize by RMS of token t=0 averaged across the batch.

    Supports dual eval modes:
        - use_running_stats=False: use batch stats at eval (eval-as-train)
        - use_running_stats=True: use running stats at eval (eval-as-inference)
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        momentum: float = 0.1,
        use_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.momentum = momentum
        self.use_running_stats = use_running_stats

        self.register_buffer("running_mean_energy", torch.ones(1))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.normalized_shape, (
            f"expected last dim {self.normalized_shape}, got {x.shape[-1]}"
        )
        energy_t0 = x[:, 0, :].pow(2).mean(dim=-1)  # (B,)
        m_batch = energy_t0.mean()  # scalar

        if self.training:
            scale = (m_batch + self.eps).rsqrt()
            self.num_batches_tracked += 1
            self.running_mean_energy.mul_(1 - self.momentum).add_(
                m_batch.detach() * self.momentum
            )
        else:
            if self.use_running_stats and self.num_batches_tracked > 0:
                scale = (self.running_mean_energy + self.eps).rsqrt()
            else:
                scale = (m_batch + self.eps).rsqrt()

        return x * scale


class Tok0(nn.Module):
    """Normalize by RMS of token t=0 for each sequence independently."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.normalized_shape, (
            f"expected last dim {self.normalized_shape}, got {x.shape[-1]}"
        )
        energy_t0 = x[:, 0, :].pow(2).mean(dim=-1, keepdim=True)  # (B, 1)
        scale = (energy_t0.unsqueeze(1) + self.eps).rsqrt()  # (B, 1, 1)
        return x * scale


class BatchNorm(nn.Module):
    """BatchNorm over model dimension using flattened (B*T, D) samples."""

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.bn = nn.BatchNorm1d(
            normalized_shape,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.normalized_shape, (
            f"expected last dim {self.normalized_shape}, got {x.shape[-1]}"
        )
        b, t, d = x.shape
        x_flat = x.reshape(b * t, d)
        x_norm = self.bn(x_flat)
        return x_norm.reshape(b, t, d)


ATTN_REGISTRY = {
    "bilinear": BilinearAttention,
    "quadratic": QuadraticAttention,
    "softmax": SoftmaxAttention,
}


NORM_TYPES = (
    "none",
    "rmsnorm",
    "layernorm",
    "maxrmsnorm",
    "causal_maxrmsnorm",
    "tok0",
    "tok0_batch",
    "batchnorm",
)
VALID_NORM_PLACES = ("post_embed", "pre_layer", "pre_unembed")


class AttentionLM(nn.Module):
    """Autoregressive language model with configurable attention.
    
    Architecture:
        - Token embedding E
        - n_layers x Attention blocks (attention + residual only, no MLP)
        - Separate unembedding U (not tied)
    
    Supports attn_type: 'quadratic' (default) or 'softmax'.
    
    norm_type selects the normalisation function:
        'none'       – no normalization
        'rmsnorm'    – RMSNorm (default)
        'layernorm'  – LayerNorm
        'maxrmsnorm' – scale by the largest per-token RMS in the sequence
        'tok0'       – scale by token-0 energy per sample
        'tok0_batch' – scale by mean token-0 energy over the batch
        'batchnorm'  – BatchNorm1d over flattened (B*T, D)
    
    norm_places controls where normalisation is applied (list):
        []                          – no normalization anywhere
        ['post_embed']              – norm after embedding only
        ['pre_layer']               – norm before each attention layer + before unembed
        ['pre_unembed']             – norm only before unembed
        ['post_embed', 'pre_unembed'] – both post-embed and pre-unembed
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
        self.norm_kwargs = norm_kwargs or {}
        self.init_type = init_type
        
        attn_cls = ATTN_REGISTRY[attn_type]
        
        def _make_norm():
            if norm_type == "rmsnorm":
                return nn.RMSNorm(d_model)
            elif norm_type == "layernorm":
                return nn.LayerNorm(d_model)
            elif norm_type == "maxrmsnorm":
                return MaxRMSNorm(d_model)
            elif norm_type == "causal_maxrmsnorm":
                return CausalMaxRMSNorm(d_model)
            elif norm_type == "tok0":
                return Tok0(d_model, **self.norm_kwargs)
            elif norm_type == "tok0_batch":
                return Tok0Batch(d_model, **self.norm_kwargs)
            elif norm_type == "batchnorm":
                return BatchNorm(d_model, **self.norm_kwargs)
            else:
                return nn.Identity()
        
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # Post-embedding norm – only used when 'post_embed' in norm_places
        if "post_embed" in norm_places:
            self.embed_norm = _make_norm()
        else:
            self.embed_norm = None
        
        # Final norm (before unembed) – used when 'pre_unembed' or 'pre_layer' in norm_places
        if "pre_unembed" in norm_places or "pre_layer" in norm_places:
            self.final_norm = _make_norm()
        else:
            self.final_norm = nn.Identity()
        
        # Per-layer pre-norms – only used when 'pre_layer' in norm_places
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
                use_rmsnorm_qk=use_rmsnorm_qk,
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
        """Initialize weights.
        
        init_type:
            'normal'     – Gaussian init with per-group std (default, original behaviour)
            'orthogonal' – Orthogonal init (nn.init.orthogonal_) with configurable gain
            'mup'        – muP-style: QK std=1/√d_head, V std=1/√d_model, O std=1/(√d_model·√n_layers)
        """
        assert init_type in ("normal", "orthogonal", "mup"), (
            f"init_type must be 'normal', 'orthogonal', or 'mup', got {init_type!r}"
        )
        
        if init_type == "normal":
            nn.init.normal_(self.embed.weight, mean=0.0, std=std_embed)
            nn.init.normal_(self.unembed.weight, mean=0.0, std=std_embed)
            
            for layer in self.layers:
            # BilinearAttention has q1,k1,q2,k2; others have q,k
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
            use_bias_qk=model_cfg.get("use_bias_qk", True),
            std_embed=init_cfg.get("std_embed", 0.02),
            std_qkv=init_cfg.get("std_qkv", 0.02),
            std_o=init_cfg.get("std_o", 0.01),
            attn_type=model_cfg.get("attn_type", "quadratic"),
            norm_type=model_cfg.get("norm_type", "rmsnorm"),
            norm_places=model_cfg.get("norm_places", []),
            norm_kwargs=model_cfg.get("norm_kwargs", {}),
            init_type=init_cfg.get("init_type", "normal"),
            init_gain=init_cfg.get("init_gain", 1.0),
        )
