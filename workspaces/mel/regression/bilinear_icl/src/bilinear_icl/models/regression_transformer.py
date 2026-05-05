import math

import torch
from torch import nn

from ._kernels import ATTN_REGISTRY
from .bilinear_mlp import BilinearMLP
from .norm import NORM_TYPES, make_norm


VALID_NORM_PLACES = ("pre_attn", "pre_mlp", "pre_unembed")


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_ctx: int,
        d_mlp: int,
        attn_scale: float,
        mlp_scale: float,
        attn_type: str,
    ):
        super().__init__()
        attn_cls = ATTN_REGISTRY[attn_type]
        self.attn = attn_cls(
            d_model=d_model,
            n_head=n_head,
            n_ctx=n_ctx,
            scale=attn_scale,
            use_bias_qk=False,
            rope_base=10000,
        )
        self.mlp = BilinearMLP(d_model, d_mlp, scale=mlp_scale)

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        if return_debug:
            x, dbg = self.attn(x, return_debug=True)
        else:
            x = self.attn(x)
            dbg = None
        x = self.mlp(x)
        return (x, dbg) if return_debug else x


class RegressionTransformer(nn.Module):
    def __init__(
        self,
        *,
        D: int,
        K: int,
        d_model: int,
        n_head: int,
        n_layers: int,
        d_mlp: int,
        attn_scale: float = 0.35,
        mlp_scale: float = 0.35,
        attn_type: str = "bilinear",
        norm_type: str = "tok0",
        norm_places: list[str] | None = None,
        bos_norm_eps: float = 1e-6,
        init_type: str = "normal",
        std_embed: float = 0.02,
        std_qkv: float = 0.02,
        std_o: float = 0.01,
        std_mlp_in: float = 0.02,
        std_mlp_out: float = 0.01,
    ):
        super().__init__()
        assert attn_type in ATTN_REGISTRY, f"attn_type must be one of {tuple(ATTN_REGISTRY.keys())}, got {attn_type!r}"
        assert norm_type in NORM_TYPES, f"norm_type must be one of {NORM_TYPES}, got {norm_type!r}"

        if norm_places is None:
            norm_places = ["pre_unembed"]
        for place in norm_places:
            assert place in VALID_NORM_PLACES, f"norm place must be one of {VALID_NORM_PLACES}, got {place!r}"

        self.D = D
        self.K = K
        self.n_ctx = 1 + 2 * K
        self.d_model = d_model
        self.attn_type = attn_type
        self.register_buffer("x_idx", torch.arange(1, 2 * K + 1, 2, dtype=torch.long), persistent=False)

        self.W_E = nn.Linear(D + 1, d_model, bias=False)
        self.bos = nn.Parameter(torch.zeros(d_model))
        self.layers = nn.ModuleList(
            [
                Block(
                    d_model=d_model,
                    n_head=n_head,
                    n_ctx=self.n_ctx,
                    d_mlp=d_mlp,
                    attn_scale=attn_scale,
                    mlp_scale=mlp_scale,
                    attn_type=attn_type,
                )
                for _ in range(n_layers)
            ]
        )

        self.pre_unembed_norm = (
            make_norm(norm_type, d_model, eps=bos_norm_eps) if "pre_unembed" in norm_places else nn.Identity()
        )
        self.pre_attn_norms = (
            nn.ModuleList([make_norm(norm_type, d_model, eps=bos_norm_eps) for _ in range(n_layers)])
            if "pre_attn" in norm_places
            else None
        )
        self.pre_mlp_norms = (
            nn.ModuleList([make_norm(norm_type, d_model, eps=bos_norm_eps) for _ in range(n_layers)])
            if "pre_mlp" in norm_places
            else None
        )

        self.w_out = nn.Linear(d_model, 1, bias=True)
        self._init_weights(
            init_type=init_type,
            std_embed=std_embed,
            std_qkv=std_qkv,
            std_o=std_o,
            std_mlp_in=std_mlp_in,
            std_mlp_out=std_mlp_out,
        )

    def _init_weights(
        self,
        *,
        init_type: str,
        std_embed: float,
        std_qkv: float,
        std_o: float,
        std_mlp_in: float,
        std_mlp_out: float,
    ):
        """Initialize weights for either legacy-normal init or muP init.

        For bilinear MLP under muP, we extend the attention-only muP scheme by using
        input-side scales ~1/sqrt(d_model) for `mlp.l` and `mlp.r`, and output-side
        scale ~1/sqrt(d_mlp*n_layers) for `mlp.d`, analogous to V/O scaling that keeps
        residual-stream variance stable with depth.
        """
        assert init_type in ("normal", "mup"), f"init_type must be 'normal' or 'mup', got {init_type!r}"

        nn.init.normal_(self.W_E.weight, std=std_embed)
        nn.init.normal_(self.bos, std=std_embed)
        nn.init.normal_(self.w_out.weight, std=std_embed)
        nn.init.zeros_(self.w_out.bias)

        n_layers = len(self.layers)
        d_model = self.d_model
        d_head = d_model // self.layers[0].attn.n_head
        d_mlp = self.layers[0].mlp.l.out_features

        if init_type == "mup":
            s_qk = 1.0 / math.sqrt(d_head)
            s_v = 1.0 / math.sqrt(d_model)
            s_o = 1.0 / math.sqrt(d_model * n_layers)
            s_lr = 1.0 / math.sqrt(d_model)
            s_d = 1.0 / math.sqrt(d_mlp * n_layers)
        else:
            s_qk, s_v, s_o = std_qkv, std_qkv, std_o
            s_lr, s_d = std_mlp_in, std_mlp_out

        for blk in self.layers:
            if hasattr(blk.attn, "q1"):
                qk_projs = [blk.attn.q1, blk.attn.k1, blk.attn.q2, blk.attn.k2]
            else:
                qk_projs = [blk.attn.q, blk.attn.k]

            for proj in qk_projs:
                nn.init.normal_(proj.weight, std=s_qk)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)

            nn.init.normal_(blk.attn.v.weight, std=s_v)
            nn.init.normal_(blk.attn.o.weight, std=s_o)
            nn.init.normal_(blk.mlp.l.weight, std=s_lr)
            nn.init.normal_(blk.mlp.r.weight, std=s_lr)
            nn.init.normal_(blk.mlp.d.weight, std=s_d)

    def embed(self, raw: torch.Tensor) -> torch.Tensor:
        h = self.W_E(raw)
        bos = h.new_zeros(1, h.size(1), h.size(-1))
        bos[0, 0] = self.bos.to(h.dtype)
        return h + bos

    def _apply_attn(self, h: torch.Tensor, blk: Block, norm: nn.Module, return_debug: bool):
        z = norm(h)
        if return_debug:
            o_out, dbg = blk.attn.branch(z, return_debug=True)
        else:
            o_out = blk.attn.branch(z)
            dbg = None

        if self.attn_type == "softmax":
            mixed = z + blk.attn.scale * o_out
        else:
            mixed = torch.lerp(z, o_out, blk.attn.scale)
        return h + (mixed - z), dbg

    def _apply_mlp(self, h: torch.Tensor, blk: Block, norm: nn.Module):
        z = norm(h)
        mixed = torch.lerp(z, blk.mlp.branch(z), blk.mlp.scale)
        return h + (mixed - z)

    def forward(self, raw: torch.Tensor, return_debug: bool = False):
        h = self.embed(raw)
        debug = []
        for i, blk in enumerate(self.layers):
            attn_norm = self.pre_attn_norms[i] if self.pre_attn_norms is not None else nn.Identity()
            h, d = self._apply_attn(h, blk, attn_norm, return_debug)
            if return_debug:
                debug.append(d)

            mlp_norm = self.pre_mlp_norms[i] if self.pre_mlp_norms is not None else nn.Identity()
            h = self._apply_mlp(h, blk, mlp_norm)

        h = self.pre_unembed_norm(h)
        y_hat_full = self.w_out(h).squeeze(-1)
        y_hat_x = y_hat_full[:, self.x_idx]
        if return_debug:
            return y_hat_x, {"hidden_pre_readout": h, "y_hat_full": y_hat_full, "layers": debug}
        return y_hat_x
