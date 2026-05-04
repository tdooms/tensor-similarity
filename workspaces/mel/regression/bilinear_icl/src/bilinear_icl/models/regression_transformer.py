import torch
from torch import nn

from ._kernels.attention_kernels.bilinear import BilinearAttention
from .bilinear_mlp import BilinearMLP
from .norm import BOSScalarNorm


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, n_ctx: int, d_mlp: int, attn_scale: float, mlp_scale: float):
        super().__init__()
        self.attn = BilinearAttention(
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
        bos_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.D = D
        self.K = K
        self.n_ctx = 1 + 2 * K
        self.d_model = d_model
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
                )
                for _ in range(n_layers)
            ]
        )
        self.bos_norm = BOSScalarNorm(eps=bos_norm_eps)
        self.w_out = nn.Linear(d_model, 1, bias=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.W_E.weight, std=0.02)
        nn.init.normal_(self.bos, std=0.02)
        for blk in self.layers:
            for proj in (blk.attn.q1, blk.attn.k1, blk.attn.q2, blk.attn.k2):
                nn.init.normal_(proj.weight, std=0.02)
            nn.init.normal_(blk.attn.v.weight, std=0.02)
            nn.init.normal_(blk.attn.o.weight, std=0.01)
            nn.init.normal_(blk.mlp.l.weight, std=0.02)
            nn.init.normal_(blk.mlp.r.weight, std=0.02)
            nn.init.normal_(blk.mlp.d.weight, std=0.01)
        nn.init.normal_(self.w_out.weight, std=0.02)
        nn.init.zeros_(self.w_out.bias)

    def embed(self, raw: torch.Tensor) -> torch.Tensor:
        h = self.W_E(raw)
        bos = h.new_zeros(1, h.size(1), h.size(-1))
        bos[0, 0] = self.bos.to(h.dtype)
        return h + bos

    def forward(self, raw: torch.Tensor, return_debug: bool = False):
        h = self.embed(raw)
        debug = []
        for blk in self.layers:
            if return_debug:
                h, d = blk(h, return_debug=True)
                debug.append(d)
            else:
                h = blk(h)

        h = self.bos_norm(h)
        y_hat_full = self.w_out(h).squeeze(-1)
        y_hat_x = y_hat_full[:, self.x_idx]
        if return_debug:
            return y_hat_x, {"hidden_pre_readout": h, "y_hat_full": y_hat_full, "layers": debug}
        return y_hat_x
