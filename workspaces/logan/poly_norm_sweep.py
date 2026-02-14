#!/usr/bin/env python3
"""Sweep of polynomial-compatible stabilization techniques on toy induction task.

Tests each technique on bilinear attention (no QK RMSNorm) for 10k steps.
Reports which techniques enable induction learning.
"""
import math, json, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from lib import AttentionLM
from lib.attention import BilinearAttention
from einops import rearrange, einsum
from tqdm import tqdm
from pathlib import Path
from datetime import datetime


class InductionDataset(Dataset):
    def __init__(self, vocab_size, half_len, size):
        self.vocab_size, self.half_len, self.size = vocab_size, half_len, size
    def __len__(self): return self.size
    def __getitem__(self, idx):
        seq = torch.randint(1, self.vocab_size, (self.half_len,))
        return torch.cat([seq, seq])


# ============================================================
# Technique implementations
# ============================================================

# --- Cubic attention: single QK pair, (scores/sqrt(d))^3 / sqrt(T) ---
class CubicAttention(BilinearAttention):
    def forward(self, x, return_debug=False):
        B, T, _ = x.shape
        q = rearrange(self.q(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        k = rearrange(self.k(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        v = rearrange(self.v(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        q, k = self.rotary(q), self.rotary(k)
        scores = einsum(q, k, "b sq nh dh, b sk nh dh -> b nh sq sk") / math.sqrt(self.d_head)
        pattern = (scores ** 3) / math.sqrt(T)
        pattern = pattern * self.causal_mask[None, None, :T, :T]
        z = einsum(pattern, v, "b nh sq sk, b sk nh dh -> b sq nh dh")
        out = x + self.scale * self.o(rearrange(z, "b t nh dh -> b t (nh dh)"))
        if return_debug: return out, {"pattern": pattern}
        return out


# --- Trilinear: 3 QK pairs, (q1k1)(q2k2)(q3k3)/d^3 ---
class TrilinearAttention(BilinearAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q3 = nn.Linear(self.d_model, self.d_model, bias=True)
        self.k3 = nn.Linear(self.d_model, self.d_model, bias=True)

    def forward(self, x, return_debug=False):
        B, T, _ = x.shape
        q1 = rearrange(self.q(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        k1 = rearrange(self.k(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        q2 = rearrange(self.q2(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        k2 = rearrange(self.k2(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        q3 = rearrange(self.q3(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        k3 = rearrange(self.k3(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        v = rearrange(self.v(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        q1, k1 = self.rotary(q1), self.rotary(k1)
        q2, k2 = self.rotary(q2), self.rotary(k2)
        q3, k3 = self.rotary(q3), self.rotary(k3)
        s1 = einsum(q1, k1, "b sq nh dh, b sk nh dh -> b nh sq sk")
        s2 = einsum(q2, k2, "b sq nh dh, b sk nh dh -> b nh sq sk")
        s3 = einsum(q3, k3, "b sq nh dh, b sk nh dh -> b nh sq sk")
        pattern = (s1 * s2 * s3) / (self.d_head ** 3)
        pattern = pattern * self.causal_mask[None, None, :T, :T]
        z = einsum(pattern, v, "b nh sq sk, b sk nh dh -> b sq nh dh")
        out = x + self.scale * self.o(rearrange(z, "b t nh dh -> b t (nh dh)"))
        if return_debug: return out, {"pattern": pattern}
        return out


# --- LayerScale: learned per-channel residual scaling (init 1e-4) ---
class BilinearLayerScale(BilinearAttention):
    def __init__(self, *args, init_scale=1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = nn.Parameter(torch.ones(self.d_model) * init_scale)

    def forward(self, x, return_debug=False):
        B, T, _ = x.shape
        q1 = rearrange(self.q(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        k1 = rearrange(self.k(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        q2 = rearrange(self.q2(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        k2 = rearrange(self.k2(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        v = rearrange(self.v(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        q1, k1 = self.rotary(q1), self.rotary(k1)
        q2, k2 = self.rotary(q2), self.rotary(k2)
        s1 = einsum(q1, k1, "b sq nh dh, b sk nh dh -> b nh sq sk")
        s2 = einsum(q2, k2, "b sq nh dh, b sk nh dh -> b nh sq sk")
        pattern = (s1 * s2) / (self.d_head ** 2)
        pattern = pattern * self.causal_mask[None, None, :T, :T]
        z = einsum(pattern, v, "b nh sq sk, b sk nh dh -> b sq nh dh")
        z_merge = rearrange(z, "b t nh dh -> b t (nh dh)")
        out = x + self.gamma * self.o(z_merge)
        if return_debug: return out, {"pattern": pattern}
        return out


# --- Weight Standardization on Q, K ---
class WSLinear(nn.Linear):
    def forward(self, x):
        w = self.weight
        w = (w - w.mean(dim=1, keepdim=True)) / w.std(dim=1, keepdim=True).clamp(min=1e-5)
        return F.linear(x, w, self.bias)

class BilinearWeightStd(BilinearAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace Q, K with weight-standardized versions
        for attr in ['q', 'k', 'q2', 'k2']:
            old = getattr(self, attr)
            new = WSLinear(old.in_features, old.out_features, bias=old.bias is not None)
            setattr(self, attr, new)


# --- BatchNorm on Q, K (from previous experiment) ---
class BilinearBatchNorm(BilinearAttention):
    def __init__(self, *args, **kwargs):
        kwargs["use_rmsnorm_qk"] = False
        super().__init__(*args, **kwargs)
        self.bn_q1 = nn.BatchNorm1d(self.d_model)
        self.bn_k1 = nn.BatchNorm1d(self.d_model)
        self.bn_q2 = nn.BatchNorm1d(self.d_model)
        self.bn_k2 = nn.BatchNorm1d(self.d_model)

    def forward(self, x, return_debug=False):
        B, T, _ = x.shape
        q1 = self.bn_q1(self.q(x).reshape(B*T, -1)).reshape(B, T, -1)
        k1 = self.bn_k1(self.k(x).reshape(B*T, -1)).reshape(B, T, -1)
        q2 = self.bn_q2(self.q2(x).reshape(B*T, -1)).reshape(B, T, -1)
        k2 = self.bn_k2(self.k2(x).reshape(B*T, -1)).reshape(B, T, -1)
        q1 = rearrange(q1, "b t (nh dh) -> b t nh dh", nh=self.n_head)
        k1 = rearrange(k1, "b t (nh dh) -> b t nh dh", nh=self.n_head)
        q2 = rearrange(q2, "b t (nh dh) -> b t nh dh", nh=self.n_head)
        k2 = rearrange(k2, "b t (nh dh) -> b t nh dh", nh=self.n_head)
        v = rearrange(self.v(x), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        q1, k1 = self.rotary(q1), self.rotary(k1)
        q2, k2 = self.rotary(q2), self.rotary(k2)
        s1 = einsum(q1, k1, "b sq nh dh, b sk nh dh -> b nh sq sk")
        s2 = einsum(q2, k2, "b sq nh dh, b sk nh dh -> b nh sq sk")
        pattern = (s1 * s2) / (self.d_head ** 2)
        pattern = pattern * self.causal_mask[None, None, :T, :T]
        z = einsum(pattern, v, "b nh sq sk, b sk nh dh -> b sq nh dh")
        out = x + self.scale * self.o(rearrange(z, "b t nh dh -> b t (nh dh)"))
        if return_debug: return out, {"pattern": pattern}
        return out


# --- Hadamard mixing before Q,K projections ---
def _hadamard(n):
    """Sylvester construction of n x n Hadamard matrix (n must be power of 2)."""
    if n == 1: return torch.tensor([[1.0]])
    h = _hadamard(n // 2)
    return torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)

class BilinearHadamard(BilinearAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        H = _hadamard(self.d_model) / math.sqrt(self.d_model)
        self.register_buffer("H", H)

    def forward(self, x, return_debug=False):
        B, T, _ = x.shape
        x_h = F.linear(x, self.H)  # Fixed Hadamard mixing
        q1 = rearrange(self.q(x_h), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        k1 = rearrange(self.k(x_h), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        q2 = rearrange(self.q2(x_h), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        k2 = rearrange(self.k2(x_h), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        v = rearrange(self.v(x_h), "b t (nh dh) -> b t nh dh", nh=self.n_head)
        q1, k1 = self.rotary(q1), self.rotary(k1)
        q2, k2 = self.rotary(q2), self.rotary(k2)
        s1 = einsum(q1, k1, "b sq nh dh, b sk nh dh -> b nh sq sk")
        s2 = einsum(q2, k2, "b sq nh dh, b sk nh dh -> b nh sq sk")
        pattern = (s1 * s2) / (self.d_head ** 2)
        pattern = pattern * self.causal_mask[None, None, :T, :T]
        z = einsum(pattern, v, "b nh sq sk, b sk nh dh -> b sq nh dh")
        out = x + self.scale * self.o(rearrange(z, "b t nh dh -> b t (nh dh)"))
        if return_debug: return out, {"pattern": pattern}
        return out


# ============================================================
# Build functions
# ============================================================

def _base_model(vocab_size, n_ctx, d_model, n_head, n_layers):
    return AttentionLM(
        vocab_size=vocab_size, n_ctx=n_ctx, d_model=d_model,
        n_head=n_head, n_layers=n_layers, attn_scale=1.0,
        attn_type="bilinear", norm_type="layernorm", norm_place="pre_unembed",
        use_rmsnorm_qk=False,
    )

def _replace_layers(model, cls, d_model, n_head, n_ctx, **kw):
    for i in range(len(model.layers)):
        model.layers[i] = cls(d_model=d_model, n_head=n_head, n_ctx=n_ctx,
                              scale=1.0, use_bias_qkv=True, use_bias_o=True, **kw)
    _reinit(model)

def _reinit(model):
    for layer in model.layers:
        for name in ['q', 'k', 'q2', 'k2', 'q3', 'k3', 'v']:
            if hasattr(layer, name):
                nn.init.normal_(getattr(layer, name).weight, std=0.02)
                if getattr(layer, name).bias is not None:
                    nn.init.zeros_(getattr(layer, name).bias)
        if hasattr(layer, 'o'):
            nn.init.normal_(layer.o.weight, std=0.01)
            if layer.o.bias is not None:
                nn.init.zeros_(layer.o.bias)


# Registry: name -> (build_fn, description)
# build_fn(vocab_size, n_ctx, d_model, n_head, n_layers, device) -> (model, extra_loss_fn_or_None)
TECHNIQUES = {}

def register(name, desc):
    def dec(fn):
        TECHNIQUES[name] = {"build": fn, "desc": desc}
        return fn
    return dec

@register("baseline", "No normalization (bilinear, no QK norm)")
def _build(vs, nc, dm, nh, nl, dev):
    return _base_model(vs, nc, dm, nh, nl).to(dev), None

@register("batchnorm", "BatchNorm on Q,K projections")
def _build(vs, nc, dm, nh, nl, dev):
    m = _base_model(vs, nc, dm, nh, nl)
    _replace_layers(m, BilinearBatchNorm, dm, nh, nc)
    return m.to(dev), None

@register("specnorm", "Spectral norm on Q,K weight matrices")
def _build(vs, nc, dm, nh, nl, dev):
    m = _base_model(vs, nc, dm, nh, nl).to(dev)
    for layer in m.layers:
        for attr in ['q', 'k', 'q2', 'k2']:
            if hasattr(layer, attr):
                torch.nn.utils.parametrizations.spectral_norm(getattr(layer, attr))
    return m, None

@register("cubic_single", "Cubic (QK^T/sqrt(d))^3/sqrt(T), single QK")
def _build(vs, nc, dm, nh, nl, dev):
    m = _base_model(vs, nc, dm, nh, nl)
    _replace_layers(m, CubicAttention, dm, nh, nc)
    return m.to(dev), None

@register("trilinear", "Trilinear: 3 QK pairs multiplied")
def _build(vs, nc, dm, nh, nl, dev):
    m = _base_model(vs, nc, dm, nh, nl)
    _replace_layers(m, TrilinearAttention, dm, nh, nc)
    return m.to(dev), None

@register("layerscale", "LayerScale (init 1e-4) on bilinear residual")
def _build(vs, nc, dm, nh, nl, dev):
    m = _base_model(vs, nc, dm, nh, nl)
    _replace_layers(m, BilinearLayerScale, dm, nh, nc, init_scale=1e-4)
    return m.to(dev), None

@register("weight_std", "Weight Standardization on Q,K projections")
def _build(vs, nc, dm, nh, nl, dev):
    m = _base_model(vs, nc, dm, nh, nl)
    _replace_layers(m, BilinearWeightStd, dm, nh, nc)
    return m.to(dev), None

@register("ortho_reg", "Orthogonal regularization on Q,K (lambda=0.1)")
def _build(vs, nc, dm, nh, nl, dev):
    m = _base_model(vs, nc, dm, nh, nl).to(dev)
    def extra_loss(model):
        loss = 0.0
        for layer in model.layers:
            for attr in ['q', 'k', 'q2', 'k2']:
                if hasattr(layer, attr):
                    W = getattr(layer, attr).weight
                    WtW = W.T @ W
                    I = torch.eye(W.shape[1], device=W.device)
                    loss = loss + torch.norm(WtW - I) ** 2
        return 0.1 * loss
    return m, extra_loss

@register("fixup", "Fixup: zero-init O, depth-scaled Q,K")
def _build(vs, nc, dm, nh, nl, dev):
    m = _base_model(vs, nc, dm, nh, nl)
    scale = nl ** (-0.25)
    for layer in m.layers:
        for attr in ['q', 'k', 'q2', 'k2', 'v']:
            nn.init.normal_(getattr(layer, attr).weight, std=0.02 * scale)
        nn.init.zeros_(layer.o.weight)
        if layer.o.bias is not None:
            nn.init.zeros_(layer.o.bias)
    return m.to(dev), None

@register("muP_init", "muP-style init: Q,K~1/sqrt(d_head), O~1/(sqrt(d)*sqrt(L))")
def _build(vs, nc, dm, nh, nl, dev):
    m = _base_model(vs, nc, dm, nh, nl)
    dh = dm // nh
    for layer in m.layers:
        for attr in ['q', 'k', 'q2', 'k2']:
            nn.init.normal_(getattr(layer, attr).weight, std=1.0/math.sqrt(dh))
        nn.init.normal_(layer.v.weight, std=1.0/math.sqrt(dm))
        nn.init.normal_(layer.o.weight, std=1.0/(math.sqrt(dm) * math.sqrt(nl)))
    return m.to(dev), None

@register("strong_wd", "Strong weight decay (0.1 instead of 0.01)")
def _build(vs, nc, dm, nh, nl, dev):
    return _base_model(vs, nc, dm, nh, nl).to(dev), None

@register("hadamard", "Hadamard mixing before Q,K projections")
def _build(vs, nc, dm, nh, nl, dev):
    m = _base_model(vs, nc, dm, nh, nl)
    _replace_layers(m, BilinearHadamard, dm, nh, nc)
    return m.to(dev), None

@register("ortho_init", "Orthogonal initialization for all weights")
def _build(vs, nc, dm, nh, nl, dev):
    m = _base_model(vs, nc, dm, nh, nl)
    for layer in m.layers:
        for attr in ['q', 'k', 'q2', 'k2', 'v', 'o']:
            if hasattr(layer, attr):
                nn.init.orthogonal_(getattr(layer, attr).weight)
    return m.to(dev), None

@register("layerscale_ortho", "LayerScale + orthogonal regularization")
def _build(vs, nc, dm, nh, nl, dev):
    m = _base_model(vs, nc, dm, nh, nl)
    _replace_layers(m, BilinearLayerScale, dm, nh, nc, init_scale=1e-4)
    m = m.to(dev)
    def extra_loss(model):
        loss = 0.0
        for layer in model.layers:
            for attr in ['q', 'k', 'q2', 'k2']:
                if hasattr(layer, attr):
                    W = getattr(layer, attr).weight
                    WtW = W.T @ W
                    I = torch.eye(W.shape[1], device=W.device)
                    loss = loss + torch.norm(WtW - I) ** 2
        return 0.1 * loss
    return m, extra_loss


# ============================================================
# Training loop
# ============================================================

def train_technique(name, build_fn, vocab_size=32, half_len=16, n_head=2,
                    n_layers=2, d_model=128, batch_size=256, lr=1e-3,
                    max_steps=10000, eval_every=500, device="cuda"):
    n_ctx = 2 * half_len
    model, extra_loss_fn = build_fn(vocab_size, n_ctx, d_model, n_head, n_layers, device)
    n_params = sum(p.numel() for p in model.parameters())

    wd = 0.1 if name == "strong_wd" else 0.01
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=lr*0.1)

    train_ds = InductionDataset(vocab_size, half_len, size=100_000)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    eval_ds = InductionDataset(vocab_size, half_len, size=1000)
    eval_seqs = torch.stack([eval_ds[i] for i in range(len(eval_ds))]).to(device)

    metrics = []

    @torch.no_grad()
    def evaluate(step):
        model.eval()
        logits = model(eval_seqs)
        sl = logits[:, half_len-1:2*half_len-1, :]
        st = eval_seqs[:, half_len:]
        B, T, V = sl.shape
        loss = F.cross_entropy(sl.reshape(B*T, V), st.reshape(B*T)).item()
        acc = (sl.argmax(-1) == st).float().mean().item()
        model.train()
        return loss, acc

    init_loss, init_acc = evaluate(0)
    print(f"  [{name}] step 0: loss={init_loss:.4f} acc={init_acc:.4f} params={n_params:,}")
    metrics.append({"step": 0, "loss": init_loss, "acc": init_acc})

    step = 0
    pbar = tqdm(total=max_steps, desc=f"[{name}]", leave=False)
    best_acc = init_acc
    try:
        while step < max_steps:
            for batch in train_dl:
                if step >= max_steps: break
                batch = batch.to(device)
                optimizer.zero_grad()
                logits = model(batch)
                sl = logits[:, half_len-1:2*half_len-1, :]
                st = batch[:, half_len:]
                B, T, V = sl.shape
                loss = F.cross_entropy(sl.reshape(B*T, V), st.reshape(B*T))
                if extra_loss_fn:
                    loss = loss + extra_loss_fn(model)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                step += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                pbar.update(1)
                if step % eval_every == 0:
                    el, ea = evaluate(step)
                    metrics.append({"step": step, "loss": el, "acc": ea})
                    best_acc = max(best_acc, ea)
                    pbar.write(f"  [{name}] step {step}: loss={el:.4f} acc={ea:.4f}")
                    # Early stop if perfect
                    if ea > 0.99:
                        pbar.write(f"  [{name}] Perfect accuracy reached, stopping early.")
                        step = max_steps
                        break
    except Exception as e:
        print(f"  [{name}] ERROR: {e}")
        pbar.close()
        return {"technique": name, "final_acc": 0, "final_loss": 99, "error": str(e),
                "metrics": metrics, "n_params": n_params}

    pbar.close()
    final_loss, final_acc = evaluate(step)
    metrics.append({"step": step, "loss": final_loss, "acc": final_acc})
    print(f"  [{name}] FINAL: loss={final_loss:.4f} acc={final_acc:.4f}")
    return {"technique": name, "final_acc": final_acc, "final_loss": final_loss,
            "best_acc": best_acc, "metrics": metrics, "n_params": n_params}


# ============================================================
# Main
# ============================================================

def main(techniques=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    if techniques is None:
        techniques = list(TECHNIQUES.keys())

    results = {}
    for name in techniques:
        info = TECHNIQUES[name]
        print(f"\n{'='*60}")
        print(f"Testing: {name} - {info['desc']}")
        print(f"{'='*60}")
        torch.manual_seed(42)
        try:
            result = train_technique(name, info["build"], device=device)
            results[name] = result
        except Exception as e:
            print(f"  [{name}] FAILED: {e}")
            import traceback; traceback.print_exc()
            results[name] = {"technique": name, "final_acc": 0, "error": str(e)}

    # Save results
    out_dir = Path("results/toy_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary table
    print(f"\n{'='*60}")
    print("TOY INDUCTION SWEEP RESULTS")
    print(f"{'='*60}")
    print(f"{'Technique':24s} {'Acc':>8s} {'Loss':>8s} {'Status':>8s}")
    print("-" * 52)
    passed = []
    for name in techniques:
        r = results.get(name, {})
        acc = r.get("final_acc", 0)
        loss = r.get("final_loss", 99)
        status = "PASS" if acc > 0.5 else "FAIL"
        if acc > 0.5:
            passed.append(name)
        print(f"  {name:22s} {acc:8.4f} {loss:8.4f} {status:>8s}")

    print(f"\nPassed ({len(passed)}): {', '.join(passed)}")
    with open(out_dir / "passed.json", "w") as f:
        json.dump(passed, f)

    return results, passed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--technique", type=str, default=None,
                        help="Run single technique (default: all)")
    args = parser.parse_args()
    techniques = [args.technique] if args.technique else None
    main(techniques)
