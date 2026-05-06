#!/usr/bin/env python3
"""Compute empirical similarity heatmaps between layer-1 attention heads.

For each checkpoint, this decomposes the layer-1 contribution into per-head
logit contributions and computes empirical raw/local similarities between
heads across checkpoints on generated induction data.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from einops import rearrange, einsum
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[5]
for _path in (str(ROOT), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from experiments.induction_heads.data import RepeatedTokenDataset  # noqa: E402
from experiments.path_decomp.path_pair_tn_heatmaps import select_steps  # noqa: E402
from models import AttentionLM  # noqa: E402


def checkpoint_path(run_dir: Path, step: int) -> Path:
    exact = run_dir / "checkpoints" / f"step_{step}.pt"
    if exact.exists():
        return exact
    for path in sorted((run_dir / "checkpoints").glob("step_*.pt")):
        if int(path.stem.removeprefix("step_")) == step:
            return path
    raise FileNotFoundError(f"No checkpoint for step {step} in {run_dir / 'checkpoints'}")


def load_model(run_dir: Path, step: int, device: torch.device) -> tuple[AttentionLM, dict]:
    with (run_dir / "config.yaml").open() as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    if cfg.get("model", {}).get("norm_type") == "tok_0":
        cfg["model"]["norm_type"] = "tok0"
    model = AttentionLM.from_config(cfg)
    checkpoint = torch.load(checkpoint_path(run_dir, step), map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model.to(device), cfg


def split_heads(linear, x, n_head):
    return rearrange(linear(x), "b t (n h) -> b t n h", n=n_head)


def l1_head_outputs(attn, x):
    """Return per-head layer-1 active residual contributions, shape (H,B,T,D)."""
    if not hasattr(attn, "q"):
        raise ValueError("Layer-1 head decomposition currently supports QuadraticAttention.")
    n, d = attn.n_head, attn.d_head
    q = attn.rotary(attn.norm_qk(split_heads(attn.q, x, n)))
    k = attn.rotary(attn.norm_qk(split_heads(attn.k, x, n)))
    v = split_heads(attn.v, x, n)
    scores = einsum(q, k, "b tq n h, b tk n h -> b n tq tk")
    pattern = (scores / d).square()
    pattern = pattern * attn.causal_mask[None, None, : x.shape[1], : x.shape[1]]
    z = einsum(pattern, v, "b n tq tk, b tk n h -> b n tq h")
    o_w = rearrange(attn.o.weight, "d (n h) -> n d h", n=n)
    per_head = einsum(z, o_w, "b n t h, n d h -> n b t d")
    return attn.scale * per_head


def active_quadratic(attn, x_q, x_k, x_v):
    n, d = attn.n_head, attn.d_head
    q = attn.rotary(attn.norm_qk(split_heads(attn.q, x_q, n)))
    k = attn.rotary(attn.norm_qk(split_heads(attn.k, x_k, n)))
    v = split_heads(attn.v, x_v, n)
    scores = einsum(q, k, "b tq n h, b tk n h -> b n tq tk")
    pattern = (scores / d).square()
    pattern = pattern * attn.causal_mask[None, None, : x_q.shape[1], : x_q.shape[1]]
    z = einsum(pattern, v, "b n tq tk, b tk n h -> b tq n h")
    return attn.o(rearrange(z, "b t n h -> b t (n h)"))


def final_norm_scale(model: AttentionLM, total: torch.Tensor) -> torch.Tensor:
    norm_places = set(getattr(model, "norm_places", []))
    norm_type = getattr(model, "norm_type", "none")
    if "pre_unembed" not in norm_places and "pre_layer" not in norm_places:
        return torch.ones((*total.shape[:-1], 1), device=total.device, dtype=total.dtype)
    if norm_type in ("none", None):
        return torch.ones((*total.shape[:-1], 1), device=total.device, dtype=total.dtype)
    if norm_type == "tok0":
        eps = getattr(model.final_norm, "eps", 1e-6)
        energy_t0 = total[:, 0, :].pow(2).mean(dim=-1, keepdim=True)
        return (energy_t0.unsqueeze(1) + eps).rsqrt()
    if norm_type == "tok0_batch":
        eps = getattr(model.final_norm, "eps", 1e-6)
        scale = (total[:, 0, :].pow(2).mean(dim=-1).mean() + eps).rsqrt()
        return torch.ones((*total.shape[:-1], 1), device=total.device, dtype=total.dtype) * scale
    raise ValueError(f"Unsupported norm_type for empirical head decomposition: {norm_type!r}")


@torch.no_grad()
def l1_head_logits(model: AttentionLM, input_ids: torch.Tensor) -> torch.Tensor:
    """Return shape (n_head, batch, seq, vocab)."""
    if model.attn_type != "quadratic":
        raise ValueError(f"Expected quadratic attention, got {model.attn_type!r}")
    if model.n_layers != 2:
        raise ValueError(f"Expected 2 layers, got {model.n_layers}")
    if getattr(model, "embed_norm", None) is not None:
        raise ValueError("post_embed normalization is not supported.")
    if getattr(model, "layer_norms", None) is not None:
        raise ValueError("pre_layer normalization is not supported.")

    h0 = model.embed(input_ids)
    attn1, attn2 = model.layers[0], model.layers[1]
    per_head = l1_head_outputs(attn1, h0)
    l1_active = per_head.sum(dim=0)
    r0 = (1.0 - attn1.scale) * h0
    z = r0 + l1_active
    total = (1.0 - attn2.scale) * z + attn2.scale * active_quadratic(attn2, z, z, z)
    scale = final_norm_scale(model, total)
    head_pre_unembed = (1.0 - attn2.scale) * per_head * scale.unsqueeze(0)
    return torch.stack([model.unembed(head_pre_unembed[h]) for h in range(attn1.n_head)], dim=0)


def build_dataset(cfg: dict, n_samples: int, seed: int, split: str) -> RepeatedTokenDataset:
    model_cfg = cfg["model"]
    data_cfg = cfg.get("data", {})
    use_bos = data_cfg.get("use_bos", False)
    bos_token_id = data_cfg.get("bos_token_id")
    if use_bos and bos_token_id is None:
        bos_token_id = model_cfg["vocab_size"] - 1
    if not use_bos:
        bos_token_id = None
    return RepeatedTokenDataset(
        vocab_size=model_cfg["vocab_size"],
        n_ctx=model_cfg["n_ctx"],
        n_samples=n_samples,
        seed=seed if split == "train" else seed + 1,
        bos_token_id=bos_token_id,
    )


@torch.no_grad()
def checkpoint_head_logits(model, dataset, batch_size, device):
    chunks = []
    for start in tqdm(range(0, len(dataset), batch_size), desc="Empirical batches", leave=False):
        batch = dataset.data[start : start + batch_size].to(device)
        chunks.append(l1_head_logits(model, batch).detach().cpu().float())
    return torch.cat(chunks, dim=1)


def compute_sims(outputs: list[torch.Tensor]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(outputs)
    h = outputs[0].shape[0]
    raw = np.full((n, n, h, h), np.nan, dtype=np.float64)
    norms = np.full((n, h), np.nan, dtype=np.float64)
    for i, out in enumerate(outputs):
        flat = out.reshape(h, -1).double()
        norms[i] = torch.einsum("hd,hd->h", flat, flat).numpy() / flat.shape[1]
    for i in range(n):
        ai = outputs[i].reshape(h, -1).double()
        for j in range(n):
            bj = outputs[j].reshape(h, -1).double()
            raw[i, j] = (ai @ bj.T).numpy() / ai.shape[1]
    local = np.full_like(raw, np.nan)
    for i in range(n):
        for j in range(n):
            denom = np.sqrt(np.outer(norms[i], norms[j]))
            with np.errstate(invalid="ignore", divide="ignore"):
                sim = raw[i, j] / denom
            local[i, j] = np.where(np.isfinite(sim) & np.isfinite(denom) & (denom > 0), sim, np.nan)
    return raw, local, norms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=None)
    parser.add_argument("--step_interval", type=int, default=500)
    parser.add_argument("--no_step_interval", action="store_true")
    parser.add_argument("--linear_checkpoints", type=int, default=0)
    parser.add_argument("--log_checkpoints", type=int, default=0)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--n_samples", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    device = torch.device(device_name)
    step_interval = None if args.no_step_interval else args.step_interval
    steps = select_steps(run_dir, args.steps, step_interval, args.linear_checkpoints, args.log_checkpoints)

    first_model, cfg = load_model(run_dir, steps[0], device)
    seed = int(cfg.get("seed", 42) if args.seed is None else args.seed)
    dataset = build_dataset(cfg, args.n_samples, seed, args.split)

    outputs = []
    for idx, step in enumerate(tqdm(steps, desc="Empirical checkpoints", unit="ckpt")):
        model = first_model if idx == 0 else load_model(run_dir, step, device)[0]
        outputs.append(checkpoint_head_logits(model, dataset, args.batch_size, device))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    raw, local, norms = compute_sims(outputs)
    labels = np.array([f"head{h}" for h in range(raw.shape[-1])], dtype=object)
    output_path = Path(args.output) if args.output else run_dir / "empirical_l1_head_sims.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        steps=np.array(steps, dtype=np.int64),
        head_labels=labels,
        empirical_head_values=raw,
        empirical_head_local_sims=local,
        empirical_head_norms=norms,
        n_samples=args.n_samples,
        split=args.split,
        seed=seed,
        run_dir=str(run_dir),
    )
    print(f"Wrote empirical L1 head data: {output_path}")
    print(f"steps={steps}")
    print(f"finite_local_sims={np.isfinite(local).sum()}/{local.size}")


if __name__ == "__main__":
    main()
