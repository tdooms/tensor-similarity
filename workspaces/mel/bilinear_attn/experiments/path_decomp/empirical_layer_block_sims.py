#!/usr/bin/env python3
"""Compute empirical layer-block similarity heatmaps from induction data.

This is the empirical analogue of ``layer_block_tn_from_paths.py``. It groups
the model logits into:

    direct, layer1, layer2

and computes raw/local cosine similarities between those group logits across
checkpoints on generated induction data.
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
from experiments.path_decomp.path_pair_tn_heatmaps import checkpoint_steps, select_steps  # noqa: E402
from models import AttentionLM  # noqa: E402


GROUP_LABELS = ["direct", "layer1", "layer2"]


def checkpoint_path(run_dir: Path, step: int) -> Path:
    exact = run_dir / "checkpoints" / f"step_{step}.pt"
    if exact.exists():
        return exact
    matches = sorted((run_dir / "checkpoints").glob(f"step_*{step}.pt"))
    for path in matches:
        stem_step = int(path.stem.removeprefix("step_"))
        if stem_step == step:
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
        energy_t0 = total[:, 0, :].pow(2).mean(dim=-1)
        scale = (energy_t0.mean() + eps).rsqrt()
        return torch.ones((*total.shape[:-1], 1), device=total.device, dtype=total.dtype) * scale
    raise ValueError(
        f"Empirical decomposition only supports no norm, tok0, or tok0_batch at pre_unembed; got {norm_type!r}."
    )


@torch.no_grad()
def group_logits(model: AttentionLM, input_ids: torch.Tensor) -> torch.Tensor:
    """Return tensor with shape (3, batch, seq, vocab)."""
    if model.attn_type != "quadratic":
        raise ValueError(f"Empirical layer decomposition currently supports quadratic attention, got {model.attn_type!r}")
    if model.n_layers != 2:
        raise ValueError(f"Expected 2 layers, got {model.n_layers}")
    if getattr(model, "embed_norm", None) is not None:
        raise ValueError("post_embed normalization is not supported for empirical group decomposition.")
    if getattr(model, "layer_norms", None) is not None:
        raise ValueError("pre_layer normalization is not supported for empirical group decomposition.")

    h0 = model.embed(input_ids)
    attn1, attn2 = model.layers[0], model.layers[1]

    l1_active = attn1.scale * active_quadratic(attn1, h0, h0, h0)
    r0 = (1.0 - attn1.scale) * h0

    direct = (1.0 - attn2.scale) * r0
    layer1 = (1.0 - attn2.scale) * l1_active
    z = r0 + l1_active
    layer2 = attn2.scale * active_quadratic(attn2, z, z, z)

    total = direct + layer1 + layer2
    scale = final_norm_scale(model, total)
    groups = torch.stack([direct * scale, layer1 * scale, layer2 * scale], dim=0)
    return torch.stack([model.unembed(groups[i]) for i in range(3)], dim=0)


def build_dataset(cfg: dict, n_samples: int, seed: int, split: str) -> RepeatedTokenDataset:
    model_cfg = cfg["model"]
    data_cfg = cfg.get("data", {})
    use_bos = data_cfg.get("use_bos", False)
    bos_token_id = data_cfg.get("bos_token_id")
    if use_bos and bos_token_id is None:
        bos_token_id = model_cfg["vocab_size"] - 1
    if not use_bos:
        bos_token_id = None
    split_seed = seed if split == "train" else seed + 1
    return RepeatedTokenDataset(
        vocab_size=model_cfg["vocab_size"],
        n_ctx=model_cfg["n_ctx"],
        n_samples=n_samples,
        seed=split_seed,
        bos_token_id=bos_token_id,
    )


@torch.no_grad()
def checkpoint_group_logits(
    model: AttentionLM,
    dataset: RepeatedTokenDataset,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    chunks = []
    for start in tqdm(range(0, len(dataset), batch_size), desc="Empirical batches", leave=False):
        batch = dataset.data[start : start + batch_size].to(device)
        chunks.append(group_logits(model, batch).detach().cpu().float())
    # (3, samples, seq, vocab)
    return torch.cat(chunks, dim=1)


def compute_sims(group_outputs: list[torch.Tensor]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(group_outputs)
    raw = np.full((n, n, 3, 3), np.nan, dtype=np.float64)
    norms = np.full((n, 3), np.nan, dtype=np.float64)
    for i, out in enumerate(group_outputs):
        flat = out.reshape(3, -1).double()
        norms[i] = torch.einsum("gd,gd->g", flat, flat).numpy() / flat.shape[1]

    for i in range(n):
        ai = group_outputs[i].reshape(3, -1).double()
        for j in range(n):
            bj = group_outputs[j].reshape(3, -1).double()
            raw[i, j] = (ai @ bj.T).numpy() / ai.shape[1]

    local = np.full_like(raw, np.nan)
    for i in range(n):
        for j in range(n):
            denom = np.sqrt(np.outer(norms[i], norms[j]))
            with np.errstate(invalid="ignore", divide="ignore"):
                sim = raw[i, j] / denom
            local[i, j] = np.where(np.isfinite(sim) & np.isfinite(denom) & (denom > 0), sim, np.nan)
    return raw, local, norms


def main() -> None:
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
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    step_interval = None if args.no_step_interval else args.step_interval
    steps = select_steps(run_dir, args.steps, step_interval, args.linear_checkpoints, args.log_checkpoints)

    first_model, cfg = load_model(run_dir, steps[0], device)
    seed = int(cfg.get("seed", 42) if args.seed is None else args.seed)
    dataset = build_dataset(cfg, args.n_samples, seed, args.split)

    outputs = []
    for idx, step in enumerate(tqdm(steps, desc="Empirical checkpoints", unit="ckpt")):
        model = first_model if idx == 0 else load_model(run_dir, step, device)[0]
        outputs.append(checkpoint_group_logits(model, dataset, args.batch_size, device))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    raw, local, norms = compute_sims(outputs)
    output_path = (
        Path(args.output)
        if args.output is not None
        else run_dir / "empirical_layer_block_sims.npz"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        steps=np.array(steps, dtype=np.int64),
        group_labels=np.array(GROUP_LABELS, dtype=object),
        empirical_block_values=raw,
        empirical_block_local_sims=local,
        empirical_block_norms=norms,
        n_samples=args.n_samples,
        split=args.split,
        seed=seed,
        run_dir=str(run_dir),
    )
    print(f"Wrote empirical layer-block data: {output_path}")
    print(f"steps={steps}")
    print(f"finite_local_sims={np.isfinite(local).sum()}/{local.size}")


if __name__ == "__main__":
    main()
