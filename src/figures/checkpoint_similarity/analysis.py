"""CPU-only mechanistic analyses on the cached checkpoint state-dicts.

Reads `_downloads/checkpoint-similarity/checkpoints/step_NNNNN.pt`, applies the
same `_absorb_final_norm` rewrite that `load_pt` does (so the analysed weights
match the function the TN similarity sees), and writes per-layer trajectories
to `artifacts/cache/checkpoint_similarity/`.

Outputs:
    weight_norms.feather       — step, param, l2_norm
    weight_deltas.feather      — step, param, frob_delta, frob_delta_rel
    singular_values.feather    — step, param, rank, sigma
    weight_cosine.feather      — step, param, cos_to_final
    pca.feather                — step, pc1, pc2, pc3
    pca_summary.json           — explained variance ratios
    metric_correlations.feather — metric_a, metric_b, corr (incl. cos→final)
"""
from itertools import accumulate
import json

import polars as pl
import torch
from loguru import logger
from tqdm import tqdm

from src.figures import CACHE_DIR
from src.figures.checkpoint_similarity.prepare import INPUT_DIR
from src.models.checkpoint_transformer import _absorb_final_norm

CACHE = CACHE_DIR / "checkpoint_similarity"


def _flatten_state(state):
    """Concatenate all parameter tensors into one 1D vector. Stable order."""
    keys = sorted(k for k in state if state[k].dtype.is_floating_point)
    return torch.cat([state[k].flatten() for k in keys])


def _load_state(path):
    """Apply the same norm-absorption + embed transpose as `load_pt` so the
    weights we analyse match the function the TN similarity sees."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob["model_state_dict"] if "model_state_dict" in blob else blob
    state = _absorb_final_norm(state)
    state = dict(state)
    state["embed.weight"] = state["embed.weight"].T.contiguous()
    return {k: v for k, v in state.items() if v.dtype.is_floating_point}


def _params_of_interest(state):
    """Skip biases (mostly zero per init), skip the TN-irrelevant scalars.
    Keeps all weight matrices (embed, q*, k*, v, o, unembed)."""
    return [k for k in sorted(state) if k.endswith(".weight") or k == "embed.weight" or k == "unembed.weight"]


def main():
    picked = json.loads((CACHE / "metadata.json").read_text(encoding="utf-8"))["steps"]
    logger.info(f"[1/6] loading {len(picked)} state-dicts from {INPUT_DIR}/checkpoints")
    states = [_load_state(INPUT_DIR / f"checkpoints/step_{s:05d}.pt") for s in tqdm(picked, desc="load")]
    params = _params_of_interest(states[0])
    final = states[-1]

    # A1: L2 norms per layer per step
    logger.info("[2/6] A1: per-layer L2 norms")
    norm_rows = [{"step": s, "param": p, "l2_norm": float(states[i][p].norm())}
                 for i, s in enumerate(picked) for p in params]
    pl.DataFrame(norm_rows).write_ipc(CACHE / "weight_norms.feather")

    # A2: step-to-step deltas per layer
    logger.info("[3/6] A2: step-to-step weight deltas")
    delta_rows = []
    for i in range(1, len(picked)):
        for p in params:
            delta = (states[i][p] - states[i - 1][p]).norm().item()
            denom = states[i][p].norm().item() + 1e-12
            delta_rows.append({"step": picked[i], "param": p,
                               "frob_delta": delta, "frob_delta_rel": delta / denom})
    pl.DataFrame(delta_rows).write_ipc(CACHE / "weight_deltas.feather")

    # A3: top singular values
    logger.info("[4/6] A3: top-8 singular values per weight matrix")
    sv_rows = []
    for i, s in enumerate(tqdm(picked, desc="svd")):
        for p in params:
            w = states[i][p]
            if w.dim() != 2:
                continue
            sigmas = torch.linalg.svdvals(w.float())[:8]
            for k, sig in enumerate(sigmas.tolist()):
                sv_rows.append({"step": s, "param": p, "rank": k, "sigma": sig})
    pl.DataFrame(sv_rows).write_ipc(CACHE / "singular_values.feather")

    # A6: per-layer weight-space cosine to final
    logger.info("[5/6] A6: per-layer weight-cosine to final")
    cos_rows = []
    for i, s in enumerate(picked):
        for p in params:
            a, b = states[i][p].flatten().float(), final[p].flatten().float()
            cos = torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)
            cos_rows.append({"step": s, "param": p, "cos_to_final": float(cos)})
    pl.DataFrame(cos_rows).write_ipc(CACHE / "weight_cosine.feather")

    # A5: PCA of flattened per-step weight vectors
    logger.info("[6/6] A5: PCA over per-step flattened weights + A4: metric correlations")
    flat = torch.stack([_flatten_state(st).float() for st in states])  # (T, D)
    centred = flat - flat.mean(0)
    u, s_pca, _ = torch.linalg.svd(centred, full_matrices=False)
    explained = (s_pca ** 2 / (s_pca ** 2).sum()).tolist()
    proj = u * s_pca  # (T, T)
    pca_rows = [{"step": picked[i], "pc1": float(proj[i, 0]),
                 "pc2": float(proj[i, 1]), "pc3": float(proj[i, 2])} for i in range(len(picked))]
    pl.DataFrame(pca_rows).write_ipc(CACHE / "pca.feather")
    cumulative = list(accumulate(explained[:8]))
    (CACHE / "pca_summary.json").write_text(
        json.dumps({"explained_variance": explained[:8], "cumulative": cumulative}, indent=2),
        encoding="utf-8",
    )

    # A4: metric correlations (incl. cos→final)
    behavior = pl.read_ipc(CACHE / "behavior.feather")
    matrix = pl.read_ipc(CACHE / "matrix.feather")
    n = len(picked)
    mat = torch.zeros(n, n, dtype=torch.float64)
    idx = {s: i for i, s in enumerate(picked)}
    for r in matrix.iter_rows(named=True):
        mat[idx[int(r["step_i"])], idx[int(r["step_j"])]] = float(r["similarity"])
    cos_final = mat[:, -1]

    pivot = behavior.pivot(values="value", index="step", on="metric").sort("step")
    metric_names = [c for c in pivot.columns if c != "step"]
    arr = torch.tensor(pivot.select(metric_names).to_numpy(), dtype=torch.float64)
    arr = torch.cat([arr, cos_final[: arr.shape[0]].unsqueeze(1)], dim=1)
    metric_names = metric_names + ["cos_to_final"]

    finite = arr.isfinite().all(dim=0)
    keep = [i for i, f in enumerate(finite.tolist()) if f]
    arr = arr[:, keep]
    metric_names = [metric_names[i] for i in keep]
    centred = arr - arr.mean(dim=0)
    cov = centred.T @ centred
    std = (cov.diag() + 1e-30).sqrt()
    corr = cov / (std.unsqueeze(0) * std.unsqueeze(1))
    corr_rows = [{"metric_a": metric_names[i], "metric_b": metric_names[j], "corr": float(corr[i, j])}
                 for i in range(len(metric_names)) for j in range(len(metric_names))]
    pl.DataFrame(corr_rows).write_ipc(CACHE / "metric_correlations.feather")

    logger.info(f"      wrote 6 analyses to {CACHE}")


if __name__ == "__main__":
    main()
