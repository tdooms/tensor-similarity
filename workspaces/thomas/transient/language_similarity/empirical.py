"""Empirical cosine similarity on discrete-token inputs.

Forwards the *actual* bundled `AttentionLM` (with `tok0_batch` norm in eval mode
using running stats) on a batch of random token IDs from the model's vocab.
Compares the resulting logit cosine trajectory to the TN/Gaussian one. If they
agree, the TN bump→negative→climb structure is genuine on real-shaped inputs.

Also emits per-position divergence ranking between two chosen checkpoints
(default: step 5947, the start of the late climb, vs step 20000, final).
Outputs concrete examples of where late training changes predictions.

Outputs (under this script's local `cache/`):
    empirical_cosine.feather   step, cos_to_final
    divergence_examples.json   list of {context_tokens, low_cos_step, top5_a, top5_b}
"""
import json
import sys
from pathlib import Path

import polars as pl
import torch
from loguru import logger
from tqdm import tqdm

from src.figures.language_similarity.prepare import CACHE as CANONICAL, INPUT_DIR

CACHE = Path(__file__).parent / "cache"

BATCH = 64
N_CTX = 64
SEED = 0
COMPARE_PAIR = (5947, 20000)
TOP_K = 5
N_EXAMPLES = 6


def _bundled_attention_lm():
    """Load the AttentionLM class shipped *with* the dataset (matches the
    forward that produced the checkpoints, including `tok0_batch` norm)."""
    sys.path.insert(0, str(INPUT_DIR))
    from models.transformer import AttentionLM
    return AttentionLM


def _load_model(AttentionLM, config_dict, ckpt_path):
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = blob["model_state_dict"] if "model_state_dict" in blob else blob
    model = AttentionLM.from_config(config_dict).eval()
    # All checkpoints use running stats so the norm is a fixed scalar
    # rather than data-dependent. Step 0's `num_batches_tracked == 0`
    # would otherwise fall back to batch stats — manually bump it so
    # every checkpoint applies the same kind of scaling.
    if "final_norm.num_batches_tracked" in state:
        state["final_norm.num_batches_tracked"] = state["final_norm.num_batches_tracked"].clamp(min=1)
    model.load_state_dict(state)
    return model


@torch.no_grad()
def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    config = json.loads((INPUT_DIR / "config.json").read_text(encoding="utf-8"))
    picked = json.loads((CANONICAL / "metadata.json").read_text(encoding="utf-8"))["steps"]
    AttentionLM = _bundled_attention_lm()

    torch.manual_seed(SEED)
    inputs = torch.randint(0, config["model"]["vocab_size"], (BATCH, N_CTX))
    logger.info(f"[1/3] forwarding {len(picked)} checkpoints on inputs of shape {tuple(inputs.shape)}")

    logits_per_step = []
    for s in tqdm(picked, desc="forward", unit="ckpt"):
        m = _load_model(AttentionLM, config, INPUT_DIR / f"checkpoints/step_{s:05d}.pt")
        logits_per_step.append(m(inputs).cpu())  # (B, T, V)

    logger.info("[2/3] empirical pairwise cosine matrix, batch-averaged")
    flats = torch.stack([l.flatten() for l in logits_per_step])  # (N, B*T*V)
    norms = flats.norm(dim=-1)
    cos_matrix = (flats @ flats.T) / (norms[:, None] * norms[None, :] + 1e-12)
    rows = [{"step_i": picked[i], "step_j": picked[j], "similarity": float(cos_matrix[i, j])}
            for i in range(len(picked)) for j in range(len(picked))]
    pl.DataFrame(rows).write_ipc(CACHE / "empirical_matrix.feather")
    pl.DataFrame([{"step": picked[i], "cos_to_final": float(cos_matrix[i, -1])}
                  for i in range(len(picked))]).write_ipc(CACHE / "empirical_cosine.feather")

    # COMPARE_PAIR steps may not be in the actual `picked` list (depends on N_STEPS);
    # snap to the nearest available step so this works at any subsample size.
    sa = min(picked, key=lambda s: abs(s - COMPARE_PAIR[0]))
    sb = min(picked, key=lambda s: abs(s - COMPARE_PAIR[1]))
    logger.info(f"[3/3] divergence examples: step {sa} vs step {sb}")
    a = logits_per_step[picked.index(sa)]  # (B, T, V)
    b = logits_per_step[picked.index(sb)]
    # per-position cosine
    a_flat = a.reshape(-1, a.shape[-1])  # (B*T, V)
    b_flat = b.reshape(-1, b.shape[-1])
    pos_cos = (a_flat * b_flat).sum(-1) / (a_flat.norm(dim=-1) * b_flat.norm(dim=-1) + 1e-12)  # (B*T,)
    order = pos_cos.argsort()  # ascending — most divergent first

    examples = []
    for k in range(N_EXAMPLES):
        flat_idx = int(order[k])
        b_idx, t_idx = flat_idx // N_CTX, flat_idx % N_CTX
        context = inputs[b_idx, : t_idx + 1].tolist()
        top_a = a[b_idx, t_idx].topk(TOP_K)
        top_b = b[b_idx, t_idx].topk(TOP_K)
        examples.append({
            "rank": k + 1,
            "batch_idx": b_idx,
            "position": t_idx,
            "cosine": float(pos_cos[flat_idx]),
            "context_tokens": context,
            f"top{TOP_K}_step_{COMPARE_PAIR[0]}": [
                {"token_id": int(tok), "logit": float(val)}
                for val, tok in zip(top_a.values, top_a.indices)
            ],
            f"top{TOP_K}_step_{COMPARE_PAIR[1]}": [
                {"token_id": int(tok), "logit": float(val)}
                for val, tok in zip(top_b.values, top_b.indices)
            ],
        })
    (CACHE / "divergence_examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
    logger.info(f"      wrote empirical_cosine.feather + divergence_examples.json to {CACHE}")


if __name__ == "__main__":
    main()
