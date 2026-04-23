"""In-context learning (ICL) score.

Implements ICL_{k1:k2} = l_{k2} - l_{k1}, where l_k is the average
cross-entropy loss predicting token x_k from context [BOS, x_1, ..., x_{k-1}].

A more negative score means the model is better at later positions relative
to earlier ones, indicating stronger in-context learning.
"""
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def compute_token_k_losses(
    model: torch.nn.Module,
    dataloader,
    k1: int,
    k2: int,
    bos_token_id: int,
    device: str = "cpu",
    max_batches: Optional[int] = None,
    prepend_bos: bool = True,
) -> Tuple[float, float]:
    """Compute l_{k1} and l_{k2} efficiently with one forward pass per sample.

    Two input conventions are supported:

    - ``prepend_bos=True`` (legacy / pretrained-tokenizer runs):
        Input:  ``[BOS, x_1, ..., x_{k2-1}]``  (length k2)
        logits[:, j] predicts x_{j+1}. So l_{k} uses logits[:, k-1] → x_k.
        Requires k1 >= 1.
    - ``prepend_bos=False`` (training distribution is contiguous text,
        e.g. Pile-DSIR streaming):
        Input:  ``[x_1, ..., x_{k2-1}]``       (length k2-1)
        logits[:, j] predicts x_{j+2}. So l_{k} uses logits[:, k-2] → x_k.
        Requires k1 >= 2.

    Args:
        model: The language model
        dataloader: Validation dataloader yielding dicts with 'input_ids'
                    and optionally 'attention_mask'
        k1: Earlier token position (1-indexed, e.g. 50)
        k2: Later token position (1-indexed, e.g. 500). Must be > k1.
        bos_token_id: BOS token id (used only when prepend_bos=True)
        device: Device for computation
        max_batches: Maximum batches to process (None = all)
        prepend_bos: If True, prepend BOS to the context (matches
            pretrained-tokenizer training). If False, feed raw contiguous
            tokens (matches Pile-DSIR streaming training).

    Returns:
        (l_k1, l_k2) averaged cross-entropy losses
    """
    min_k1 = 1 if prepend_bos else 2
    assert k2 > k1 >= min_k1, (
        f"Need k2 > k1 >= {min_k1} (prepend_bos={prepend_bos}), "
        f"got k1={k1}, k2={k2}"
    )

    model.eval()
    total_loss_k1 = 0.0
    total_loss_k2 = 0.0
    n_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)  # (B, T)
            attn_mask = batch.get("attention_mask")
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)

            B, T = input_ids.shape

            # Need at least k2 real tokens
            if attn_mask is not None:
                real_len = attn_mask.sum(dim=1)  # (B,)
                valid = real_len >= k2
                if not valid.any():
                    continue
                input_ids = input_ids[valid]
                B = input_ids.shape[0]
            elif T < k2:
                continue

            # real_tokens[:, i] == x_{i+1} (1-indexed)
            real_tokens = input_ids[:, :k2]  # (B, k2) — x_1 ... x_{k2}

            if prepend_bos:
                bos_col = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)
                inp = torch.cat([bos_col, real_tokens[:, :-1]], dim=1)  # (B, k2)
                idx_k1, idx_k2 = k1 - 1, k2 - 1
            else:
                # Feed raw contiguous tokens [x_1, ..., x_{k2-1}] of length k2-1.
                inp = real_tokens[:, : k2 - 1]  # (B, k2-1)
                idx_k1, idx_k2 = k1 - 2, k2 - 2

            logits = model(inp)  # (B, L, V)

            target_k1 = real_tokens[:, k1 - 1]  # (B,)  == x_{k1}
            loss_k1 = F.cross_entropy(logits[:, idx_k1, :], target_k1, reduction="sum")

            target_k2 = real_tokens[:, k2 - 1]  # (B,)  == x_{k2}
            loss_k2 = F.cross_entropy(logits[:, idx_k2, :], target_k2, reduction="sum")

            total_loss_k1 += loss_k1.item()
            total_loss_k2 += loss_k2.item()
            n_samples += B

    if n_samples == 0:
        return float("nan"), float("nan")

    return total_loss_k1 / n_samples, total_loss_k2 / n_samples


def compute_icl_score(
    model: torch.nn.Module,
    dataloader,
    bos_token_id: int,
    k1: int = 50,
    k2: int = 500,
    device: str = "cpu",
    max_batches: Optional[int] = None,
    prepend_bos: bool = True,
) -> Dict[str, float]:
    """Compute the ICL score: l_{k2} - l_{k1}.

    More negative = better in-context learning (lower loss later than earlier).

    Args:
        model: The language model
        dataloader: Validation dataloader
        bos_token_id: BOS token id to prepend
        k1: Earlier token position (default 50)
        k2: Later token position (default 500)
        device: Device for computation
        max_batches: Maximum batches to process (None = all)

    Returns:
        Dict with keys: loss_{k1}, loss_{k2}, icl_{k1}_{k2}
    """
    l_k1, l_k2 = compute_token_k_losses(
        model, dataloader, k1, k2, bos_token_id,
        device=device, max_batches=max_batches,
        prepend_bos=prepend_bos,
    )

    return {
        f"loss_{k1}": l_k1,
        f"loss_{k2}": l_k2,
        f"icl_{k1}_{k2}": l_k2 - l_k1,
    }
