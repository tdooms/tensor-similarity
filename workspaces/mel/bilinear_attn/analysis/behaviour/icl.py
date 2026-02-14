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
) -> Tuple[float, float]:
    """Compute l_{k1} and l_{k2} efficiently with one forward pass per sample.

    For each validation sequence with >= k2 real tokens:
      - Input:  [BOS, x_1, ..., x_{k2-1}]   (length k2)
      - Run one forward pass
      - l_{k1} = CE(logits at position k1-1, x_{k1})
      - l_{k2} = CE(logits at position k2-1, x_{k2})

    Positions are 1-indexed over real tokens (excluding BOS).  In the
    input tensor [BOS, x_1, ..., x_{k2-1}], the logits at index j
    predict x_{j+1}.  So:
      - logits[:, k1-1, :] predicts x_{k1}  (target = input_ids[:, k1])
      - logits[:, k2-1, :] predicts x_{k2}  (target = input_ids[:, k2])

    Args:
        model: The language model
        dataloader: Validation dataloader yielding dicts with 'input_ids'
                    and optionally 'attention_mask'
        k1: Earlier token position (1-indexed, e.g. 50)
        k2: Later token position (1-indexed, e.g. 500). Must be > k1.
        bos_token_id: BOS token id to prepend
        device: Device for computation
        max_batches: Maximum batches to process (None = all)

    Returns:
        (l_k1, l_k2) averaged cross-entropy losses
    """
    assert k2 > k1 >= 1, f"Need k2 > k1 >= 1, got k1={k1}, k2={k2}"

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

            # Build input: [BOS, x_1, ..., x_{k2-1}]  (length k2)
            real_tokens = input_ids[:, :k2]  # (B, k2) — x_1 ... x_{k2}
            bos_col = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)
            inp = torch.cat([bos_col, real_tokens[:, :-1]], dim=1)  # (B, k2)

            logits = model(inp)  # (B, k2, V)

            # l_{k1}: logits at position k1-1 predict x_{k1}
            # In real_tokens, x_{k1} is at index k1-1 (0-indexed)
            target_k1 = real_tokens[:, k1 - 1]  # (B,)
            loss_k1 = F.cross_entropy(logits[:, k1 - 1, :], target_k1, reduction="sum")

            # l_{k2}: logits at position k2-1 predict x_{k2}
            target_k2 = real_tokens[:, k2 - 1]  # (B,)
            loss_k2 = F.cross_entropy(logits[:, k2 - 1, :], target_k2, reduction="sum")

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
    )

    return {
        f"loss_{k1}": l_k1,
        f"loss_{k2}": l_k2,
        f"icl_{k1}_{k2}": l_k2 - l_k1,
    }
