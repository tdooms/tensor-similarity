import torch
import torch.nn.functional as F


def next_token_ce(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Compute next-token cross-entropy loss.
    
    Args:
        logits: Model output of shape (B, T, V)
        input_ids: Token IDs of shape (B, T)
        label_smoothing: Label smoothing factor
        
    Returns:
        Scalar loss tensor
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    
    B, T, V = shift_logits.shape
    loss = F.cross_entropy(
        shift_logits.view(B * T, V),
        shift_labels.view(B * T),
        label_smoothing=label_smoothing,
    )
    return loss


def per_position_ce(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute per-position cross-entropy loss (for analysis).
    
    Args:
        logits: Model output of shape (B, T, V)
        input_ids: Token IDs of shape (B, T)
        
    Returns:
        Per-position loss of shape (T-1,)
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    
    B, T, V = shift_logits.shape
    loss = F.cross_entropy(
        shift_logits.view(B * T, V),
        shift_labels.view(B * T),
        reduction="none",
    )
    return loss.view(B, T).mean(dim=0)


def compute_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    loss_type: str = "next_token_ce",
    label_smoothing: float = 0.0,
    **kwargs,
) -> torch.Tensor:
    """Dispatch to appropriate loss function.
    
    Args:
        logits: Model output of shape (B, T, V)
        input_ids: Token IDs of shape (B, T)
        loss_type: Type of loss ("next_token_ce")
        label_smoothing: Label smoothing factor
        
    Returns:
        Loss tensor
    """
    if loss_type == "next_token_ce":
        return next_token_ce(logits, input_ids, label_smoothing)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
