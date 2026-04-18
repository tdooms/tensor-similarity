import torch
from .losses import compute_loss


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader,
    device: str = "cpu",
    max_batches: int = None,
) -> float:
    """Evaluate model on validation set.
    
    Args:
        model: The model to evaluate
        dataloader: Validation dataloader
        device: Device to run on
        max_batches: Maximum number of batches to evaluate (None = all)
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    total_weight = 0
    
    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
            
        input_ids = batch["input_ids"].to(device)
        logits = model(input_ids)
        loss = compute_loss(logits, input_ids)
        
        batch_size = batch["input_ids"].shape[0]
        total_loss += loss.item() * batch_size
        total_weight += batch_size
    
    return total_loss / max(1, total_weight)
