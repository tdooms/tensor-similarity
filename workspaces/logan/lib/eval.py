import torch
from .losses import compute_loss


@torch.no_grad()
def evaluate(model, dataloader, device="cpu", max_batches=None):
    model.eval()
    total_loss, n = 0.0, 0
    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        logits = model(input_ids)
        total_loss += compute_loss(logits, input_ids).item()
        n += 1
    return total_loss / max(1, n)
