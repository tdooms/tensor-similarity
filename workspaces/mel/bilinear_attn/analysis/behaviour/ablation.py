"""Position-ablated loss for measuring positional embedding contribution.

Computes the validation loss with RoPE disabled (replaced by identity),
giving a measure of how much the model relies on positional information.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from typing import Optional


@contextmanager
def ablate_rotary(model: nn.Module):
    """Context manager that temporarily replaces all Rotary modules with identity.
    
    Finds every ``Rotary`` instance in the model's submodules and swaps its
    ``forward`` method with an identity function.  On exit the original
    forwards are restored.
    
    Args:
        model: The model whose Rotary modules should be ablated
    """
    from models.attention_kernels.rotary import Rotary
    
    originals = {}
    for name, module in model.named_modules():
        if isinstance(module, Rotary):
            originals[name] = module.forward
            module.forward = lambda x, _orig=module: x  # identity
    
    try:
        yield
    finally:
        for name, module in model.named_modules():
            if name in originals:
                module.forward = originals[name]


@torch.no_grad()
def compute_ablated_loss(
    model: nn.Module,
    dataloader,
    device: str = "cpu",
    max_batches: Optional[int] = None,
) -> float:
    """Compute validation loss with positional embeddings ablated.
    
    Args:
        model: The language model
        dataloader: Validation dataloader yielding batches with 'input_ids'
        device: Device for computation
        max_batches: Maximum batches to evaluate (None = all)
        
    Returns:
        Average cross-entropy loss with RoPE disabled
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with ablate_rotary(model):
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            
            input_ids = batch["input_ids"].to(device)
            logits = model(input_ids)
            
            # Standard next-token cross-entropy
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                input_ids[:, 1:].contiguous().view(-1),
            )
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / max(1, n_batches)


@torch.no_grad()
def compute_val_loss(
    model: nn.Module,
    dataloader,
    device: str = "cpu",
    max_batches: Optional[int] = None,
) -> float:
    """Compute standard validation loss (for pairing with ablated loss).
    
    Args:
        model: The language model
        dataloader: Validation dataloader yielding batches with 'input_ids'
        device: Device for computation
        max_batches: Maximum batches to evaluate (None = all)
        
    Returns:
        Average cross-entropy loss
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        
        input_ids = batch["input_ids"].to(device)
        logits = model(input_ids)
        
        loss = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
            input_ids[:, 1:].contiguous().view(-1),
        )
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(1, n_batches)
