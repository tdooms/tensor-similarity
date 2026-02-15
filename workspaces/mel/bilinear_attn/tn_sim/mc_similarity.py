"""Monte Carlo tensor similarity for bilinear attention models.

Estimates the tensor inner product by sampling inputs and computing
dot products of model outputs. Under the law of large numbers:

  <T_A, T_B> ≈ (1/N) sum_i f_A(x_i) · f_B(x_i)

where x_i are drawn from the input distribution and f is the full model function.

For cosine similarity:
  sim(A, B) = <T_A, T_B> / sqrt(<T_A, T_A> * <T_B, T_B>)
"""

import torch
import numpy as np
from experiments.induction_heads.data import create_repeated_token_dataloaders


@torch.no_grad()
def mc_inner_product(model_A, model_B, dataloader, device, max_batches=None):
    """Estimate <T_A, T_B> by averaging dot products of logit outputs.
    
    Uses the actual token data (not Gaussian) for a realistic estimate.
    
    Returns: scalar estimate of inner product.
    """
    model_A.eval()
    model_B.eval()
    
    total_ip = 0.0
    total_count = 0
    
    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        
        input_ids = batch["input_ids"].to(device)
        
        logits_A = model_A(input_ids)  # (B, T, V)
        logits_B = model_B(input_ids)  # (B, T, V)
        
        # Flatten to (B*T, V) and compute dot product per position
        B, T, V = logits_A.shape
        flat_A = logits_A.reshape(-1, V)
        flat_B = logits_B.reshape(-1, V)
        
        # Inner product: sum of element-wise products
        ip = (flat_A * flat_B).sum().item()
        total_ip += ip
        total_count += B * T
    
    return total_ip / max(1, total_count)


@torch.no_grad()
def mc_similarity(model_A, model_B, dataloader, device, max_batches=None):
    """Compute MC cosine tensor similarity.
    
    Returns: cosine similarity in [-1, 1].
    """
    ip_AB = mc_inner_product(model_A, model_B, dataloader, device, max_batches)
    ip_AA = mc_inner_product(model_A, model_A, dataloader, device, max_batches)
    ip_BB = mc_inner_product(model_B, model_B, dataloader, device, max_batches)
    
    denom = np.sqrt(abs(ip_AA) * abs(ip_BB))
    if denom < 1e-30:
        return 0.0
    return ip_AB / denom


@torch.no_grad()
def mc_similarity_gaussian(model_A, model_B, vocab_size, n_ctx, device,
                           n_samples=2000, batch_size=64):
    """Compute MC cosine similarity using random token inputs (uniform distribution).
    
    This matches the Gaussian assumption used in TN similarity more closely.
    """
    model_A.eval()
    model_B.eval()
    
    ip_AB, ip_AA, ip_BB = 0.0, 0.0, 0.0
    total_count = 0
    
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for _ in range(n_batches):
        bs = min(batch_size, n_samples - total_count // n_ctx)
        if bs <= 0:
            break
        
        input_ids = torch.randint(0, vocab_size, (bs, n_ctx), device=device)
        
        logits_A = model_A(input_ids)
        logits_B = model_B(input_ids)
        
        B, T, V = logits_A.shape
        flat_A = logits_A.reshape(-1, V)
        flat_B = logits_B.reshape(-1, V)
        
        ip_AB += (flat_A * flat_B).sum().item()
        ip_AA += (flat_A * flat_A).sum().item()
        ip_BB += (flat_B * flat_B).sum().item()
        total_count += B * T
    
    ip_AB /= max(1, total_count)
    ip_AA /= max(1, total_count)
    ip_BB /= max(1, total_count)
    
    denom = np.sqrt(abs(ip_AA) * abs(ip_BB))
    if denom < 1e-30:
        return 0.0
    return ip_AB / denom
