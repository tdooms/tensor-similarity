"""
Debug script to verify tensor similarity computation.
"""

import torch
import numpy as np
from tensor_sim_experiment import BilinearMLP, tensor_inner_product, tensor_similarity

# Create two models with known seeds
torch.manual_seed(42)
model1 = BilinearMLP(input_dim=784, hidden_dim=128, output_dim=10)

torch.manual_seed(43)
model2 = BilinearMLP(input_dim=784, hidden_dim=128, output_dim=10)

print("=" * 60)
print("SANITY CHECKS")
print("=" * 60)

# Check 1: Self-similarity should be 1.0
self_sim = tensor_similarity(model1, model1)
print(f"\n1. Self-similarity (should be 1.0): {self_sim:.6f}")

# Check 2: Inner products
inner_11 = tensor_inner_product(model1, model1)
inner_22 = tensor_inner_product(model2, model2)
inner_12 = tensor_inner_product(model1, model2)

print(f"\n2. Inner products:")
print(f"   inner(M1, M1) = {inner_11:.6e}")
print(f"   inner(M2, M2) = {inner_22:.6e}")
print(f"   inner(M1, M2) = {inner_12:.6e}")
print(f"   cosine = {inner_12 / np.sqrt(inner_11 * inner_22):.6f}")

# Check 3: Weight statistics
print(f"\n3. Weight statistics:")
print(f"   W_l1 shape: {model1.W_l.shape}, mean: {model1.W_l.mean():.6f}, std: {model1.W_l.std():.6f}")
print(f"   W_r1 shape: {model1.W_r.shape}, mean: {model1.W_r.mean():.6f}, std: {model1.W_r.std():.6f}")
print(f"   W_p1 shape: {model1.W_p.shape}, mean: {model1.W_p.mean():.6f}, std: {model1.W_p.std():.6f}")

# Check 4: Gram matrix statistics
W_l1 = model1.W_l.detach()
W_r1 = model1.W_r.detach()
W_p1 = model1.W_p.detach()
W_l2 = model2.W_l.detach()
W_r2 = model2.W_r.detach()
W_p2 = model2.W_p.detach()

ll = W_l1 @ W_l2.T
rr = W_r1 @ W_r2.T
lr = W_l1 @ W_r2.T
rl = W_r1 @ W_l2.T

print(f"\n4. Gram matrix statistics (model1 vs model2):")
print(f"   ll: shape {ll.shape}, mean: {ll.mean():.6e}, std: {ll.std():.6e}")
print(f"   rr: shape {rr.shape}, mean: {rr.mean():.6e}, std: {rr.std():.6e}")
print(f"   lr: shape {lr.shape}, mean: {lr.mean():.6e}, std: {lr.std():.6e}")
print(f"   rl: shape {rl.shape}, mean: {rl.mean():.6e}, std: {rl.std():.6e}")

aligned = ll * rr
swapped = lr * rl
core = 0.5 * (aligned + swapped)

print(f"\n5. Core computation:")
print(f"   aligned (ll*rr): mean: {aligned.mean():.6e}, std: {aligned.std():.6e}")
print(f"   swapped (lr*rl): mean: {swapped.mean():.6e}, std: {swapped.std():.6e}")
print(f"   core: mean: {core.mean():.6e}, std: {core.std():.6e}")

dd = W_p1.T @ W_p2
print(f"\n6. Output projection gram (dd):")
print(f"   dd: shape {dd.shape}, mean: {dd.mean():.6e}, std: {dd.std():.6e}")

final = core @ dd
print(f"\n7. Final (core @ dd):")
print(f"   shape: {final.shape}, trace: {torch.trace(final):.6e}")

# Check 5: What about using the formula as literally written?
# ll = W_l1^T @ W_l2 (not W_l1 @ W_l2^T)
print("\n" + "=" * 60)
print("ALTERNATIVE INTERPRETATION: ll = W_l1.T @ W_l2")
print("=" * 60)

ll_alt = W_l1.T @ W_l2  # (784, 128) @ (128, 784) won't work...
print(f"W_l1.T shape: {W_l1.T.shape}")
print(f"W_l2 shape: {W_l2.shape}")
# This would give (input, input) = (784, 784) - very large!

# Actually let's check if dimensions even work
print(f"\nDimension check:")
print(f"  W_l1.T @ W_l2 would be: ({W_l1.T.shape[0]}, {W_l1.T.shape[1]}) @ ({W_l2.shape[0]}, {W_l2.shape[1]})")
print(f"  = ({W_l1.T.shape[0]}, {W_l2.shape[1]}) = (784, 784)")

print("\n" + "=" * 60)
print("CHECKING WITH TRAINED MODELS")
print("=" * 60)

# Load a trained model and check
import os
ckpt_path = "checkpoints/seed_0/epoch_20.pt"
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    trained_model = BilinearMLP(input_dim=784, hidden_dim=128, output_dim=10)
    trained_model.load_state_dict(ckpt['model_state_dict'])

    print(f"\nTrained model weight stats:")
    print(f"   W_l: mean: {trained_model.W_l.mean():.6f}, std: {trained_model.W_l.std():.6f}")
    print(f"   W_r: mean: {trained_model.W_r.mean():.6f}, std: {trained_model.W_r.std():.6f}")
    print(f"   W_p: mean: {trained_model.W_p.mean():.6f}, std: {trained_model.W_p.std():.6f}")

    print(f"\nSelf-similarity of trained model: {tensor_similarity(trained_model, trained_model):.6f}")

    # Compare to random init
    torch.manual_seed(999)
    random_model = BilinearMLP(input_dim=784, hidden_dim=128, output_dim=10)
    print(f"Similarity (trained vs random): {tensor_similarity(trained_model, random_model):.6f}")
