"""
Test tensor similarity between nearby checkpoints of the same model.
Epoch 17 vs Epoch 20 should have HIGH similarity.
"""

import torch
from tensor_sim_experiment import BilinearMLP, tensor_similarity

def load_checkpoint(seed, epoch):
    path = f"checkpoints/seed_{seed}/epoch_{epoch:02d}.pt"
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model = BilinearMLP(input_dim=784, hidden_dim=128, output_dim=10)
    model.load_state_dict(ckpt['model_state_dict'])
    return model, ckpt['accuracy']

print("="*60)
print("CHECKPOINT SIMILARITY TEST")
print("="*60)

# Load checkpoints from seed 0
epochs = [1, 5, 9, 13, 17, 20]

print("\n1. Loading checkpoints from seed 0...")
models = {}
for epoch in epochs:
    model, acc = load_checkpoint(0, epoch)
    models[epoch] = model
    print(f"   Epoch {epoch:2d}: accuracy = {acc:.4f}")

print("\n2. Tensor similarity matrix (seed 0):")
print("      ", end="")
for e in epochs:
    print(f"  ep{e:02d}", end="")
print()

for e1 in epochs:
    print(f"  ep{e1:02d}", end="")
    for e2 in epochs:
        sim = tensor_similarity(models[e1], models[e2])
        print(f"  {sim:5.3f}", end="")
    print()

print("\n3. Key comparisons:")
print(f"   Epoch 17 vs Epoch 20: {tensor_similarity(models[17], models[20]):.6f}")
print(f"   Epoch 13 vs Epoch 17: {tensor_similarity(models[13], models[17]):.6f}")
print(f"   Epoch 1 vs Epoch 20:  {tensor_similarity(models[1], models[20]):.6f}")
print(f"   Epoch 5 vs Epoch 9:   {tensor_similarity(models[5], models[9]):.6f}")

# Also check across seeds
print("\n4. Cross-seed comparison (epoch 20):")
model_seed0, _ = load_checkpoint(0, 20)
model_seed1, _ = load_checkpoint(1, 20)
model_seed2, _ = load_checkpoint(2, 20)

print(f"   Seed 0 vs Seed 1: {tensor_similarity(model_seed0, model_seed1):.6f}")
print(f"   Seed 0 vs Seed 2: {tensor_similarity(model_seed0, model_seed2):.6f}")
print(f"   Seed 1 vs Seed 2: {tensor_similarity(model_seed1, model_seed2):.6f}")
