# %%
"""
Interactive test to measure similarity between models trained with different seeds.
"""

import torch
import numpy as np
import random
from pathlib import Path

from torch.optim import Muon, AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from kornia.augmentation import RandomGaussianNoise
from tqdm import tqdm
from itertools import cycle
from quimb.tensor import TensorNetwork, Tensor

from src.datasets import EMNIST, MNIST
from src.models.toy import ToyModel
from src.utils import sequential

# %%
def set_seed(seed: int = 0):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(seed: int, max_steps: int = 2**9, verbose: bool = True):
    """Train a model with a specific seed and return the trained model."""
    set_seed(seed)

    # Load data
    data = MNIST()

    # Initialize the model
    params = dict(
        n_layers=1,
        d_model=128,
        d_input=data.size[0]*data.size[1],
        d_output=data.classes
    )
    model = ToyModel.from_config(**params, tags=[f'seed_{seed}']).cuda()

    # Initialize optimizers and learning rate scheduler
    muon = Muon([p for p in model.parameters() if p.ndim >= 2], lr=0.03, weight_decay=0.1)
    adam = AdamW([p for p in model.parameters() if p.ndim < 2], lr=0.001, weight_decay=0.1)
    scheduler = LinearLR(muon, start_factor=1.0, end_factor=0.0, total_iters=max_steps)

    # Set up the data loader
    loader = DataLoader(data.train, batch_size=2048, shuffle=True, drop_last=True)
    progress = tqdm(
        zip(range(max_steps), cycle(loader)),
        total=max_steps,
        desc=f"Training seed {seed}",
        disable=not verbose
    )
    preprocess = RandomGaussianNoise(mean=0, std=0.1, p=1)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for step, (inputs, labels) in progress:
        inputs = preprocess(inputs.cuda().float() / 255.0)
        logits = model(inputs)
        loss = criterion(logits, labels.cuda())

        metrics = {
            'accuracy': float((logits.argmax(dim=1) == labels.cuda()).float().mean().detach()),
            'loss': float(loss.detach()),
        }

        if verbose:
            progress.set_description(
                f"Seed {seed} | loss = {metrics['loss']:.3f} | accuracy = {metrics['accuracy']:.3f}"
            )

        muon.zero_grad()
        adam.zero_grad()
        loss.backward()
        muon.step()
        adam.step()
        scheduler.step()

    return model, metrics


def inner_product(model1, model2, draw=False):
    """
    Compute the inner product between two models using tensor network contraction.

    This computes: <model1, model2> = trace(model1† * model2)

    Args:
        model1, model2: Models to compare
        draw: If True, draw the tensor network used for inner product computation

    Returns:
        Scalar inner product value
    """
    # Convert models to tensor networks
    net1 = sequential(
        model1.embed.network(),
        model1.body.network(),
        model1.head.network(),
        tag='m1'
    )
    net2 = sequential(
        model2.embed.network(),
        model2.body.network(),
        model2.head.network(),
        tag='m2'
    )

    # Debug: print index information
    if draw:
        print(f"net1 outer indices: {net1.outer_inds()}")
        print(f"net2 outer indices: {net2.outer_inds()}")
        print(f"net1 inner indices: {net1.inner_inds()}")
        print(f"net2 inner indices: {net2.inner_inds()}")

    # For inner product, we need to contract all corresponding indices
    # Create a copy of net2 and reindex its outer indices to match contraction
    net2_copy = net2.copy()

    # Get the outer indices and create a mapping
    # We want to contract net1's indices with net2's indices
    outer_map = {ix: ix for ix in net2_copy.outer_inds()}

    # Combine conjugate of net1 with net2 and contract everything
    combined = net1.conj() | net2_copy

    if draw:
        print(f"\nCombined network outer indices: {combined.outer_inds()}")
        print(f"Combined network inner indices: {combined.inner_inds()}")
        print("\nDrawing inner product network:")
        combined.draw(figsize=(10, 8))

    # Contract all indices to get scalar
    result = combined.contract(all, output_inds=[])

    return float(result.data.real)


def compute_similarity(model1, model2):
    """
    Compute normalized cosine similarity between two models.

    Returns a value between -1 and 1, where:
    - 1 means identical models
    - 0 means orthogonal models
    - -1 means opposite models
    """
    # Use the same inner_product function for all computations
    ip_12 = inner_product(model1, model2)
    ip_11 = inner_product(model1, model1)
    ip_22 = inner_product(model2, model2)

    # Compute cosine similarity
    norm1 = np.sqrt(ip_11)
    norm2 = np.sqrt(ip_22)

    similarity = ip_12 / (norm1 * norm2)

    return similarity

# %%
# Train first model with seed 0
print("Training model 1 with seed 0...")
model1, metrics1 = train_model(seed=0, max_steps=2**9)
print(f"Model 1 - Final loss: {metrics1['loss']:.4f}, accuracy: {metrics1['accuracy']:.4f}")

# %%
# Visualize the inner product network for model1 with itself (for debugging)
print("Visualizing inner product network (model1 with itself):")
ip_11 = inner_product(model1, model1, draw=True)
print(f"\n<model1, model1> = {ip_11}")
print(f"||model1|| = {np.sqrt(ip_11)}")

# %%
# Train second model with seed 1
print("Training model 2 with seed 1...")
model2, metrics2 = train_model(seed=1, max_steps=2**9)
print(f"Model 2 - Final loss: {metrics2['loss']:.4f}, accuracy: {metrics2['accuracy']:.4f}")

# %%
# Visualize the cross inner product network
print("Visualizing inner product network (model1 with model2):")
ip_12 = inner_product(model1, model2, draw=True)
ip_11 = inner_product(model1, model1, draw=False)
ip_22 = inner_product(model2, model2, draw=False)

print(f"\n<model1, model1> = {ip_11:.6f}")
print(f"<model2, model2> = {ip_22:.6f}")
print(f"<model1, model2> = {ip_12:.6f}")
print(f"\n||model1|| = {np.sqrt(ip_11):.6f}")
print(f"||model2|| = {np.sqrt(ip_22):.6f}")
print(f"\nCosine similarity = {ip_12 / (np.sqrt(ip_11) * np.sqrt(ip_22)):.6f}")

# %%
# Compute self-similarity for model 1 (should be exactly 1.0)
sim_11 = compute_similarity(model1, model1)
print(f"Model 1 vs Model 1 (self-similarity): {sim_11:.6f}")
assert np.abs(sim_11 - 1.0) < 1e-6, f"Self-similarity should be 1.0, got {sim_11}"

# %%
# Compute self-similarity for model 2 (should be exactly 1.0)
sim_22 = compute_similarity(model2, model2)
print(f"Model 2 vs Model 2 (self-similarity): {sim_22:.6f}")
assert np.abs(sim_22 - 1.0) < 1e-6, f"Self-similarity should be 1.0, got {sim_22}"

# %%
# Compute cross-similarity between models with different seeds
sim_12 = compute_similarity(model1, model2)
print(f"Model 1 vs Model 2 (cross-similarity): {sim_12:.6f}")
assert -1.0 <= sim_12 <= 1.0, f"Similarity should be in [-1, 1], got {sim_12}"

# %%
# Optional: Train a third model and compare
print("Training model 3 with seed 2...")
model3, metrics3 = train_model(seed=2, max_steps=2**9)
print(f"Model 3 - Final loss: {metrics3['loss']:.4f}, accuracy: {metrics3['accuracy']:.4f}")

# %%
# Compare all three models
print("\nPairwise Similarities:")
print(f"Model 1 vs Model 2: {compute_similarity(model1, model2):.6f}")
print(f"Model 1 vs Model 3: {compute_similarity(model1, model3):.6f}")
print(f"Model 2 vs Model 3: {compute_similarity(model2, model3):.6f}")

# %%
# Optional: Visualize one of the tensor networks
net = sequential(model1.embed.network(), model1.body.network(), model1.head.network(), tag='m1')
print(f"Model 1 norm: {net.make_norm().contract(all, output_inds=[]).data}")
net.draw()

# %%
# Visualize similarity matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Compute all pairwise similarities
models = [model1, model2, model3]
model_names = ["Seed 0", "Seed 1", "Seed 2"]
n_models = len(models)

similarity_matrix = np.zeros((n_models, n_models))
for i in range(n_models):
    for j in range(n_models):
        similarity_matrix[i, j] = compute_similarity(models[i], models[j])

# Create heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    similarity_matrix,
    annot=True,
    fmt='.4f',
    cmap='RdYlGn',
    center=0,
    vmin=-1,
    vmax=1,
    xticklabels=model_names,
    yticklabels=model_names,
    cbar_kws={'label': 'Similarity'}
)
plt.title('Model Similarity Matrix\n(Tensor Network Cosine Similarity)')
plt.tight_layout()
plt.show()

# %%
# Print summary statistics
print("\n" + "="*60)
print("SIMILARITY TEST RESULTS")
print("="*60)
print("\nFinal Training Metrics:")
print(f"  Model 1 (seed 0): loss={metrics1['loss']:.4f}, acc={metrics1['accuracy']:.4f}")
print(f"  Model 2 (seed 1): loss={metrics2['loss']:.4f}, acc={metrics2['accuracy']:.4f}")
print(f"  Model 3 (seed 2): loss={metrics3['loss']:.4f}, acc={metrics3['accuracy']:.4f}")

print("\nSelf-Similarities (should be ≈1.0):")
print(f"  Model 1 vs Model 1: {similarity_matrix[0, 0]:.6f}")
print(f"  Model 2 vs Model 2: {similarity_matrix[1, 1]:.6f}")
print(f"  Model 3 vs Model 3: {similarity_matrix[2, 2]:.6f}")

print("\nCross-Similarities (different seeds):")
print(f"  Model 1 vs Model 2: {similarity_matrix[0, 1]:.6f}")
print(f"  Model 1 vs Model 3: {similarity_matrix[0, 2]:.6f}")
print(f"  Model 2 vs Model 3: {similarity_matrix[1, 2]:.6f}")

# Compute statistics on cross-similarities
cross_sims = [similarity_matrix[0, 1], similarity_matrix[0, 2], similarity_matrix[1, 2]]
print(f"\nCross-Similarity Statistics:")
print(f"  Mean: {np.mean(cross_sims):.6f}")
print(f"  Std:  {np.std(cross_sims):.6f}")
print(f"  Min:  {np.min(cross_sims):.6f}")
print(f"  Max:  {np.max(cross_sims):.6f}")
print("="*60)

# %%
