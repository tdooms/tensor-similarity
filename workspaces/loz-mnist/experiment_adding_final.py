#%%
"""
Progressive digit addition across multiple seeds
Compare how different random initializations handle incremental digit addition
"""

import sys
from pathlib import Path
import os

os.chdir('/Users/wroe/Documents/AI/mnist-sim-clean')

# Add project root to path
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from kornia.augmentation import RandomGaussianNoise
from copy import deepcopy

from functions.model import Model, Config
from functions.datasets import MNIST
from functions.tn_sim import get_interaction_matrix, tensor_similarity, model_similarity, covariance_similarity

device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"

#%%
# Create figures directory
figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)
savefigBool = False
print(f"Saving figures to: {figures_dir.absolute()}")

ForgetBool = False  # If True, only train on newly added digit at each stage

# Set global sizes
plt.rcParams.update({
    'font.size': 12,          # Standard text
    'axes.titlesize': 18,     # Titles
    'axes.labelsize': 18,     # X and Y labels
    'xtick.labelsize': 16,    # Axis tick numbers
    'ytick.labelsize': 16,
    'legend.fontsize': 18
})


epoch_setup = 20 # 60
d_hidden_setup = 128
d_embed_setup = 256
batch_size_setup = 248 #248
record_every_n_batches_setup = 5 # 5 for 2048 batch size
LatentNoiseSetup = 0.0


# Define progressive addition curriculum
digit_curriculum = [
    {'name': 'base', 'digits': list(range(5)), 'epochs': 20, 'lr': 1e-3},      # {0,1,2,3,4}
    {'name': 'add 5', 'digits': list(range(6)), 'epochs': 20, 'lr': 1e-3},     # {0,1,2,3,4,5}
    {'name': 'add 6', 'digits': list(range(7)), 'epochs': 20, 'lr': 1e-3},     # {0,1,2,3,4,5,6}
    {'name': 'add 7', 'digits': list(range(8)), 'epochs': 20, 'lr': 1e-3},     # {0,1,2,3,4,5,6,7}
    {'name': 'add 8', 'digits': list(range(9)), 'epochs': 20, 'lr': 1e-3},     # {0,1,2,3,4,5,6,7,8}
    {'name': 'add 9', 'digits': list(range(10)), 'epochs': 20, 'lr': 1e-3},
    {'name': 'remove 9', 'digits': list(range(9)), 'epochs': 20, 'lr': 1e-3},    # {0,1,2,3,4,5,6,7,8}
    {'name': 're-add 9', 'digits': list(range(10)), 'epochs': 20, 'lr': 1e-3},    # {0,1,2,3,4,5,6,7,8,9}
    # {'name': 'remove_9_again', 'digits': list(range(9)), 'epochs': 20, 'lr': 1e-3},    # {0,1,2,3,4,5,6,7,8}
    # {'name': 're-add_9_again', 'digits': list(range(10)), 'epochs': 20, 'lr': 1e-3},    # {0,1,2,3,4,5,6,7,8,9}
    # {'name': 'remove_9_again2', 'digits': list(range(9)), 'epochs': 20, 'lr': 1e-3},    # {0,1,2,3,4,5,6,7,8}
    # {'name': 're-add_9_again2', 'digits': list(range(10)), 'epochs': 20, 'lr': 1e-3},    # {0,1,2,3,4,5,6,7,8,9}
    # {'name': 'remove_9_again3', 'digits': list(range(9)), 'epochs': 20, 'lr': 1e-3},    # {0,1,2,3,4,5,6,7,8}
    # {'name': 're-add_9_again3', 'digits': list(range(10)), 'epochs': 20, 'lr': 1e-3},    # {0,1,2,3,4,5,6,7,8,9}
    # {'name': 'remove_9_again4', 'digits': list(range(9)), 'epochs': 20, 'lr': 1e-3},    # {0,1,2,3,4,5,6,7,8}
    # {'name': 're-add_9_again4', 'digits': list(range(10)), 'epochs': 20, 'lr': 1e-3},    # {0,1,2,3,4,5,6,7,8,9}
    # {'name': 'remove_9_again5', 'digits': list(range(9)), 'epochs': 20, 'lr': 1e-3},    # {0,1,2,3,4,5,6,7,8}
    # {'name': 're-add_9_again5', 'digits': list(range(10)), 'epochs': 20, 'lr': 1e-3},    # {0,1,2,3,4,5,6,7,8,9}
    # {'name': 'remove_9_again6', 'digits': list(range(9)), 'epochs': 20, 'lr': 1e-3},    # {0,1,2,3,4,5,6,7,8}
    # {'name': 're-add_9_again6', 'digits': list(range(10)), 'epochs': 20, 'lr': 1e-3},    # {0,1,2,3,4,5,6,7,8,9}
    # {'name': 'remove_9_again7', 'digits': list(range(9)), 'epochs': 20, 'lr': 1e-3},    # {0,1,2,3,4,5,6,7,8}
    # {'name': 're-add_9_again7', 'digits': list(range(10)), 'epochs': 20, 'lr': 1e-3},    # {0,1,2,3,4,5,6,7,8,9}
    # {'name': 'remove_9_again8', 'digits': list(range(9)), 'epochs': 20, 'lr': 1e-3},    # {0,1,2,3,4,5,6,7,8}
    # {'name': 're-add_9_again8', 'digits': list(range(10)), 'epochs': 20, 'lr': 1e-3},    # {0,1,2,3,4,5,6,7,8,9}
]

colors_progressive = {
    'base': 'red',
    'add 5': 'blue',
    'add 6': 'green',
    'add 7': 'orange',
    'add 8': 'purple',
    'add 9': 'brown',
    'remove 9': 'cyan',
    're-add 9': 'magenta',
    # 'remove_9_again': 'gray',
    # 're-add_9_again': 'black',
    # 'remove_9_again2': 'pink',
    # 're-add_9_again2': 'olive',
    # 'remove_9_again3': 'teal',
    # 're-add_9_again3': 'navy',
    # 'remove_9_again4': 'maroon',
    # 're-add_9_again4': 'lime',
    # 'remove_9_again5': 'coral',
    # 're-add_9_again5': 'gold',
    # 'remove_9_again6': 'silver',
    # 're-add_9_again6': 'cyan',
    # 'remove_9_again7': 'magenta',
    # 're-add_9_again7': 'black',
    # 'remove_9_again8': 'gray',
    # 're-add_9_again8': 'orange',
}



#%% 
base_config = digit_curriculum[0]
print(f"\n=== Phase 1: Full model")

train_data = MNIST(train=True, download=True, device=device)
test_data = MNIST(train=False, download=True, device=device)

base_model_config = {
    'epochs': epoch_setup,
    'seed': 42,
    'd_hidden': d_hidden_setup,
    'd_embed': d_embed_setup,
    'batch_size': batch_size_setup,
}

base_model = Model.from_config(**base_model_config).to(device)

base_history, base_checkpoints = base_model.fit(
    train_data,
    test_data,
    RandomGaussianNoise(std=LatentNoiseSetup),
    record_every_n_batches=5,
    save_checkpoints=True,
)

#%%
# Train with progressive digit addition across multiple seeds
# seeds = [42, 123, 456]
seeds = [42]
progressive_models = {seed: {} for seed in seeds}
progressive_histories = {seed: {} for seed in seeds}
progressive_checkpoints = {seed: {} for seed in seeds}

for seed in seeds:
    print(f"\n{'='*60}")
    print(f"PROGRESSIVE TRAINING WITH SEED {seed}")
    print(f"{'='*60}")
    
    # Phase 1: Train base model
    base_config = digit_curriculum[0]
    print(f"\n=== Phase 1: Training {base_config['name']} with digits {base_config['digits']} ===")
    
    train_data = MNIST(train=True, download=True, device=device, digits=base_config['digits'])
    test_data = MNIST(train=False, download=True, device=device, digits=base_config['digits'])
    
    model_config = {
        'epochs': epoch_setup,
        'seed': seed,
        'd_hidden': d_hidden_setup,
        'd_embed': d_embed_setup,
        'batch_size': batch_size_setup,
    }
    
    model = Model.from_config(  
        **model_config
    ).to(device)
    
    history, checkpoints = model.fit(
        train_data,
        test_data,
        RandomGaussianNoise(LatentNoiseSetup),
        record_every_n_batches=5,
        save_checkpoints=True,
    )
    
    progressive_models[seed][base_config['name']] = model
    progressive_histories[seed][base_config['name']] = history
    progressive_checkpoints[seed][base_config['name']] = checkpoints
    
    print(f"Seed {seed}, {base_config['name']} - Final val acc: {history['val/acc'].iloc[-1]:.4f}")
    
    # Save the final state to use as initialization for next phases
    base_state = deepcopy(model.state_dict())
    
    # Phase 2: Progressive addition
    for stage in digit_curriculum[1:]:
        print(f"\n=== Phase 2: Training {stage['name']} with digits {stage['digits']} ===")
        print(f"    Starting from {base_config['name']} checkpoint")

        if ForgetBool == True:
            train_data = MNIST(train=True, download=True, device=device, digits=[stage['digits'][-1]])
            test_data = MNIST(train=False, download=True, device=device, digits=[stage['digits'][-1]])
        else:
            train_data = MNIST(train=True, download=True, device=device, digits=stage['digits'])
            test_data = MNIST(train=False, download=True, device=device, digits=stage['digits'])
        
        
        # Initialize with base model weights
        model = Model.from_config(**base_model_config).to(device).to(device)
        model.load_state_dict(base_state)  # Start from base checkpoint
        
        history, checkpoints = model.fit(
            train_data,
            test_data,
            RandomGaussianNoise(LatentNoiseSetup),
            record_every_n_batches=5,
            save_checkpoints=True,
        )
        
        progressive_models[seed][stage['name']] = model
        progressive_histories[seed][stage['name']] = history
        progressive_checkpoints[seed][stage['name']] = checkpoints
        
        print(f"Seed {seed}, {stage['name']} - Final val acc: {history['val/acc'].iloc[-1]:.4f}")
        
        # Update base_state for next stage
        base_state = deepcopy(model.state_dict())

#%%
# Compute similarities: All seeds compared to seed 42's final model
print("\n=== Computing progressive similarity evolution ===")

reference_seed = 42
reference_model = progressive_models[reference_seed]['add 9']  # Seed 42's final model
# reference_model = base_model # Alternatively, use the base model

progressive_similarities = {
    'tensor': {seed: {} for seed in seeds},
}

progressive_accuracies = {
    'train': {seed: {} for seed in seeds},
    'test': {seed: {} for seed in seeds},
}

for seed in seeds:
    print(f"\nProcessing seed {seed}...")
    
    for stage_config in digit_curriculum:
        stage_name = stage_config['name']
        stage_digits = stage_config['digits']
        print(f"  Stage: {stage_name}")
        
        # Load the appropriate datasets for this stage
        train_data_stage = MNIST(train=True, download=True, device=device, digits=stage_digits)
        test_data_stage = MNIST(train=False, download=True, device=device, digits=stage_digits)
        
        tensor_sims = []
        train_accs = []
        test_accs = []
        
        for cp in tqdm(progressive_checkpoints[seed][stage_name], desc=f"Seed {seed} {stage_name}"):
            model_temp = Model.from_config(**base_model_config).to(device)
            model_temp.load_state_dict(cp['state_dict'])
            model_temp.eval()
            
            # Tensor similarity to seed 42's final
            tensor_sim = model_similarity(
                model_temp,
                reference_model,
                include_embedding=True,
                symmetrize=True
            )
            tensor_sims.append(tensor_sim)
            
            # Compute accuracies on this stage's data
            with torch.no_grad():
                train_loss, train_acc = model_temp.step(train_data_stage.x, train_data_stage.y)
                test_loss, test_acc = model_temp.step(test_data_stage.x, test_data_stage.y)
                
                train_accs.append(train_acc.item())
                test_accs.append(test_acc.item())
            
        progressive_similarities['tensor'][seed][stage_name] = tensor_sims
        progressive_accuracies['train'][seed][stage_name] = train_accs
        progressive_accuracies['test'][seed][stage_name] = test_accs

#%%
# Plot: Progressive digit addition - Tensor similarity across seeds
fig, ax = plt.subplots(figsize=(16, 8))

for seed in seeds:
    cumulative_batch = 0
    
    for stage_config in digit_curriculum:
        stage_name = stage_config['name']
        checkpoints = progressive_checkpoints[seed][stage_name]
        
        # Create batch steps relative to cumulative count
        batch_steps = [cumulative_batch + cp['batch'] for cp in checkpoints]
        
        color = colors_progressive[stage_name]
        
        if seed == reference_seed:
            # Seed 42: solid lines
            ax.plot(batch_steps, progressive_similarities['tensor'][seed][stage_name],
                   label=f'{stage_name}',
                   linewidth=2.5, alpha=1.0, color=color, linestyle='-')
        else:
            # Other seeds: dashed lines
            ax.plot(batch_steps, progressive_similarities['tensor'][seed][stage_name],
                   label=f'{stage_name}',
                   linewidth=2, alpha=0.6, color=color, linestyle='--')
        
        # Mark phase transitions (only once)
        # if seed == seeds[0] and stage_name != 'base':
            # ax.axvline(x=cumulative_batch, color='red', linestyle=':', alpha=0.5, linewidth=1)
        
        cumulative_batch = batch_steps[-1]

ax.set_xlabel('Batch Steps')
ax.set_ylabel(f'Tensor Similarity')
# ax.set_title(f'Progressive Digit Addition: Tensor Similarity Across Seeds\n' +
            #  f'Color = stage | Solid = seed {reference_seed} | Dashed = other seeds', fontsize=13)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])
# ax.set_xlim([0, 6000])

plt.tight_layout()
if savefigBool:
    plt.savefig(figures_dir / "progressive_addition_tensor_similarity_all_seeds.png", dpi=300, bbox_inches='tight')
plt.show()



# 
#%%
# Plot: Progressive digit addition - Tensor similarity and accuracy across seeds
fig, ax1 = plt.subplots(figsize=(16, 8))

# Primary y-axis: Tensor Similarity (left)
ax1.set_xlabel('Batch Steps')
ax1.set_ylabel('Tensor Similarity', color='black')

for seed in seeds:
    cumulative_batch = 0
    
    for stage_config in digit_curriculum:
        stage_name = stage_config['name']
        checkpoints = progressive_checkpoints[seed][stage_name]
        
        # Create batch steps relative to cumulative count
        batch_steps = [cumulative_batch + cp['batch'] for cp in checkpoints]
        
        color = colors_progressive[stage_name]
        
        if seed == reference_seed:
            # Tensor similarity: solid lines
            ax1.plot(batch_steps, progressive_similarities['tensor'][seed][stage_name],
                   label=f'{stage_name}',
                   linewidth=2.5, alpha=1.0, color=color, linestyle='-')
        else:
            # Other seeds: dashed lines
            ax1.plot(batch_steps, progressive_similarities['tensor'][seed][stage_name],
                   label=f'{stage_name}',
                   linewidth=2, alpha=0.6, color=color, linestyle='--')
        
        cumulative_batch = batch_steps[-1]

ax1.tick_params(axis='y', labelcolor='black')
ax1.set_ylim([0, 1.05])
ax1.grid(True, alpha=0.3)

# Secondary y-axis: Accuracy (right)
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', color='gray')

for seed in seeds:
    cumulative_batch = 0
    
    for stage_config in digit_curriculum:
        stage_name = stage_config['name']
        checkpoints = progressive_checkpoints[seed][stage_name]
        
        # Create batch steps relative to cumulative count
        batch_steps = [cumulative_batch + cp['batch'] for cp in checkpoints]
        
        color = colors_progressive[stage_name]
        
        if seed == reference_seed:
            # Test accuracy: dashed lines (thinner, more transparent)
            ax2.plot(batch_steps, progressive_accuracies['test'][seed][stage_name],
                   linewidth=2.0, alpha=0.5, color=color, linestyle='--')
        else:
            # Other seeds: dashed lines
            ax2.plot(batch_steps, progressive_accuracies['test'][seed][stage_name],
                   linewidth=1.5, alpha=0.4, color=color, linestyle='--')
        
        cumulative_batch = batch_steps[-1]

ax2.tick_params(axis='y', labelcolor='gray')
ax2.set_ylim([0.9, 1])

# Legend only from ax1 (tensor similarity)
ax1.legend(bbox_to_anchor=(1.15, 1), loc='upper left')

plt.tight_layout()
if savefigBool:
    plt.savefig(figures_dir / "progressive_addition_tensor_similarity_and_accuracy.png", dpi=300, bbox_inches='tight')
plt.show()



#%%
# Create heatmap of tensor similarities at key checkpoints across stages
print("\n=== Computing similarity heatmap at key checkpoints ===")

reference_seed = 42

# Select checkpoints: start (0), middle (50%), end (100%) of each stage
checkpoint_labels = []
checkpoint_models = []

for stage_config in digit_curriculum:
    stage_name = stage_config['name']
    checkpoints = progressive_checkpoints[reference_seed][stage_name]
    
    n_checkpoints = len(checkpoints)
    
    # Start, middle, end indices
    start_idx = 0
    mid_idx = n_checkpoints // 2
    end_idx = n_checkpoints - 1
    
    for label_suffix, idx in [('start', start_idx), ('mid', mid_idx), ('end', end_idx)]:
        checkpoint_labels.append(f"{stage_name} ({label_suffix})")
        
        # Load model at this checkpoint
        model_temp = Model.from_config(**base_model_config).to(device)
        model_temp.load_state_dict(checkpoints[idx]['state_dict'])
        checkpoint_models.append(model_temp)

# Compute pairwise similarity matrix
n_checkpoints = len(checkpoint_models)
similarity_matrix = np.zeros((n_checkpoints, n_checkpoints))

print(f"Computing {n_checkpoints}x{n_checkpoints} similarity matrix...")
for i in tqdm(range(n_checkpoints), desc="Computing similarities"):
    for j in range(n_checkpoints):
        sim = model_similarity(
            checkpoint_models[i],
            checkpoint_models[j],
            include_embedding=True,
            symmetrize=True
        )
        similarity_matrix[i, j] = sim

#%%
# Plot heatmap
fig, ax = plt.subplots(figsize=(16, 14))

im = ax.imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)

# Set ticks and labels
ax.set_xticks(range(n_checkpoints))
ax.set_yticks(range(n_checkpoints))
ax.set_xticklabels(checkpoint_labels, rotation=90, ha='right')
ax.set_yticklabels(checkpoint_labels)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, label='Tensor Similarity', fraction=0.046, pad=0.04)

# Add text annotations (optional - can be commented out if too cluttered)
for i in range(n_checkpoints):
    for j in range(n_checkpoints):
        text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                      ha="center", va="center", 
                      color="black" if similarity_matrix[i, j] > 0.5 else "white",
                      fontsize=10)

# Add gridlines to separate stages
stage_boundaries = []
cumulative = 0
for stage_config in digit_curriculum:
    cumulative += 3  # 3 checkpoints per stage (start, mid, end)
    stage_boundaries.append(cumulative - 0.5)

for boundary in stage_boundaries[:-1]:  # Don't draw line after last stage
    ax.axhline(y=boundary, color='black', linewidth=2, alpha=0.7)
    ax.axvline(x=boundary, color='black', linewidth=2, alpha=0.7)

ax.set_xlabel('Checkpoint')
ax.set_ylabel('Checkpoint')
# ax.set_title('Pairwise Tensor Similarity: Progressive Training Checkpoints', fontsize=14)

plt.tight_layout()
if savefigBool:
    plt.savefig(figures_dir / "progressive_similarity_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()



#%%
# Create heatmap of tensor similarities at key checkpoints across stages
print("\n=== Computing similarity heatmap at key checkpoints ===")

reference_seed = 42

# Select checkpoints: base start, then middle and end of each stage
checkpoint_labels = []
checkpoint_models = []

for stage_idx, stage_config in enumerate(digit_curriculum):
    stage_name = stage_config['name']
    checkpoints = progressive_checkpoints[reference_seed][stage_name]
    
    n_checkpoints = len(checkpoints)
    
    if stage_idx == 0:
        # First stage (base): include start
        start_idx = 0
        checkpoint_labels.append(f"{stage_name} (start)")
        model_temp = Model.from_config(**base_model_config).to(device)
        model_temp.load_state_dict(checkpoints[start_idx]['state_dict'])
        checkpoint_models.append(model_temp)
    
    # All stages: include middle and end
    mid_idx = n_checkpoints // 2
    end_idx = n_checkpoints - 1
    
    for label_suffix, idx in [('mid', mid_idx), ('end', end_idx)]:
        checkpoint_labels.append(f"{stage_name} ({label_suffix})")
        
        # Load model at this checkpoint
        model_temp = Model.from_config(**base_model_config).to(device)
        model_temp.load_state_dict(checkpoints[idx]['state_dict'])
        checkpoint_models.append(model_temp)

# Compute pairwise similarity matrix
n_checkpoints = len(checkpoint_models)
similarity_matrix = np.zeros((n_checkpoints, n_checkpoints))

print(f"Computing {n_checkpoints}x{n_checkpoints} similarity matrix...")
for i in tqdm(range(n_checkpoints), desc="Computing similarities"):
    for j in range(n_checkpoints):
        sim = model_similarity(
            checkpoint_models[i],
            checkpoint_models[j],
            include_embedding=True,
            symmetrize=True
        )
        similarity_matrix[i, j] = sim

#%%
# Plot heatmap (lower triangle only, excluding diagonal)
fig, ax = plt.subplots(figsize=(16, 14))

# Mask upper triangle and diagonal
similarity_matrix_masked = np.copy(similarity_matrix)
for i in range(n_checkpoints):
    for j in range(n_checkpoints):
        if j >= i:  # Upper triangle and diagonal
            similarity_matrix_masked[i, j] = np.nan

im = ax.imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)

# Set ticks and labels
ax.set_xticks(range(n_checkpoints))
ax.set_yticks(range(n_checkpoints))
ax.set_xticklabels(checkpoint_labels, rotation=90, ha='right')
ax.set_yticklabels(checkpoint_labels)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, label='Tensor Similarity', fraction=0.046, pad=0.04)

# Add text annotations (only for lower triangle)
for i in range(n_checkpoints):
    for j in range(n_checkpoints):
        # if j < i:  # Only lower triangle
        text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                        ha="center", va="center", 
                        color="black" if similarity_matrix[i, j] > 0.3 else "white",
                        fontsize=14)

# Add gridlines to separate stages
# First stage has 3 checkpoints (start, mid, end), rest have 2 (mid, end)
stage_boundaries = [2.5]  # After base stage (3 checkpoints)
cumulative = 3
for stage_idx in range(1, len(digit_curriculum)):
    cumulative += 2  # 2 checkpoints per subsequent stage (mid, end)
    if stage_idx < len(digit_curriculum) - 1:
        stage_boundaries.append(cumulative - 0.5)

for boundary in stage_boundaries:
    ax.axhline(y=boundary, color='black', linewidth=2, alpha=0.7)
    ax.axvline(x=boundary, color='black', linewidth=2, alpha=0.7)

ax.set_xlabel('Checkpoint')
ax.set_ylabel('Checkpoint')

plt.tight_layout()
if savefigBool:
    plt.savefig(figures_dir / "progressive_similarity_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()


# %%
