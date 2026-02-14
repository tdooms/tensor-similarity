#%%
"""
Compare models trained with progressively fewer digits across different seeds.
Track how different random initializations affect convergence.
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

from functions.model import Model, Config
from functions.datasets import MNIST
from functions.tn_sim import get_interaction_matrix, tensor_similarity, model_similarity, covariance_similarity

device = "cpu"

#%%
# Create figures directory
figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)
savefigBool = False
print(f"Saving figures to: {figures_dir.absolute()}")

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

legendBool = False

#%%
# Create progressive digit dropping datasets
digit_configs = {
    'all': list(range(10)),           # [0,1,2,3,4,5,6,7,8,9]
    # 'drop_9': list(range(9)),         # [0,1,2,3,4,5,6,7,8]
    # 'drop_9_8': list(range(8)),       # [0,1,2,3,4,5,6,7]
    # 'drop_9_8_7': list(range(7)),     # [0,1,2,3,4,5,6]
    # 'drop_9_8_7_6': list(range(6)),   # [0,1,2,3,4,5]
    # 'drop_9_8_7_6_5': list(range(5)), # [0,1,2,3,4]
    # 'drop_9_8_7_6_5_4': list(range(4)), # [0,1,2,3]
    # 'drop_9_8_7_6_5_4_3': list(range(3)), # [0,1,2]
    'drop_9_8_7_6_5_4_3_2': list(range(2)), # [0,1]
    # 'drop_9_8_7_6_5_4_3_2_1': list(range(1)), # [0]
}

# Define color map for each config
config_colors = {
    'all': 'red',
    'drop_9': 'blue',
    'drop_9_8': 'green',
    'drop_9_8_7': 'orange',
    'drop_9_8_7_6': 'purple',
    'drop_9_8_7_6_5': 'brown',
    'drop_9_8_7_6_5_4': 'pink',
    'drop_9_8_7_6_5_4_3': 'gray',
    'drop_9_8_7_6_5_4_3_2': 'olive',
    'drop_9_8_7_6_5_4_3_2_1': 'teal',
}



# Load datasets for each configuration
datasets = {}
for name, digits in digit_configs.items():
    datasets[name] = {
        'train': MNIST(train=True, download=True, device=device, digits=digits),
        'test': MNIST(train=False, download=True, device=device, digits=digits),
        'digits': digits
    }
    print(f"{name}: Training with digits {digits}")

config_names = list(digit_configs.keys())
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 42]
# seeds = [1, 42]
# seeds = [42]

# Store models by seed and config
models = {seed: {} for seed in seeds}
histories = {seed: {} for seed in seeds}
all_checkpoints = {seed: {} for seed in seeds}

#%%
# Train models with checkpoints for each seed
for seed in seeds:
    print(f"\n{'='*60}")
    print(f"TRAINING WITH SEED {seed}")
    print(f"{'='*60}")
    
    for name in config_names:
        print(f"\n=== Training model with config: {name}, seed: {seed} ===")
        print(f"    Digits: {datasets[name]['digits']}")

        model_config = {
            'epochs': epoch_setup,
            'seed': seed,
            'd_hidden': d_hidden_setup,
            'd_embed': d_embed_setup,
            'batch_size': batch_size_setup,
        }
        
        model = Model.from_config(**model_config).to(device)
        
        # Train with checkpoints every 5 batches
        history, checkpoints = model.fit(
            datasets[name]['train'], 
            datasets[name]['test'], 
            RandomGaussianNoise(std=0.4), 
            record_every_n_batches=record_every_n_batches_setup, 
            save_checkpoints=True,
        )
        
        models[seed][name] = model
        histories[seed][name] = history
        all_checkpoints[seed][name] = checkpoints
        
        print(f"Seed {seed}, {name} - Final val acc: {history['val/acc'].iloc[-1]:.4f}")
        print(f"Seed {seed}, {name} - Number of checkpoints: {len(checkpoints)}")

#%%
# Store final models
print("\n=== Final models ready ===")
final_models = {seed: {} for seed in seeds}
for seed in seeds:
    for name in config_names:
        final_models[seed][name] = models[seed][name]
        print(f"Seed {seed}, {name} - Final model stored")

#%%
# Compute similarities: seed 42 checkpoints vs all seeds' finals
print("\n=== Computing cross-seed similarities ===")

# We'll compare seed 42's training trajectory to final models from all seeds
reference_seed = 42

similarity_evolution_cross_seed = {
    'tensor': {},
}

for target_seed in seeds:
    similarity_evolution_cross_seed['tensor'][target_seed] = {}
    
    print(f"\nComparing seed {reference_seed} checkpoints to seed {target_seed} finals...")
    
    for name in config_names:
        print(f"  Config: {name}")
        
        tensor_sims = []
        
        # Loop through seed 42's checkpoints
        for cp in tqdm(all_checkpoints[reference_seed][name], desc=f"Seed {reference_seed} {name} → Seed {target_seed}"):
            # Load checkpoint from seed 42
            model_config = {
                'epochs': epoch_setup,
                'seed': seed,
                'd_hidden': d_hidden_setup,
                'd_embed': d_embed_setup,
                'batch_size': batch_size_setup,
            }
            model_temp = Model.from_config(**model_config).to(device)
            model_temp.load_state_dict(cp['state_dict'])
            
            # Compare to target seed's final model
            tensor_sim = model_similarity(
                model_temp, 
                final_models[target_seed][name], 
                include_embedding=True, 
                symmetrize=True
            )
            tensor_sims.append(tensor_sim)
            
        
        similarity_evolution_cross_seed['tensor'][target_seed][name] = tensor_sims


#%%
# Plot 5 (modified): Convergence to full model across seeds
fig, ax = plt.subplots(figsize=(12, 8))

# Left: Tensor similarity
for target_seed in seeds:
    for name in config_names:
        if name == 'all':
            continue
        color = config_colors[name]
        batch_steps = [cp['batch'] for cp in all_checkpoints[reference_seed][name]]
        
        if target_seed == reference_seed:
            ax.plot(batch_steps, similarity_evolution_cross_seed['tensor'][target_seed][name],
                        label=f'{name} → self', linewidth=2.5, alpha=0.9,color=color)
        else:
            # Only show one other seed to avoid clutter
            # if target_seed == seeds[1]:  # Just show seed 123
            ax.plot(batch_steps, similarity_evolution_cross_seed['tensor'][target_seed][name],
                            label=f'{name} → seed {target_seed}', linewidth=2.5, alpha=0.9, color = color,linestyle='--')

ax.set_xlabel('Batch Steps')
ax.set_ylabel('Tensor Similarity ')
# ax.set_title(f'Tensor Similarity: Seed {reference_seed} Convergence\n(Self vs Seed {seeds[1]})')

ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])
if legendBool:
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)


plt.tight_layout()
if savefigBool:
    plt.savefig(figures_dir / "cross_seed_convergence_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

print("\nCross-seed analysis complete!")

#%%
# First, compute similarities to seed 42's "all" model specifically
print("\n=== Computing similarities to seed 42 'all' model ===")

reference_seed = 42
reference_model = final_models[reference_seed]['all']

similarity_to_all_42 = {
    'tensor': {}
}

for seed in seeds:
    similarity_to_all_42['tensor'][seed] = {}
    
    print(f"\nComparing seed {seed} checkpoints to seed {reference_seed} 'all' final...")
    
    for name in config_names:
        if name == 'all' and seed == reference_seed:
            # Skip comparing seed 42 'all' to itself - we already have this
            continue
            
        print(f"  Config: {name}")
        
        tensor_sims = []
        
        for cp in tqdm(all_checkpoints[seed][name], desc=f"Seed {seed} {name} → Seed 42 'all'"):
            # Load checkpoint
            model_temp = Model.from_config(epochs=20, d_hidden=128).to(device)
            model_temp.load_state_dict(cp['state_dict'])
            
            # Compare to seed 42's 'all' final model
            tensor_sim = model_similarity(
                model_temp, 
                reference_model, 
                include_embedding=True, 
                symmetrize=True
            )
            tensor_sims.append(tensor_sim)
        
        similarity_to_all_42['tensor'][seed][name] = tensor_sims
        

# Also get seed 42 'all' to itself for reference
similarity_to_all_42['tensor'][reference_seed]['all'] = similarity_evolution_cross_seed['tensor'][reference_seed]['all']


#%%
fig, ax = plt.subplots(figsize=(12, 8))

for seed in seeds:
    for name in config_names:
        batch_steps = [cp['batch'] for cp in all_checkpoints[seed][name]]
        color = config_colors[name]
        
        if seed == reference_seed and name == 'all':
            ax.plot(batch_steps, similarity_to_all_42['tensor'][seed][name],
                   label=f'{name} (seed {seed})', linewidth=3.5, alpha=1.0, 
                   linestyle='-', color=color)
        elif seed == reference_seed:
            ax.plot(batch_steps, similarity_to_all_42['tensor'][seed][name],
                   label=f'{name} (seed {seed})', linewidth=2.5, alpha=0.9, 
                   linestyle='-', color=color)
        else:
            ax.plot(batch_steps, similarity_to_all_42['tensor'][seed][name],
                   label=f'{name} (seed {seed})', linewidth=1.8, alpha=0.6, 
                   linestyle='--', color=color)

ax.set_xlabel('Batch Steps')
# ax.set_ylabel(f'Tensor Similarity to Seed {reference_seed} "All" Model')
ax.set_ylabel(f'Tensor Similarity')
# ax.set_title(f'Tensor Similarity: All Models Converging to Seed {reference_seed} "All"\n' + 
            #  'Color = digit config | Solid = seed 42 | Dashed = other seeds', fontsize=13)
if legendBool:
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

plt.tight_layout()
if savefigBool:
    plt.savefig(figures_dir / "tensor_similarity_to_seed42_all_single.png", dpi=300, bbox_inches='tight')
plt.show()

print("\nColor-coded analysis complete!")
print("\nColor mapping:")
for name, color in config_colors.items():
    print(f"  {name}: {color}")
print("\nLine styles:")
print(f"  Solid: seed {reference_seed}")
print(f"  Dashed: other seeds")
print("\nAnalysis complete!")


#%%
# Extract accuracies from training history (already computed!)
print("\n=== Extracting accuracies from training history ===")

checkpoint_accuracies = {seed: {} for seed in seeds}

for seed in seeds:
    for name in config_names:
        history = histories[seed][name]
        
        checkpoint_accuracies[seed][name] = {
            'train': history['train/acc'].values,
            'test': history['val/acc'].values,  # 'val' in history is test set
            'batch': history['batch'].values,
        }

print("Accuracy extraction complete!")


#%% 
# Plot accuracies: Train and Test
fig, ax = plt.subplots(figsize=(12, 8))


for seed in seeds:
    for name in config_names:
        batch_steps = checkpoint_accuracies[seed][name]['batch']
        color = config_colors[name]
        
        if seed == reference_seed:
            ax.plot(batch_steps, checkpoint_accuracies[seed][name]['train'],
                    label=f'{name} (seed {seed})', linewidth=2.5, alpha=0.9, 
                    linestyle='-', color=color)
            ax.plot(batch_steps, checkpoint_accuracies[seed][name]['test'],
                    label=f'{name} (seed {seed})',linewidth=2.5, alpha=0.9, 
                    linestyle='-.', color=color)
        # else:
            # ax1.plot(batch_steps, checkpoint_accuracies[seed][name]['train'],
                    # linewidth=1.8, alpha=0.6, linestyle='--', color=color)

ax.set_xlabel('Batch Steps')
ax.set_ylabel('Train Accuracy')
ax.set_title('Training and Test Accuracy Evolution')
if legendBool:
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])


plt.tight_layout()
if savefigBool:
    plt.savefig(figures_dir / "train_test_accuracy_evolution.png", dpi=300, bbox_inches='tight')
plt.show()


#%%
# Combined plot: Tensor similarity + Train and Test Accuracy (Single seed, 'all' config only)
fig, ax1 = plt.subplots(figsize=(12, 8))

reference_seed = 42
name = 'all'

# Primary y-axis: Tensor Similarity (RED)
ax1.set_xlabel('Batch Steps')
ax1.set_ylabel('Tensor Similarity', color='red')

batch_steps_sim = [cp['batch'] for cp in all_checkpoints[reference_seed][name]]
line1 = ax1.plot(batch_steps_sim, similarity_to_all_42['tensor'][reference_seed][name],
        linewidth=3.5, alpha=1.0, linestyle='-', color='red', 
        label='Tensor Similarity')[0]

ax1.tick_params(axis='y', labelcolor='red')
ax1.set_ylim([0, 1.05])
ax1.grid(True, alpha=0.3)
# ax1.set_xlim([0, 500])

# Secondary y-axis: Accuracy (BLUE)
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', color='blue')

batch_steps_acc = checkpoint_accuracies[reference_seed][name]['batch']
line2 = ax2.plot(batch_steps_acc, checkpoint_accuracies[reference_seed][name]['train'],
        linewidth=2.5, alpha=0.9, linestyle='-.', color='blue',
        label='Train Accuracy')[0]
line3 = ax2.plot(batch_steps_acc, checkpoint_accuracies[reference_seed][name]['test'],
        linewidth=2.5, alpha=0.9, linestyle='--', color='blue',
        label='Test Accuracy')[0]

ax2.tick_params(axis='y', labelcolor='blue')
ax2.set_ylim([0, 1.05])


# Combine legends
# if legendBool:
ax1.legend(handles=[line1, line2, line3], 
            labels=['Tensor Similarity', 'Train', 'Test'],
             loc='lower right')

# plt.title(f'Tensor Similarity and Accuracy Evolution (Seed {reference_seed}, All Digits)')
plt.tight_layout()
if savefigBool:
    plt.savefig(figures_dir / "combined_tn_sim_and_accuracy_single.png", dpi=300, bbox_inches='tight')
plt.show()


#%%
# Combined plot: Tensor similarity + Train and Test Accuracy (Single seed, 'all' config only)
fig, ax1 = plt.subplots(figsize=(12, 8))

reference_seed = 42
name = 'all'

# Primary y-axis: Tensor Similarity (RED)
ax1.set_xlabel('Batch Steps')
ax1.set_ylabel('Tensor Similarity', color='red')

batch_steps_sim = [cp['batch'] for cp in all_checkpoints[reference_seed][name]]
line1 = ax1.plot(batch_steps_sim, similarity_to_all_42['tensor'][reference_seed][name],
        linewidth=3.5, alpha=1.0, linestyle='-', color='red', 
        label='Tensor Similarity')[0]

ax1.tick_params(axis='y', labelcolor='red')
ax1.set_ylim([0, 1.05])
ax1.grid(True, alpha=0.3)
# ax1.set_xlim([0, 500])

# Secondary y-axis: Accuracy (BLUE)
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', color='blue')

batch_steps_acc = checkpoint_accuracies[reference_seed][name]['batch']
line2 = ax2.plot(batch_steps_acc, checkpoint_accuracies[reference_seed][name]['train'],
        linewidth=2.5, alpha=0.9, linestyle='-.', color='blue',
        label='Train Accuracy')[0]
line3 = ax2.plot(batch_steps_acc, checkpoint_accuracies[reference_seed][name]['test'],
        linewidth=2.5, alpha=0.9, linestyle='--', color='blue',
        label='Test Accuracy')[0]

ax2.tick_params(axis='y', labelcolor='blue')
ax2.set_ylim([0, 1.05])


# Combine legends
# if legendBool:
ax1.legend(handles=[line1, line2, line3], 
            labels=['Tensor Similarity', 'Train', 'Test'],
             loc='lower right')

# plt.title(f'Tensor Similarity and Accuracy Evolution (Seed {reference_seed}, All Digits)')
plt.tight_layout()
if savefigBool:
    plt.savefig(figures_dir / "combined_tn_sim_and_accuracy_single.png", dpi=300, bbox_inches='tight')
plt.show()

#%%
# Combined plot: Tensor similarity for all seeds + Train and Test Accuracy for seed 42
fig, ax1 = plt.subplots(figsize=(12, 8))

reference_seed = 42
name = 'all'

# Primary y-axis: Tensor Similarity (RED shades)
ax1.set_xlabel('Batch Steps')
ax1.set_ylabel('Tensor Similarity', color='red')

# Plot tensor similarity for all seeds
legend_lines = []
legend_labels = []

# Track if we've already added same-seed and cross-seed to legend
same_seed_added = False
cross_seed_added = False

for seed in seeds:
    batch_steps_sim = [cp['batch'] for cp in all_checkpoints[seed][name]]
    
    if seed == reference_seed:
        # Reference seed: solid red, thicker
        line = ax1.plot(batch_steps_sim, similarity_to_all_42['tensor'][seed][name],
                linewidth=3.5, alpha=1.0, linestyle='-', color='red')[0]
        if not same_seed_added:
            legend_lines.append(line)
            legend_labels.append('Tensor Similarity (Same Seed)')
            same_seed_added = True
    else:
        # Other seeds: dashed red, thinner
        line = ax1.plot(batch_steps_sim, similarity_to_all_42['tensor'][seed][name],
                linewidth=2.5, alpha=0.7, linestyle='--', color='red')[0]
        if not cross_seed_added:
            legend_lines.append(line)
            legend_labels.append('Tensor Similarity (Cross Seed)')
            cross_seed_added = True

ax1.tick_params(axis='y', labelcolor='red')
ax1.set_ylim([0, 1.05])
ax1.grid(True, alpha=0.3)

# Secondary y-axis: Accuracy (BLUE) - only for seed 42
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', color='blue')

batch_steps_acc = checkpoint_accuracies[reference_seed][name]['batch']
line_train = ax2.plot(batch_steps_acc, checkpoint_accuracies[reference_seed][name]['train'],
        linewidth=2.5, alpha=0.9, linestyle='-.', color='blue')[0]
line_test = ax2.plot(batch_steps_acc, checkpoint_accuracies[reference_seed][name]['test'],
        linewidth=2.5, alpha=0.9, linestyle='--', color='blue')[0]

ax2.tick_params(axis='y', labelcolor='blue')
ax2.set_ylim([0, 1.05])

# Combine legends
legend_lines.extend([line_train, line_test])
legend_labels.extend(['Train', 'Test'])

ax1.legend(handles=legend_lines, labels=legend_labels, loc='lower right')

plt.tight_layout()
if savefigBool:
    plt.savefig(figures_dir / "combined_tn_sim_all_seeds_and_accuracy.png", dpi=300, bbox_inches='tight')
plt.show()
# %%
