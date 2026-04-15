#%%
"""
Progressive digit addition across multiple seeds
Compare how different random initializations handle incremental digit addition
"""

import sys
from pathlib import Path
import os

os.chdir(Path(__file__).parent)

# Add project root to path
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from kornia.augmentation import RandomGaussianNoise
from copy import deepcopy

from einops import einsum as einops_einsum
from functions.model import Model, Config
from functions.datasets import MNIST
from functions.tn_sim import get_interaction_matrix, tensor_similarity, model_similarity, covariance_similarity

def _cos_sim(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return (a @ b / (a.norm() * b.norm())).item()

def slice_similarity(model1, model2, digit, include_embedding=True, symmetrize=True):
    """Cosine similarity between the digit-th slice of each model's interaction matrix."""
    M1 = get_interaction_matrix(model1, include_embedding, symmetrize)
    M2 = get_interaction_matrix(model2, include_embedding, symmetrize)
    return _cos_sim(M1[digit], M2[digit])

def component_similarities(model1, model2):
    """Cosine similarity for embed, unembed, el, and er between two models."""
    e1, e2 = model1.embed.weight, model2.embed.weight
    u1, u2 = model1.w_u, model2.w_u
    l1, r1 = model1.w_lr[0].unbind()
    l2, r2 = model2.w_lr[0].unbind()

    el1 = einops_einsum(e1, l1, "e i, h e -> i h")
    el2 = einops_einsum(e2, l2, "e i, h e -> i h")
    er1 = einops_einsum(e1, r1, "e i, h e -> i h")
    er2 = einops_einsum(e2, r2, "e i, h e -> i h")

    result = {
        'embed':   _cos_sim(e1, e2),
        'unembed': _cos_sim(u1, u2),
        'el':      _cos_sim(el1, el2),
        'er':      _cos_sim(er1, er2),
    }
    for d in local_digits:
        result[f'u_row_{d}'] = _cos_sim(u1[d], u2[d])
    return result

local_digits = [0, 6, 9]
component_keys = ['embed', 'unembed', 'el', 'er']
unembed_row_keys = [f'u_row_{d}' for d in local_digits]

device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"


# Create figures directory
figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)
savefigBool = False
print(f"Saving figures to: {figures_dir.absolute()}")

#%% KEY VARIABLES

# Set global sizes
plt.rcParams.update({
    'font.size': 12,          # Standard text
    'axes.titlesize': 18,     # Titles
    'axes.labelsize': 18,     # X and Y labels
    'xtick.labelsize': 16,    # Axis tick numbers
    'ytick.labelsize': 16,
    'legend.fontsize': 18
})

ForgetBool = False  # If True, only train on newly added digit at each stage
N_SimSteps = 250 # Number of checkpoints to sample for the 500x500 heatmap (uniformly sampled across all stages)
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
# Save key checkpoints
SaveWeightsBool = False
if SaveWeightsBool == True:
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)

    checkpoint_stages = ['base', 'add 9', 'remove 9', 're-add 9']
    checkpoint_filenames = ['base', 'add9', 'remove9', 'readd9']

    for stage_name, filename in zip(checkpoint_stages, checkpoint_filenames):
        save_path = weights_dir / f"seed42_{filename}.pt"
        torch.save(progressive_models[42][stage_name].state_dict(), save_path)
        print(f"Saved {stage_name} -> {save_path}")

#%%
# Compute similarities: All seeds compared to seed 42's final model
print("\n=== Computing progressive similarity evolution ===")

reference_seed = 42
reference_model = progressive_models[reference_seed]['add 9']  # Seed 42's final model
# reference_model = base_model # Alternatively, use the base model

progressive_similarities = {
    'tensor': {seed: {} for seed in seeds},
    'local': {digit: {seed: {} for seed in seeds} for digit in local_digits},
    'components': {comp: {seed: {} for seed in seeds} for comp in component_keys},
    'unembed_rows': {key: {seed: {} for seed in seeds} for key in unembed_row_keys},
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
        local_sims = {digit: [] for digit in local_digits}
        comp_sims = {comp: [] for comp in component_keys}
        unembed_sims = {key: [] for key in unembed_row_keys}
        train_accs = []
        test_accs = []

        for cp in tqdm(progressive_checkpoints[seed][stage_name], desc=f"Seed {seed} {stage_name}"):
            model_temp = Model.from_config(**base_model_config).to(device)
            model_temp.load_state_dict(cp['state_dict'])
            model_temp.eval()

            # Global tensor similarity to reference
            tensor_sim = model_similarity(
                model_temp,
                reference_model,
                include_embedding=True,
                symmetrize=True
            )
            tensor_sims.append(tensor_sim)

            # Local (per-slice) similarity for digits 0, 6, 9
            for digit in local_digits:
                local_sims[digit].append(slice_similarity(model_temp, reference_model, digit))

            # Component similarities: embed, unembed, el, er
            comp = component_similarities(model_temp, reference_model)
            for key in component_keys:
                comp_sims[key].append(comp[key])

            for key in unembed_row_keys:
                unembed_sims[key].append(comp[key])            

            # Compute accuracies on this stage's data
            with torch.no_grad():
                train_loss, train_acc = model_temp.step(train_data_stage.x, train_data_stage.y)
                test_loss, test_acc = model_temp.step(test_data_stage.x, test_data_stage.y)

                train_accs.append(train_acc.item())
                test_accs.append(test_acc.item())

        progressive_similarities['tensor'][seed][stage_name] = tensor_sims
        for digit in local_digits:
            progressive_similarities['local'][digit][seed][stage_name] = local_sims[digit]
        for key in component_keys:
            progressive_similarities['components'][key][seed][stage_name] = comp_sims[key]
        for key in unembed_row_keys:
            progressive_similarities['unembed_rows'][key][seed][stage_name] = unembed_sims[key]
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
# PLOT SIMILARITY VS ACCURACY
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
# Plot: Global + local (digit 0, 6, 9) similarity vs reference
local_linestyles = {0: '--', 6: ':', 9: '-.'}

fig, ax = plt.subplots(figsize=(16, 8))

for seed in seeds:
    cumulative_batch = 0

    for stage_config in digit_curriculum:
        stage_name = stage_config['name']
        checkpoints = progressive_checkpoints[seed][stage_name]
        batch_steps = [cumulative_batch + cp['batch'] for cp in checkpoints]
        color = colors_progressive[stage_name]

        # Global similarity — solid
        ax.plot(batch_steps, progressive_similarities['tensor'][seed][stage_name],
                color=color, linewidth=2.5, linestyle='-',
                label=f'{stage_name} (global)' if seed == reference_seed else None)

        # Local similarities — dashed / dotted / dash-dot
        for digit in local_digits:
            ax.plot(batch_steps, progressive_similarities['local'][digit][seed][stage_name],
                    color=color, linewidth=1.5, linestyle=local_linestyles[digit], alpha=0.7,
                    label=f'{stage_name} (digit {digit})' if seed == reference_seed else None)

        cumulative_batch = batch_steps[-1]

ax.set_xlabel('Batch Steps')
ax.set_ylabel('Tensor Similarity')
ax.set_ylim([-0.1, 1.05])
ax.grid(True, alpha=0.3)

# Build a clean legend: stage colours + linestyle meaning
from matplotlib.lines import Line2D
stage_handles = [Line2D([0], [0], color=colors_progressive[s['name']], linewidth=2.5, label=s['name'])
                 for s in digit_curriculum]
style_handles = [
    Line2D([0], [0], color='gray', linewidth=2.5, linestyle='-',  label='global'),
    Line2D([0], [0], color='gray', linewidth=1.5, linestyle='--', label='digit 0'),
    Line2D([0], [0], color='gray', linewidth=1.5, linestyle=':',  label='digit 6'),
    Line2D([0], [0], color='gray', linewidth=1.5, linestyle='-.', label='digit 9'),
]
ax.legend(handles=stage_handles + style_handles, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
if savefigBool:
    plt.savefig(figures_dir / "progressive_addition_local_similarity.png", dpi=300, bbox_inches='tight')
plt.show()

#%%
# Plot: Component similarities (embed, unembed, el, er) vs reference
component_linestyles = {'embed': '--', 'unembed': ':', 'el': '-.', 'er': (0, (3, 1, 1, 1, 1, 1))}

fig, ax = plt.subplots(figsize=(16, 8))

for seed in seeds:
    cumulative_batch = 0

    for stage_config in digit_curriculum:
        stage_name = stage_config['name']
        checkpoints = progressive_checkpoints[seed][stage_name]
        batch_steps = [cumulative_batch + cp['batch'] for cp in checkpoints]
        color = colors_progressive[stage_name]

        # Global similarity — solid, thicker
        ax.plot(batch_steps, progressive_similarities['tensor'][seed][stage_name],
                color=color, linewidth=2.5, linestyle='-',
                label=f'{stage_name} (global)' if seed == reference_seed else None)

        # Component similarities
        for key in component_keys:
            ax.plot(batch_steps, progressive_similarities['components'][key][seed][stage_name],
                    color=color, linewidth=1.5, linestyle=component_linestyles[key], alpha=0.7,
                    label=f'{stage_name} ({key})' if seed == reference_seed else None)

        cumulative_batch = batch_steps[-1]

ax.set_xlabel('Batch Steps')
ax.set_ylabel('Cosine Similarity')
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)

stage_handles = [Line2D([0], [0], color=colors_progressive[s['name']], linewidth=2.5, label=s['name'])
                 for s in digit_curriculum]
style_handles = [Line2D([0], [0], color='gray', linewidth=2.5, linestyle='-', label='global')] + [
    Line2D([0], [0], color='gray', linewidth=1.5, linestyle=component_linestyles[k], label=k)
    for k in component_keys
]
ax.legend(handles=stage_handles + style_handles, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
if savefigBool:
    plt.savefig(figures_dir / "progressive_addition_component_similarity.png", dpi=300, bbox_inches='tight')
plt.show()

#%%
# Plot: Per-row unembedding similarity for digits 0, 6, 9
unembed_row_linestyles = {f'u_row_{d}': ls for d, ls in zip(local_digits, ['--', ':', '-.'])}

fig, ax = plt.subplots(figsize=(16, 8))

for seed in seeds:
    cumulative_batch = 0

    for stage_config in digit_curriculum:
        stage_name = stage_config['name']
        checkpoints = progressive_checkpoints[seed][stage_name]
        batch_steps = [cumulative_batch + cp['batch'] for cp in checkpoints]
        color = colors_progressive[stage_name]

        # Global similarity — solid
        ax.plot(batch_steps, progressive_similarities['tensor'][seed][stage_name],
                color=color, linewidth=2.5, linestyle='-',
                label=f'{stage_name} (global)' if seed == reference_seed else None)

        # Per-row unembed similarity for digits 0, 6, 9
        for d, key in zip(local_digits, unembed_row_keys):
            ax.plot(batch_steps, progressive_similarities['unembed_rows'][key][seed][stage_name],
                    color=color, linewidth=1.5, linestyle=unembed_row_linestyles[key], alpha=0.7,
                    label=f'{stage_name} (u row {d})' if seed == reference_seed else None)

        cumulative_batch = batch_steps[-1]

ax.set_xlabel('Batch Steps')
ax.set_ylabel('Cosine Similarity')
ax.set_ylim([-0.1, 1.05])
ax.grid(True, alpha=0.3)

stage_handles = [Line2D([0], [0], color=colors_progressive[s['name']], linewidth=2.5, label=s['name'])
                 for s in digit_curriculum]
style_handles = [Line2D([0], [0], color='gray', linewidth=2.5, linestyle='-', label='global')] + [
    Line2D([0], [0], color='gray', linewidth=1.5, linestyle=unembed_row_linestyles[k], label=f'u row {d}')
    for d, k in zip(local_digits, unembed_row_keys)
]
ax.legend(handles=stage_handles + style_handles, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
if savefigBool:
    plt.savefig(figures_dir / "progressive_addition_unembed_row_similarity.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
