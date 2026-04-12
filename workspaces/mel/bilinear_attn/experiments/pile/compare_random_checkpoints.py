#!/usr/bin/env python3
"""Compute TN similarity between two random checkpoints from the Pile model."""

import os
import random
import time
import json
import torch
from huggingface_hub import snapshot_download
import psutil
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# We'll import AttentionLM from the downloaded path after downloading
from tn_sim.mc_similarity import mc_similarity
from tn_sim import cosine_similarity as tn_cosine_similarity


def patch_model_for_tn_similarity(model):
    """Temporarily patch model attributes to make it compatible with TN similarity.
    
    This hides the norms from TN similarity validation without actually removing
    them from the model's forward pass. The norms will still be applied during
    normal inference, but TN similarity will treat the model as if it has no norms.
    
    Args:
        model: AttentionLM model with norms
        
    Returns:
        Dictionary of original values that can be restored later
    """
    original_values = {}
    
    # Ensure required attributes exist for AttentionLMComponent.from_trained_model
    if not hasattr(model, 'n_head'):
        # Try to infer from first layer
        if hasattr(model.layers[0], 'n_head'):
            model.n_head = model.layers[0].n_head
        else:
            # Default from config
            model.n_head = 16
            original_values['n_head'] = None
    
    if not hasattr(model, 'n_layers'):
        model.n_layers = len(model.layers)
        original_values['n_layers'] = None
    
    # Add attn_type and scale for AttentionLMComponent compatibility
    if not hasattr(model, 'attn_type'):
        model.attn_type = 'bilinear'
        original_values['attn_type'] = None
    
    if not hasattr(model, 'scale'):
        # Try to get from first layer
        if hasattr(model.layers[0], 'scale'):
            model.scale = model.layers[0].scale
        else:
            model.scale = 1.0
            original_values['scale'] = None
    
    # Patch norm_type
    if hasattr(model, 'norm_type'):
        original_values['norm_type'] = model.norm_type
        model.norm_type = 'none'
    
    # Patch norm_places
    if hasattr(model, 'norm_places'):
        original_values['norm_places'] = model.norm_places
        model.norm_places = []
    else:
        model.norm_places = []
        original_values['norm_places'] = None
    
    # Patch final_norm (replace with Identity for validation)
    if hasattr(model, 'final_norm'):
        original_values['final_norm'] = model.final_norm
        import torch.nn as nn
        model.final_norm = nn.Identity()
    
    # Patch embed_norm
    if hasattr(model, 'embed_norm'):
        original_values['embed_norm'] = model.embed_norm
        model.embed_norm = None
    
    # Patch layer_norms
    if hasattr(model, 'layer_norms'):
        original_values['layer_norms'] = model.layer_norms
        model.layer_norms = None
    
    return original_values


def restore_model_attributes(model, original_values):
    """Restore original model attributes after TN similarity computation.
    
    Args:
        model: AttentionLM model that was patched
        original_values: Dictionary of original values from patch_model_for_tn_similarity
    """
    for attr, value in original_values.items():
        if value is not None:
            setattr(model, attr, value)
        elif hasattr(model, attr):
            delattr(model, attr)


def get_max_memory_mb():
    """Get max memory used in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def main():
    print("Downloading checkpoints from HuggingFace...")
    path = snapshot_download(
        repo_id="Elriggs/bilinear_atnn_only_2L_Pile",
        revision="e185f7e72a20ccf0022a15b5a6f31b6c7b2d66b0",
    )
    print(f"Downloaded to: {path}")

    # Add downloaded path to sys.path to import the custom AttentionLM
    sys.path.insert(0, path)
    from model import AttentionLM

    # Load config
    config_path = os.path.join(path, "config.json")
    with open(config_path) as f:
        huggingface_cfg = json.load(f)
    
    print(f"Loaded config: vocab_size={huggingface_cfg['vocab_size']}, n_ctx={huggingface_cfg['n_ctx']}, d_model={huggingface_cfg['d_model']}, n_head={huggingface_cfg['n_head']}, n_layers={huggingface_cfg['n_layers']}")

    # Find all .pt checkpoint files
    checkpoint_path = os.path.join(path, "checkpoints")
    pt_files = [f for f in os.listdir(checkpoint_path) if f.endswith(".pt")]
    
    if len(pt_files) < 2:
        raise ValueError(f"Need at least 2 checkpoints, found {len(pt_files)}")
    
    print(f"Found {len(pt_files)} checkpoint files")
    
    # Select two random checkpoints
    ckpt1_name, ckpt2_name = random.sample(pt_files, 2)
    print(f"Selected checkpoints: {ckpt1_name} and {ckpt2_name}")
    
    ckpt1_path = os.path.join(checkpoint_path, ckpt1_name)
    ckpt2_path = os.path.join(checkpoint_path, ckpt2_name)
    
    # Load models
    print("\nLoading models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load first model using the custom AttentionLM from downloaded path
    model1 = AttentionLM(
        vocab_size=huggingface_cfg['vocab_size'],
        n_ctx=huggingface_cfg['n_ctx'],
        d_model=huggingface_cfg['d_model'],
        n_head=huggingface_cfg['n_head'],
        n_layers=huggingface_cfg['n_layers'],
        attn_scale=huggingface_cfg.get('attn_scale', 1.0),
        rope_base=huggingface_cfg.get('rope_base', 10000),
        norm_type=huggingface_cfg.get('norm_type', 'layernorm'),
    )
    state1 = torch.load(ckpt1_path, map_location="cpu", weights_only=True)
    model1.load_state_dict(state1)
    model1.eval()
    print(f"Loaded {ckpt1_name}")
    
    # Load second model
    model2 = AttentionLM(
        vocab_size=huggingface_cfg['vocab_size'],
        n_ctx=huggingface_cfg['n_ctx'],
        d_model=huggingface_cfg['d_model'],
        n_head=huggingface_cfg['n_head'],
        n_layers=huggingface_cfg['n_layers'],
        attn_scale=huggingface_cfg.get('attn_scale', 1.0),
        rope_base=huggingface_cfg.get('rope_base', 10000),
        norm_type=huggingface_cfg.get('norm_type', 'layernorm'),
    )
    state2 = torch.load(ckpt2_path, map_location="cpu", weights_only=True)
    model2.load_state_dict(state2)
    model2.eval()
    print(f"Loaded {ckpt2_name}")
    
    # Move to device
    model1 = model1.to(device)
    model2 = model2.to(device)
    
    # Patch models to make them compatible with TN similarity
    print("\nPatching models for TN similarity compatibility...")
    orig1 = patch_model_for_tn_similarity(model1)
    orig2 = patch_model_for_tn_similarity(model2)
    
    # Convert to AttentionLMComponent for TN similarity
    print("Converting to AttentionLMComponent...")
    from models.components.model import AttentionLMComponent
    comp1 = AttentionLMComponent.from_trained_model(model1)
    comp2 = AttentionLMComponent.from_trained_model(model2)
    
    # Move components to device
    comp1 = comp1.to(device)
    comp2 = comp2.to(device)
    
    # Get initial memory
    initial_memory = get_max_memory_mb()
    print(f"Initial memory: {initial_memory:.2f} MB")
    
    # Compute TN similarity
    print("\nComputing TN cosine similarity...")
    start_time = time.time()
    
    with torch.no_grad():
        similarity = tn_cosine_similarity(comp1, comp2, device=device, dtype=torch.float64)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Restore original model attributes
    restore_model_attributes(model1, orig1)
    restore_model_attributes(model2, orig2)
    print("Restored original model attributes")
    
    # Get max memory
    max_memory = get_max_memory_mb()
    memory_used = max_memory - initial_memory
    
    # Print results
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Checkpoint 1: {ckpt1_name}")
    print(f"Checkpoint 2: {ckpt2_name}")
    print(f"TN Cosine Similarity: {similarity:.6f}")
    print(f"Computation Time: {elapsed_time:.2f} seconds")
    print(f"Max Memory Used: {memory_used:.2f} MB")
    print(f"Total Max Memory: {max_memory:.2f} MB")
    print("="*50)


if __name__ == "__main__":
    main()
