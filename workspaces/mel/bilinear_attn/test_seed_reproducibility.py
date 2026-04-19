"""Test seed reproducibility for the induction task setup.

This script tests whether setting the same seed produces identical results
across multiple runs.
"""

import torch
import yaml
from pathlib import Path
import numpy as np

# Test 1: Basic PyTorch seed reproducibility
def test_basic_seed():
    """Test if torch.manual_seed produces reproducible results."""
    print("=" * 60)
    print("Test 1: Basic PyTorch Seed Reproducibility")
    print("=" * 60)
    
    results = []
    for run in range(3):
        torch.manual_seed(42)
        x = torch.randn(10, 10)
        results.append(x.clone())
    
    # Check if all results are identical
    all_same = all(torch.allclose(results[0], r) for r in results[1:])
    print(f"Basic seed test: {'✓ PASS' if all_same else '✗ FAIL'}")
    if not all_same:
        print("  Different results across runs with same seed!")
        for i, r in enumerate(results):
            print(f"  Run {i}: sum={r.sum().item():.6f}")
    return all_same


# Test 2: Model initialization reproducibility
def test_model_init():
    """Test if model initialization is reproducible."""
    print("\n" + "=" * 60)
    print("Test 2: Model Initialization Reproducibility")
    print("=" * 60)
    
    from models import AttentionLM
    
    config = {
        "model": {
            "vocab_size": 256,
            "n_ctx": 16,
            "d_model": 16,
            "n_head": 4,
            "n_layers": 2,
            "attn_type": "bilinear",
            "attn_scale": 0.35,
            "rope_base": 10000,
            "norm_type": "none",
            "norm_places": [],
            "use_rmsnorm_qk": False,
            "use_bias_qk": True,
        },
        "init": {
            "std_embed": 0.02,
            "std_qkv": 0.02,
            "std_o": 0.01,
        }
    }
    
    models = []
    for run in range(3):
        torch.manual_seed(42)
        model = AttentionLM.from_config(config)
        models.append(model)
    
    # Compare first layer weights
    all_same = True
    for i in range(1, len(models)):
        for name, param in models[0].named_parameters():
            param_i = dict(models[i].named_parameters())[name]
            if not torch.allclose(param, param_i):
                all_same = False
                print(f"  ✗ Parameter {name} differs between run 0 and run {i}")
                print(f"    Run 0 sum: {param.sum().item():.6f}")
                print(f"    Run {i} sum: {param_i.sum().item():.6f}")
                break
        if not all_same:
            break
    
    print(f"Model init test: {'✓ PASS' if all_same else '✗ FAIL'}")
    return all_same


# Test 3: Data generation reproducibility
def test_data_generation():
    """Test if data generation is reproducible."""
    print("\n" + "=" * 60)
    print("Test 3: Data Generation Reproducibility")
    print("=" * 60)
    
    from experiments.induction_heads.data import create_repeated_token_dataloaders
    
    dataloaders = []
    for run in range(3):
        train_dl, val_dl = create_repeated_token_dataloaders(
            vocab_size=256,
            seq_len=8,
            batch_size=64,
            n_train=1000,
            n_val=100,
            seed=42,
        )
        # Get first batch
        batch = next(iter(train_dl))
        dataloaders.append(batch["input_ids"].clone())
    
    all_same = all(torch.equal(dataloaders[0], d) for d in dataloaders[1:])
    print(f"Data generation test: {'✓ PASS' if all_same else '✗ FAIL'}")
    if not all_same:
        print("  Different data across runs with same seed!")
        for i, d in enumerate(dataloaders):
            print(f"  Run {i}: first token={d[0, 0].item()}, sum={d.sum().item()}")
    return all_same


# Test 4: Forward pass reproducibility
def test_forward_pass():
    """Test if forward pass is reproducible."""
    print("\n" + "=" * 60)
    print("Test 4: Forward Pass Reproducibility")
    print("=" * 60)
    
    from models import AttentionLM
    
    config = {
        "model": {
            "vocab_size": 256,
            "n_ctx": 16,
            "d_model": 16,
            "n_head": 4,
            "n_layers": 2,
            "attn_type": "bilinear",
            "attn_scale": 0.35,
            "rope_base": 10000,
            "norm_type": "none",
            "norm_places": [],
            "use_rmsnorm_qk": False,
            "use_bias_qk": True,
        },
        "init": {
            "std_embed": 0.02,
            "std_qkv": 0.02,
            "std_o": 0.01,
        }
    }
    
    outputs = []
    for run in range(3):
        torch.manual_seed(42)
        model = AttentionLM.from_config(config)
        model.eval()
        
        # Fixed input
        torch.manual_seed(99)
        input_ids = torch.randint(0, 256, (2, 16))
        
        with torch.no_grad():
            output = model(input_ids)
        outputs.append(output.clone())
    
    all_same = all(torch.allclose(outputs[0], o, rtol=1e-5, atol=1e-7) for o in outputs[1:])
    print(f"Forward pass test: {'✓ PASS' if all_same else '✗ FAIL'}")
    if not all_same:
        print("  Different outputs across runs!")
        for i, o in enumerate(outputs):
            print(f"  Run {i}: sum={o.sum().item():.6f}, mean={o.mean().item():.6f}")
    return all_same


# Test 5: Training step reproducibility
def test_training_step():
    """Test if a single training step is reproducible."""
    print("\n" + "=" * 60)
    print("Test 5: Training Step Reproducibility")
    print("=" * 60)
    
    from models import AttentionLM
    from train.optim import create_optimizer
    
    config = {
        "model": {
            "vocab_size": 256,
            "n_ctx": 16,
            "d_model": 16,
            "n_head": 4,
            "n_layers": 2,
            "attn_type": "bilinear",
            "attn_scale": 0.35,
            "rope_base": 10000,
            "norm_type": "none",
            "norm_places": [],
            "use_rmsnorm_qk": False,
            "use_bias_qk": True,
        },
        "init": {
            "std_embed": 0.02,
            "std_qkv": 0.02,
            "std_o": 0.01,
        }
    }
    
    losses = []
    param_sums = []
    
    for run in range(3):
        torch.manual_seed(42)
        model = AttentionLM.from_config(config)
        model.train()
        
        opt_result = create_optimizer(
            model,
            lr=3e-4,
            muon_lr=0.02,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            use_muon=True,
        )
        from train.optim import Optimizers
        optimizer = opt_result.muon if isinstance(opt_result, Optimizers) else opt_result
        
        # Fixed input
        torch.manual_seed(99)
        input_ids = torch.randint(0, 256, (2, 16))
        
        # Forward + backward
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, 256),
            input_ids[:, 1:].reshape(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        param_sum = sum(p.sum().item() for p in model.parameters())
        param_sums.append(param_sum)
    
    loss_same = all(abs(losses[0] - l) < 1e-6 for l in losses[1:])
    param_same = all(abs(param_sums[0] - p) < 1e-4 for p in param_sums[1:])
    
    all_same = loss_same and param_same
    print(f"Training step test: {'✓ PASS' if all_same else '✗ FAIL'}")
    if not all_same:
        print("  Different results across runs!")
        for i in range(len(losses)):
            print(f"  Run {i}: loss={losses[i]:.6f}, param_sum={param_sums[i]:.6f}")
    return all_same


def main():
    print("\n" + "=" * 60)
    print("SEED REPRODUCIBILITY TEST SUITE")
    print("=" * 60)
    print()
    
    results = {
        "Basic seed": test_basic_seed(),
        "Model init": test_model_init(),
        "Data generation": test_data_generation(),
        "Forward pass": test_forward_pass(),
        "Training step": test_training_step(),
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(results.values())
    print()
    if all_passed:
        print("✓ All tests passed - seed reproducibility is working")
    else:
        print("✗ Some tests failed - seed reproducibility is broken")
        print("\nThis indicates a non-deterministic operation somewhere in the code.")
    
    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
