"""Train 100 BilinearMLP models on modular addition with multiprocessing."""

import json
import torch
import torch.multiprocessing as mp
from pathlib import Path
from functools import partial
import time

from config import CONFIG
from data import get_dataloaders
from train import train_model


def train_single_seed(seed, config, train_indices, test_indices):
    """Train a single model (worker function for multiprocessing)."""
    from data import ModularAdditionDataset
    from torch.utils.data import DataLoader

    # Recreate data loaders in this process (can't pickle DataLoader)
    train_ds = ModularAdditionDataset(config["p"], train_indices)
    test_ds = ModularAdditionDataset(config["p"], test_indices)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

    # Use CPU for multiprocessing (MPS doesn't work well with mp)
    worker_config = config.copy()
    worker_config["device"] = "cpu"

    model, checkpoints = train_model(seed, train_loader, test_loader, worker_config)

    # Save checkpoint
    ckpt_dir = Path("checkpoints")
    ckpt_path = ckpt_dir / f"seed_{seed:03d}.pt"
    torch.save({
        "seed": seed,
        "model_state_dict": model.state_dict(),
        "config": {k: str(v) for k, v in config.items()},
        "checkpoints": checkpoints,
    }, ckpt_path)

    if checkpoints:
        final = checkpoints[-1]
        return seed, final["train_acc"], final["test_acc"]
    return seed, 0.0, 0.0


def main():
    config = CONFIG
    num_workers = config.get("num_workers", 4)

    print(f"Configuration: {config}")
    print(f"Using {num_workers} parallel workers")

    # Create checkpoint directory
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    # Get train/test split indices (same for all seeds)
    torch.manual_seed(42)
    p = config["p"]
    n_total = p * p
    indices = torch.randperm(n_total).tolist()
    n_train = int(n_total * config["train_fraction"])
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    print(f"Train size: {len(train_indices)}, Test size: {len(test_indices)}")

    start_time = time.time()

    # Check which seeds still need training
    seeds_to_train = []
    for seed in range(config["num_seeds"]):
        ckpt_path = ckpt_dir / f"seed_{seed:03d}.pt"
        if not ckpt_path.exists():
            seeds_to_train.append(seed)

    print(f"Seeds to train: {len(seeds_to_train)} (skipping {config['num_seeds'] - len(seeds_to_train)} existing)")

    if not seeds_to_train:
        print("All models already trained!")
        return

    # Use multiprocessing pool
    mp.set_start_method('spawn', force=True)

    worker_fn = partial(
        train_single_seed,
        config=config,
        train_indices=train_indices,
        test_indices=test_indices
    )

    all_results = []

    with mp.Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(worker_fn, seeds_to_train):
            seed, train_acc, test_acc = result
            all_results.append(result)
            elapsed = time.time() - start_time
            remaining = len(seeds_to_train) - len(all_results)
            rate = len(all_results) / elapsed if elapsed > 0 else 0
            eta = remaining / rate if rate > 0 else 0
            print(f"[{len(all_results)}/{len(seeds_to_train)}] Seed {seed}: "
                  f"train={train_acc:.4f}, test={test_acc:.4f} "
                  f"(elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s)")

    # Collect metadata from all checkpoints
    all_metadata = {}
    for seed in range(config["num_seeds"]):
        ckpt_path = ckpt_dir / f"seed_{seed:03d}.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location='cpu')
            all_metadata[seed] = ckpt.get("checkpoints", [])

    # Save training metadata
    with open("training_metadata.json", "w") as f:
        json.dump(all_metadata, f, indent=2)

    total_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"Training complete in {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Models saved to: {ckpt_dir}/")
    print(f"Metadata saved to: training_metadata.json")
    print("="*60)


if __name__ == "__main__":
    main()
