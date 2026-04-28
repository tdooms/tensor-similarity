"""Training stage for the seed-convergence figure family."""

from copy import deepcopy
import json

import polars as pl
import torch
from loguru import logger
from safetensors.torch import save_file

from src.datasets import MNIST
from src.figures import EXPERIMENT_DIR
from src.models.deep_mlp import DeepMLP

OUT = EXPERIMENT_DIR / "seed_convergence"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [1, 2, 3, 42, 99]
REFERENCE_SEED = 42
EPOCHS = 20
BATCH_SIZE = 256
CHECKPOINT_EVERY = 10


def build_model(seed):
    torch.manual_seed(seed)
    return DeepMLP(d_input=784, d_model=128, d_hidden=256, d_output=10, n_layers=1).to(DEVICE)


def seed_dir(seed):
    return OUT / f"seed_{seed}"


def save_checkpoints(seed, checkpoints):
    root = seed_dir(seed)
    manifest = []
    for checkpoint in checkpoints:
        filename = f"batch_{checkpoint['batch']:06d}.safetensors"
        relative = f"checkpoints/{filename}"
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {name: tensor.detach().cpu().contiguous() for name, tensor in checkpoint["state_dict"].items()},
            str(root / relative),
        )
        manifest.append({"batch": checkpoint["batch"], "file": relative})
    root.mkdir(parents=True, exist_ok=True)
    (root / "checkpoints.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def train_one_seed(seed, train, test):
    model = build_model(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_x = (train.x.float() / 255.0 - 0.1307) / 0.3081
    test_x = (test.x.float() / 255.0 - 0.1307) / 0.3081

    checkpoints = []
    history = {"batch": [], "train_loss": [], "train_acc": [], "val_acc": []}
    batch_idx = 0

    for epoch in range(EPOCHS):
        model.train()
        for idx in torch.randperm(train.x.size(0), device=DEVICE).split(BATCH_SIZE):
            x, y = train_x[idx], train.y[idx]
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_idx += 1

            if batch_idx % CHECKPOINT_EVERY == 0:
                checkpoints.append({"batch": batch_idx, "state_dict": deepcopy(model.state_dict())})
                acc = (logits.argmax(-1) == y).float().mean().item()
                model.eval()
                with torch.no_grad():
                    val_acc = (model(test_x).argmax(-1) == test.y).float().mean().item()
                model.train()

                history["batch"].append(batch_idx)
                history["train_loss"].append(loss.item())
                history["train_acc"].append(acc)
                history["val_acc"].append(val_acc)

        logger.info(f"      seed={seed} epoch={epoch + 1}/{EPOCHS} "
                    f"loss={history['train_loss'][-1]:.4f} val_acc={history['val_acc'][-1]:.4f}")

    return checkpoints, history


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mnist = MNIST(device=DEVICE)
    for i, seed in enumerate(SEEDS, start=1):
        logger.info(f"[{i}/{len(SEEDS)}] training seed {seed}")
        checkpoints, history = train_one_seed(seed, mnist.train, mnist.val)
        save_checkpoints(seed, checkpoints)
        seed_dir(seed).mkdir(parents=True, exist_ok=True)
        pl.DataFrame(history).write_ipc(seed_dir(seed) / "history.feather")
        logger.info(f"      saved {len(checkpoints)} checkpoints to {seed_dir(seed)}")
