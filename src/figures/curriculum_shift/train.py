"""Training stage for the curriculum-shift figure family."""

from copy import deepcopy
import json

import polars as pl
import torch
from loguru import logger
from safetensors.torch import save_file

from src.datasets import MNIST
from src.figures import EXPERIMENT_DIR
from src.models.deep_mlp import DeepMLP

OUT = EXPERIMENT_DIR / "curriculum_shift"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CURRICULUM = [
    {"name": "base", "digits": list(range(5))},
    {"name": "add_5", "digits": list(range(6))},
    {"name": "add_6", "digits": list(range(7))},
    {"name": "add_7", "digits": list(range(8))},
    {"name": "add_8", "digits": list(range(9))},
    {"name": "add_9", "digits": list(range(10))},
    {"name": "remove_9", "digits": list(range(9))},
    {"name": "readd_9", "digits": list(range(10))},
]
SEED = 42
EPOCHS_PER_STAGE = 15
BATCH_SIZE = 256
CHECKPOINT_EVERY = 10


def build_model(seed):
    torch.manual_seed(seed)
    return DeepMLP(d_input=784, d_model=128, d_hidden=256, d_output=10, n_layers=1).to(DEVICE)


def stage_dir(name):
    return OUT / name


def save_checkpoints(name, checkpoints):
    root = stage_dir(name)
    manifest = []
    for checkpoint in checkpoints:
        filename = f"batch_{checkpoint['batch']:06d}.safetensors"
        relative = f"checkpoints/{filename}"
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {key: value.detach().cpu().contiguous() for key, value in checkpoint["state_dict"].items()},
            str(root / relative),
        )
        manifest.append({"batch": checkpoint["batch"], "file": relative})
    root.mkdir(parents=True, exist_ok=True)
    (root / "checkpoints.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def train_stage(model, optimizer, train, test, stage_name):
    loss_fn = torch.nn.CrossEntropyLoss()
    checkpoints = []
    history = {"batch": [], "train_loss": [], "train_acc": [], "val_acc": []}
    batch_idx = 0
    train_x = (train.x.float() / 255.0 - 0.1307) / 0.3081
    test_x = (test.x.float() / 255.0 - 0.1307) / 0.3081

    for epoch in range(EPOCHS_PER_STAGE):
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
        logger.info(f"      {stage_name} epoch={epoch + 1}/{EPOCHS_PER_STAGE} val_acc={history['val_acc'][-1]:.4f}")

    return checkpoints, history


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = build_model(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mnist = MNIST(device=DEVICE)
    (OUT / "curriculum.json").write_text(json.dumps(CURRICULUM, indent=2), encoding="utf-8")
    for i, stage in enumerate(CURRICULUM, start=1):
        name, digits = stage["name"], stage["digits"]
        keep = torch.tensor(digits, device=DEVICE)
        train_mask = (mnist.train.y[:, None] == keep).any(1)
        test_mask = (mnist.val.y[:, None] == keep).any(1)
        train = type(mnist.train)(mnist.train.x[train_mask], mnist.train.y[train_mask])
        test = type(mnist.val)(mnist.val.x[test_mask], mnist.val.y[test_mask])
        logger.info(f"[{i}/{len(CURRICULUM)}] stage {name} (digits {digits})")
        checkpoints, history = train_stage(model, optimizer, train, test, name)
        save_checkpoints(name, checkpoints)
        stage_dir(name).mkdir(parents=True, exist_ok=True)
        pl.DataFrame(history).write_ipc(stage_dir(name) / "history.feather")
        logger.info(f"      saved {len(checkpoints)} checkpoints to {stage_dir(name)}")
