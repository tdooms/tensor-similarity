"""Training stage for the Laurence-derived subset-training figure family."""

from copy import deepcopy
import json

import polars as pl
import torch
from quimb.tensor import Tensor, TensorNetwork
from safetensors.torch import save_file
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.components.base import Term
from src.components.compose import pad
from src.components.linear import Linear
from loguru import logger

from src.datasets import MNIST
from src.figures import EXPERIMENT_DIR

OUT = EXPERIMENT_DIR / "subset_training"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 42]
SUBSET_CONFIGS = {
    "all": list(range(10)),
    "drop_9_8_7_6_5_4_3_2": [0, 1],
}
MODEL_CONFIG = dict(epochs=20, d_hidden=128, d_embed=256, batch_size=248)
RECORD_EVERY = 5


class Bilinear(nn.Linear):
    def __init__(self, d_in: int, d_out: int, bias: bool = False) -> None:
        super().__init__(d_in, 2 * d_out, bias=bias)

    def forward(self, x):
        left, right = super().forward(x).chunk(2, dim=-1)
        return left * right


class BilinearReadoutView:
    """Tensor-network view of the bilinear block + readout used for exact similarity."""

    def __init__(self, block: Bilinear, head: Linear) -> None:
        self.block = block
        self.head = head

    def _like(self):
        return dict(device=self.block.weight.device, dtype=self.block.weight.dtype)

    def network(self):
        d_hidden = self.head.in_features
        left, right = self.block.weight.view(2, d_hidden, self.block.in_features)
        out = torch.cat([torch.zeros(1, d_hidden, **self._like()), self.head.weight], dim=0)
        return TensorNetwork(
            [
                Tensor(pad(left, constant=False), inds=("h:b", "in:d0"), tags=("L",)),
                Tensor(pad(right, constant=False), inds=("h:b", "in:d1"), tags=("R",)),
                Tensor(out, inds=("out:d", "h:b"), tags=("U",)),
            ]
        )

    def terms(self, n_ctx):
        constant = torch.zeros(self.head.out_features + 1, self.block.in_features + 1, **self._like())
        constant[0, 0] = 1
        identity = TensorNetwork([Tensor(constant, inds=("out:d", "in:d0"), tags=("C",))])
        return [
            Term(identity, {"in:d0": "out:s"}),
            Term(self.network(), {"in:d0": "out:s", "in:d1": "out:s"}),
        ]


class SubsetTrainingModel(nn.Module):
    n_ctx = 1

    def __init__(
        self,
        *,
        seed: int,
        d_hidden: int = 128,
        d_embed: int = 256,
        d_input: int = 784,
        d_output: int = 10,
        batch_size: int = 248,
        lr: float = 1e-3,
        wd: float = 0.5,
        epochs: int = 20,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.seed, self.batch_size, self.lr, self.wd, self.epochs = seed, batch_size, lr, wd, epochs
        self.embed = Linear(d_input, d_embed, bias=False)
        self.block = Bilinear(d_embed, d_hidden, bias=False)
        self.head = Linear(d_hidden, d_output, bias=False)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.head(self.block(self.embed(x.flatten(start_dim=1))))

    def accuracy(self, logits, y):
        return (logits.argmax(dim=-1) == y).float().mean()

    def components(self):
        return [self.embed, BilinearReadoutView(self.block, self.head)]

    def step(self, x, y):
        logits = self(x)
        return self.criterion(logits, y), self.accuracy(logits, y)

    def fit(self, train, test, record_every_n_batches: int = 5):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        train_x = train.x.float() / 255.0
        test_x = test.x.float() / 255.0
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs * (train.x.size(0) // self.batch_size))
        history, checkpoints, batch_count = [], [], 0
        with torch.no_grad():
            val_loss, val_acc = self.eval().step(test_x, test.y)
        checkpoints.append({"batch": 0, "epoch": 0.0, "state_dict": deepcopy(self.state_dict())})
        history.append(
            {
                "train_loss": 0.0,
                "train_acc": 0.0,
                "val_loss": val_loss.item(),
                "val_acc": val_acc.item(),
                "batch": 0,
                "epoch": 0.0,
            }
        )

        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            epoch_metrics = []
            for batch_idx, idx in enumerate(torch.randperm(train.x.size(0), device=DEVICE).split(self.batch_size)):
                if idx.numel() < self.batch_size:
                    continue
                x, y = train_x[idx], train.y[idx]
                loss, acc = self.train().step(x, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_metrics.append((loss.item(), acc.item()))
                batch_count += 1
                if batch_count % record_every_n_batches == 0:
                    with torch.no_grad():
                        val_loss, val_acc = self.eval().step(test_x, test.y)
                    metrics = {
                        "train_loss": sum(l for l, _ in epoch_metrics) / len(epoch_metrics),
                        "train_acc": sum(a for _, a in epoch_metrics) / len(epoch_metrics),
                        "val_loss": val_loss.item(),
                        "val_acc": val_acc.item(),
                        "batch": batch_count,
                        "epoch": epoch + (batch_idx + 1) / (train.x.size(0) // self.batch_size),
                    }
                    history.append(metrics)
                    checkpoints.append(
                        {
                            "batch": batch_count,
                            "epoch": metrics["epoch"],
                            "state_dict": deepcopy(self.state_dict()),
                        }
                    )
                    pbar.set_description(", ".join(f"{key}: {value:.3f}" for key, value in list(metrics.items())[:4]))
        return history, checkpoints


def build_model(seed: int) -> SubsetTrainingModel:
    return SubsetTrainingModel(seed=seed, **MODEL_CONFIG).to(DEVICE)


def run_dir(seed: int, config_name: str):
    return OUT / f"seed_{seed}" / config_name


def save_checkpoints(seed: int, config_name: str, checkpoints):
    root = run_dir(seed, config_name)
    manifest = []
    for checkpoint in checkpoints:
        filename = f"batch_{checkpoint['batch']:06d}.safetensors"
        relative = f"checkpoints/{filename}"
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {key: value.detach().cpu().contiguous() for key, value in checkpoint["state_dict"].items()},
            str(root / relative),
        )
        manifest.append({"batch": checkpoint["batch"], "epoch": checkpoint["epoch"], "file": relative})
    root.mkdir(parents=True, exist_ok=True)
    (root / "checkpoints.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mnist = MNIST(device=DEVICE)
    datasets_by_name = {
        name: {
            "train": type(mnist.train)(
                mnist.train.x[(mnist.train.y[:, None] == torch.tensor(digits, device=DEVICE)).any(1)],
                mnist.train.y[(mnist.train.y[:, None] == torch.tensor(digits, device=DEVICE)).any(1)],
            ),
            "test": type(mnist.val)(
                mnist.val.x[(mnist.val.y[:, None] == torch.tensor(digits, device=DEVICE)).any(1)],
                mnist.val.y[(mnist.val.y[:, None] == torch.tensor(digits, device=DEVICE)).any(1)],
            ),
        }
        for name, digits in SUBSET_CONFIGS.items()
    }
    for i, seed in enumerate(SEEDS, start=1):
        logger.info(f"[{i}/{len(SEEDS)}] seed {seed}")
        for j, (name, dataset_bundle) in enumerate(datasets_by_name.items(), start=1):
            logger.info(f"      [{j}/{len(datasets_by_name)}] config {name}")
            model = build_model(seed)
            history, checkpoints = model.fit(
                dataset_bundle["train"], dataset_bundle["test"], record_every_n_batches=RECORD_EVERY
            )
            run_dir(seed, name).mkdir(parents=True, exist_ok=True)
            pl.DataFrame(history).write_ipc(run_dir(seed, name) / "history.feather")
            save_checkpoints(seed, name, checkpoints)
