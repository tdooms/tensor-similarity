import torch
from torch import nn, Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from transformers import PretrainedConfig, PreTrainedModel
from jaxtyping import Float
from tqdm import tqdm
from pandas import DataFrame
from einops import *
from copy import deepcopy

from src.components.linear import Linear
from src.components.mlp import MLP


def _collator(transform=None):
    def inner(batch):
        x = torch.stack([item[0] for item in batch]).float()
        y = torch.tensor([item[1] for item in batch])
        if x.is_mps:
            y = y.to("mps")
        return (x, y) if transform is None else (transform(x), y)
    return inner


class Config(PretrainedConfig):
    def __init__(
        self,
        lr: float = 1e-3,
        wd: float = 0.5,
        epochs: int = 100,
        batch_size: int = 2048,
        d_hidden: int = 256,
        d_embed: int = 128,
        n_layer: int = 1,
        d_input: int = 784,
        d_output: int = 10,
        bias: bool = False,
        seed: int = 42,
        **kwargs
    ):
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

        self.d_hidden = d_hidden
        self.d_embed = d_embed
        self.n_layer = n_layer
        self.d_input = d_input
        self.d_output = d_output
        self.bias = bias

        super().__init__(**kwargs)


class Model(PreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        torch.manual_seed(config.seed)

        d_input, d_embed, d_hidden, d_output = config.d_input, config.d_embed, config.d_hidden, config.d_output
        bias, n_layer = config.bias, config.n_layer

        self.embed = Linear(d_input, d_embed, bias=False)

        # scale=1.0: pure bilinear (no residual), matching Thomas's PaperModel
        self.blocks = nn.ModuleList([
            MLP(d_embed, d_hidden, bias=bias, scale=1.0)
            for _ in range(n_layer)
        ])

        # MLP projects back to d_embed, so head takes d_embed
        self.head = Linear(d_embed, d_output, bias=False)

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = lambda y_hat, y: (y_hat.argmax(dim=-1) == y).float().mean()

    def forward(self, x: Float[Tensor, "... inputs"]) -> Float[Tensor, "... outputs"]:
        x = self.embed(x.flatten(start_dim=1))
        for layer in self.blocks:
            x = layer(x)
        return self.head(x)

    def components(self):
        """Return components for use with src.components.similarity."""
        return [self.embed] + list(self.blocks) + [self.head]

    @property
    def w_e(self):
        return self.embed.weight.data  # [d_embed, d_input]

    @property
    def w_u(self):
        return self.head.weight.data  # [d_output, d_embed]

    @property
    def w_d(self):
        return self.blocks[0].d.weight.data  # [d_embed, d_hidden]

    @property
    def w_lr(self):
        """Get bilinear weights as [n_layer, 2, d_hidden, d_embed]."""
        return torch.stack([
            torch.stack([layer.l.weight.data, layer.r.weight.data])
            for layer in self.blocks
        ])

    @property
    def w_l(self):
        return self.w_lr.unbind(1)[0]

    @property
    def w_r(self):
        return self.w_lr.unbind(1)[1]

    @classmethod
    def from_config(cls, *args, **kwargs):
        return cls(Config(*args, **kwargs))

    @classmethod
    def from_pretrained(cls, path, *args, **kwargs):
        new = cls(Config(*args, **kwargs))
        new.load_state_dict(torch.load(path))
        return new

    def step(self, x, y):
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        return loss, accuracy

    def fit(self, train, test, transform=None, record_every_n_batches=None, save_checkpoints=False):
        torch.manual_seed(self.config.seed)
        torch.set_grad_enabled(True)

        optimizer = AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd)

        loader = DataLoader(train, batch_size=self.config.batch_size, shuffle=True,
                            drop_last=True, collate_fn=_collator(transform))

        batches_per_epoch = len(loader)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs * batches_per_epoch)

        test_x, test_y = test.x, test.y
        train_x, train_y = train.x, train.y

        pbar = tqdm(range(self.config.epochs))
        history = []
        checkpoints = [] if save_checkpoints else None
        batch_count = 0

        if save_checkpoints:
            val_loss, val_acc     = self.eval().step(test_x, test_y)
            tr_loss_0, tr_acc_0   = self.eval().step(train_x, train_y)
            checkpoints.append({
                'batch': 0, 'epoch': 0,
                'state_dict': deepcopy(self.state_dict()),
                'val_loss': val_loss.item(), 'val_acc': val_acc.item(),
            })
            history.append({
                "train/loss": tr_loss_0.item(), "train/acc": tr_acc_0.item(),
                "val/loss": val_loss.item(), "val/acc": val_acc.item(),
                "batch": 0, "epoch": 0,
            })

        for epoch in pbar:
            epoch_metrics = []

            for batch_idx, (x, y) in enumerate(loader):
                loss, acc = self.train().step(x, y)
                epoch_metrics.append((loss.item(), acc.item()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_count += 1

                if record_every_n_batches is not None and batch_count % record_every_n_batches == 0:
                    val_loss, val_acc = self.eval().step(test_x, test_y)
                    train_loss_eval, train_acc_eval = self.eval().step(train_x, train_y)

                    metrics = {
                        "train/loss": train_loss_eval.item(),
                        "train/acc": train_acc_eval.item(),
                        "val/loss": val_loss.item(), "val/acc": val_acc.item(),
                        "batch": batch_count,
                        "epoch": epoch + (batch_idx + 1) / batches_per_epoch,
                    }
                    history.append(metrics)

                    if save_checkpoints:
                        checkpoints.append({
                            'batch': batch_count,
                            'epoch': epoch + (batch_idx + 1) / batches_per_epoch,
                            'state_dict': deepcopy(self.state_dict()),
                            'val_loss': val_loss.item(), 'val_acc': val_acc.item(),
                        })

                    pbar.set_description(', '.join(f"{k}: {v:.3f}" for k, v in list(metrics.items())[:4]))

                scheduler.step()

            if record_every_n_batches is None:
                val_loss, val_acc = self.eval().step(test_x, test_y)

                metrics = {
                    "train/loss": sum(l for l, _ in epoch_metrics) / len(epoch_metrics),
                    "train/acc": sum(a for _, a in epoch_metrics) / len(epoch_metrics),
                    "val/loss": val_loss.item(), "val/acc": val_acc.item(),
                    "batch": batch_count, "epoch": epoch + 1,
                }
                history.append(metrics)

                if save_checkpoints:
                    checkpoints.append({
                        'batch': batch_count, 'epoch': epoch + 1,
                        'state_dict': deepcopy(self.state_dict()),
                        'val_loss': val_loss.item(), 'val_acc': val_acc.item(),
                    })

                pbar.set_description(', '.join(f"{k}: {v:.3f}" for k, v in list(metrics.items())[:4]))

        torch.set_grad_enabled(False)

        if save_checkpoints:
            return DataFrame.from_records(history), checkpoints
        else:
            return DataFrame.from_records(history)

    def decompose(self):
        """Decompose into per-class interaction matrices in input space."""
        l, r = self.w_lr[0].unbind()        # [d_hidden, d_embed]
        u_eff = self.w_u @ self.w_d          # [d_output, d_hidden]
        b = einsum(u_eff, l, r, "cls h, h emb1, h emb2 -> cls emb1 emb2")
        b = 0.5 * (b + b.mT)
        vals, vecs = torch.linalg.eigh(b)
        vecs = einsum(vecs, self.embed.weight, "cls emb comp, emb inp -> cls comp inp")
        return vals, vecs
