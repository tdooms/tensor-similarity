"""Shared utilities for scripts/vision/ experiments.

Each script must set up sys.path before importing this module:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tensor-mars root
    sys.path.insert(0, str(Path(__file__).resolve().parent))      # scripts/vision/
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from kornia.color import RgbToGrayscale
from tqdm import tqdm

import torch.nn as nn
from pandas import DataFrame
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.components.similarity import precompile, similarity_parts
from src.datasets.base import Dataset
from src.datasets.mnist import MNIST
from src.datasets.svhn import SVHN
from src.figures import cosine_from_parts
from src.models.deep_mlp import DeepMLP


# ── Training ───────────────────────────────────────────────────────────────

def _cpu_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _collate(batch):
    x = torch.stack([b[0] for b in batch]).float()
    y = torch.stack([b[1] for b in batch])
    return x, y


def fit(model, train, val, epochs=10, lr=1e-3, wd=0.5, batch_size=248,
        record_every_n_batches=50, save_checkpoints=False, seed=42):
    """Train model and return (history_df, checkpoints) or just history_df.

    train / val: Dataset objects with .x and .y tensors.
    Checkpoints are dicts with keys: batch, epoch, state_dict, val_acc, val_loss.
    """
    torch.manual_seed(seed)
    torch.set_grad_enabled(True)

    criterion = nn.CrossEntropyLoss()
    accuracy  = lambda yh, y: (yh.argmax(-1) == y).float().mean()

    def step(x, y):
        yh = model(x)
        return criterion(yh, y), accuracy(yh, y)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loader    = DataLoader(train, batch_size=batch_size, shuffle=True,
                           drop_last=True, collate_fn=_collate)
    n_batches = len(loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * n_batches)

    val_x, val_y     = val.x.float(), val.y
    train_x, train_y = train.x.float(), train.y

    history     = []
    checkpoints = [] if save_checkpoints else None
    batch_count = 0

    if save_checkpoints:
        model.eval()
        with torch.no_grad():
            vl, va = step(val_x, val_y)
            tl, ta = step(train_x, train_y)
        checkpoints.append({'batch': 0, 'epoch': 0,
                            'state_dict': _cpu_state_dict(model),
                            'val_loss': vl.item(), 'val_acc': va.item()})
        history.append({'train/loss': tl.item(), 'train/acc': ta.item(),
                        'val/loss': vl.item(), 'val/acc': va.item(),
                        'batch': 0, 'epoch': 0})

    for epoch in range(epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(loader):
            loss, _ = step(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            batch_count += 1

            if record_every_n_batches and batch_count % record_every_n_batches == 0:
                model.eval()
                with torch.no_grad():
                    vl, va = step(val_x, val_y)
                    tl, ta = step(train_x, train_y)
                model.train()

                metrics = {'train/loss': tl.item(), 'train/acc': ta.item(),
                           'val/loss': vl.item(), 'val/acc': va.item(),
                           'batch': batch_count,
                           'epoch': epoch + (batch_idx + 1) / n_batches}
                history.append(metrics)

                if save_checkpoints:
                    checkpoints.append({'batch': batch_count,
                                        'epoch': metrics['epoch'],
                                        'state_dict': _cpu_state_dict(model),
                                        'val_loss': vl.item(), 'val_acc': va.item()})

        if not record_every_n_batches:
            model.eval()
            with torch.no_grad():
                vl, va = step(val_x, val_y)
                tl, ta = step(train_x, train_y)
            metrics = {'train/loss': tl.item(), 'train/acc': ta.item(),
                       'val/loss': vl.item(), 'val/acc': va.item(),
                       'batch': batch_count, 'epoch': epoch + 1}
            history.append(metrics)
            if save_checkpoints:
                checkpoints.append({'batch': batch_count, 'epoch': epoch + 1,
                                    'state_dict': _cpu_state_dict(model),
                                    'val_loss': vl.item(), 'val_acc': va.item()})

    torch.set_grad_enabled(False)
    model.eval()

    df = DataFrame.from_records(history)
    return (df, checkpoints) if save_checkpoints else df

_REPO       = Path(__file__).resolve().parents[2]
_WORKSPACE  = _REPO / "scripts" / "vision"
DATA_DIR    = _WORKSPACE / "data"
DATA_DIR.mkdir(exist_ok=True)


# ── Dataset loaders ────────────────────────────────────────────────────────

def _filter_digits(x, y, digits):
    if digits is None:
        return x, y
    mask = torch.zeros(len(y), dtype=torch.bool, device=y.device)
    for d in digits:
        mask |= (y == d)
    return x[mask], y[mask]


def load_mnist(device='cpu', digits=None):
    ds = MNIST(device=device)
    def _process(split):
        x, y = _filter_digits(split.x.float() / 255.0, split.y, digits)
        return Dataset(x, y)
    return _process(ds.train), _process(ds.val)


def load_svhn(device='cpu', digits=None, n_train=73257, offset=0):
    ds = SVHN(device='cpu')  # grayscale on CPU first — RGB→gray is 3× smaller before moving to device
    to_gray = RgbToGrayscale()
    def _process(split, n=None, off=0):
        x_raw = split.x[off:off + n] if n is not None else split.x
        y_raw = split.y[off:off + n] if n is not None else split.y
        x, y = _filter_digits(to_gray(x_raw.float() / 255.0), y_raw, digits)
        return Dataset(x.to(device), y.to(device))
    return _process(ds.train, n_train, offset), _process(ds.val)


# ── Model ──────────────────────────────────────────────────────────────────

def new_model(seed=42, d_input=784, d_model=128, d_hidden=256, d_output=10, device='cpu'):
    torch.manual_seed(seed)
    return DeepMLP(d_input=d_input, d_model=d_model, d_hidden=d_hidden,
                   d_output=d_output).to(device)


# ── Similarities ───────────────────────────────────────────────────────────

@torch.no_grad()
def tn_sim(m1, m2):
    """Thomas's Gaussian functional similarity (data-free, gauge-invariant)."""
    return cosine_from_parts(*similarity_parts(m1, m2)).item()


def naive_weight_cosine(m1, m2):
    w1 = torch.cat([p.data.flatten() for p in m1.parameters()])
    w2 = torch.cat([p.data.flatten() for p in m2.parameters()])
    return (w1 @ w2 / (w1.norm() * w2.norm())).item()


def weight_slice_cosine(m1, m2, digit: int):
    """Cosine similarity of the output head row for a single digit."""
    w1 = m1.head.weight[digit]
    w2 = m2.head.weight[digit]
    return (w1 @ w2 / (w1.norm() * w2.norm())).item()


def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    return ((X.T @ Y).norm(p='fro') ** 2 /
            ((X.T @ X).norm(p='fro') * (Y.T @ Y).norm(p='fro'))).item()


# ── Interaction matrix ─────────────────────────────────────────────────────

@torch.no_grad()
def get_interaction_matrix(model, include_embedding=True, symmetrize=True):
    """M[o,i,j] = Σ_h u_eff[o,h] · el[i,h] · er[j,h] for a single-body DeepMLP."""
    mlp  = model.body[0]
    l    = mlp.l.weight          # [d_hidden, d_model]
    r    = mlp.r.weight          # [d_hidden, d_model]
    d_w  = mlp.d.weight          # [d_model,  d_hidden]
    u    = model.head.weight     # [d_output, d_model]
    u_eff = u @ d_w              # [d_output, d_hidden]

    if include_embedding:
        e  = model.embed.weight  # [d_model, d_input]
        el = torch.einsum("mi,hm->ih", e, l)   # [d_input, d_hidden]
        er = torch.einsum("mi,hm->ih", e, r)
        M  = torch.einsum("oh,ih,jh->oij", u_eff, el, er)
    else:
        M = torch.einsum("oh,hi,hj->oij", u_eff, l, r)

    if symmetrize:
        M = 0.5 * (M + M.transpose(1, 2))
    return M


def slice_sim(m1, m2, digit=7):
    """Gaussian functional similarity restricted to a single output class.

    Isserlis: E[f1·f2] = Tr(M1)Tr(M2) + 2·Tr(M1@M2), E[f²] = Tr(M)² + 2·Tr(M²).
    """
    M1 = get_interaction_matrix(m1)[digit]
    M2 = get_interaction_matrix(m2)[digit]
    cross = M1.trace() * M2.trace() + 2 * (M1 @ M2).trace()
    norm1 = M1.trace() ** 2 + 2 * (M1 @ M1).trace()
    norm2 = M2.trace() ** 2 + 2 * (M2 @ M2).trace()
    return (cross / (norm1 * norm2).sqrt()).item()


# ── Progressive training ───────────────────────────────────────────────────

def fit_progressive(seed, digit_curriculum, device='cpu',
                    d_input=784, d_model=128, d_hidden=256,
                    lr=1e-3, wd=0.5, batch_size=248,
                    record_every_n_batches=50,
                    load_fn=load_mnist):
    """Train a DeepMLP progressively through digit_curriculum.

    Returns (cps_by_stage, hist_by_stage). Each stage value is a list of checkpoint
    dicts with keys: batch, epoch, state_dict, val_loss, val_acc.
    """
    cps_by_stage, hist_by_stage = {}, {}
    state = None
    for stage in digit_curriculum:
        name, digits = stage['name'], stage['digits']
        epochs = stage.get('epochs', 20)
        train, val = load_fn(device=device, digits=digits)
        model = new_model(seed=seed, d_input=d_input, d_model=d_model, d_hidden=d_hidden, device=device)
        if state is not None:
            model.load_state_dict(state)
        history, checkpoints = fit(
            model, train, val, epochs=epochs, lr=lr, wd=wd,
            batch_size=batch_size,
            record_every_n_batches=record_every_n_batches,
            save_checkpoints=True, seed=seed,
        )
        cps_by_stage[name] = checkpoints
        hist_by_stage[name] = history
        state = deepcopy(model.state_dict())
        print(f"  {name}: val_acc={history['val/acc'].iloc[-1]:.4f}")
    return cps_by_stage, hist_by_stage


# ── Plotting helpers ───────────────────────────────────────────────────────

def build_cumulative_history(hist_by_stage, digit_curriculum):
    cum_batch, train_acc, val_acc, train_loss, val_loss = [], [], [], [], []
    offset = 0
    for stage in digit_curriculum:
        h = hist_by_stage[stage['name']]
        skip_zero = offset > 0
        for _, row in h.iterrows():
            if skip_zero and row['batch'] == 0:
                continue
            cum_batch.append(offset + row['batch'])
            train_acc.append(row['train/acc'])
            val_acc.append(row['val/acc'])
            train_loss.append(row['train/loss'])
            val_loss.append(row['val/loss'])
        if len(h) > 0:
            offset += h['batch'].iloc[-1]
    return (np.array(cum_batch), np.array(train_acc), np.array(val_acc),
            np.array(train_loss), np.array(val_loss))


def get_stage_boundaries(hist_by_stage, digit_curriculum):
    boundaries, offset = [], 0
    for i, stage in enumerate(digit_curriculum):
        if i > 0:
            boundaries.append((offset, stage['name']))
        h = hist_by_stage[stage['name']]
        if len(h) > 0:
            offset += h['batch'].iloc[-1]
    return boundaries


def get_stage_spans(hist_by_stage, digit_curriculum):
    """Returns [(x_start, x_end, name), ...] for every stage."""
    spans, offset = [], 0
    for stage in digit_curriculum:
        name = stage['name']
        h = hist_by_stage[name]
        end = offset + (h['batch'].iloc[-1] if len(h) > 0 else 0)
        spans.append((offset, end, name))
        offset = end
    return spans


