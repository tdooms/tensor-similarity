#!/usr/bin/env python3
"""Train 2-layer attn-only d_model=256 models on SimpleStories for 5B tokens.

Two models:
  --attn bilinear   : bilinear QK with batchnorm + ortho_init
  --attn softmax    : softmax with QK RMSNorm

Architecture: d_model=256, n_head=8, d_head=32, n_layers=2, n_ctx=1024
Data: SimpleStories concatenated into 1024-token windows (~586k windows/epoch)
Target: 5B tokens ≈ 38k steps at batch=128

Usage:
    python train_d256_5B.py --attn bilinear
    python train_d256_5B.py --attn softmax
"""
import torch, torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from lib import AttentionLM, Trainer
from train_induction_static_norm import BilinearBatchNorm
from pathlib import Path


CACHE_DIR = Path(__file__).parent / "cached_tokens"
N_CTX = 1024
TARGET_TOKENS = 5_000_000_000


class ConcatDataset(Dataset):
    """Concatenate all stories end-to-end, chunk into n_ctx windows."""
    def __init__(self, split="train", n_ctx=N_CTX, max_samples=None):
        path = CACHE_DIR / f"{split}_perstory.pt"
        data = torch.load(path, weights_only=True).to(torch.long)
        flat = data.reshape(-1)
        flat = flat[flat > 0]
        n_windows = len(flat) // n_ctx
        self.data = flat[:n_windows * n_ctx].reshape(n_windows, n_ctx)
        if max_samples:
            self.data = self.data[:max_samples]
        self.total_tokens = len(flat)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return {"input_ids": self.data[idx]}


# ---- Techniques ----

def apply_batchnorm(model, d_model, n_head, n_ctx):
    for i, layer in enumerate(model.layers):
        new_layer = BilinearBatchNorm(
            d_model=d_model, n_head=n_head, n_ctx=n_ctx,
            scale=1.0, use_bias_qkv=True, use_bias_o=True,
        )
        model.layers[i] = new_layer
    for layer in model.layers:
        nn.init.normal_(layer.q.weight, std=0.02)
        nn.init.normal_(layer.k.weight, std=0.02)
        nn.init.normal_(layer.q2.weight, std=0.02)
        nn.init.normal_(layer.k2.weight, std=0.02)
        nn.init.normal_(layer.v.weight, std=0.02)
        nn.init.normal_(layer.o.weight, std=0.01)


def apply_ortho_init_bilinear(model):
    for layer in model.layers:
        for attr in ['q', 'k', 'q2', 'k2', 'v', 'o']:
            if hasattr(layer, attr):
                nn.init.orthogonal_(getattr(layer, attr).weight)
                if getattr(layer, attr).bias is not None:
                    nn.init.zeros_(getattr(layer, attr).bias)


# ---- Induction evaluation ----

def make_repeated_sequences(vocab_size, half_len=512, n_sequences=100, seed=42):
    """half_len=512 so full seq = 1024 = n_ctx."""
    rng = np.random.RandomState(seed)
    seqs = rng.randint(1, vocab_size, size=(n_sequences, half_len))
    doubled = np.concatenate([seqs, seqs], axis=1)
    return torch.tensor(doubled, dtype=torch.long), half_len


@torch.no_grad()
def eval_induction_full(model, sequences, half_len, device, batch_size=16):
    model.eval()
    n = sequences.shape[0]
    all_losses = []
    total_correct = 0
    total_tokens = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = sequences[start:end].to(device)
        logits = model(batch)
        shift_logits = logits[:, :-1, :]
        shift_labels = batch[:, 1:]
        B, T, V = shift_logits.shape
        loss = F.cross_entropy(
            shift_logits.reshape(B * T, V), shift_labels.reshape(B * T),
            reduction="none",
        ).reshape(B, T)
        all_losses.append(loss.cpu())

        pred_logits = logits[:, half_len:2*half_len-1, :]
        targets = batch[:, half_len+1:2*half_len]
        preds = pred_logits.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_tokens += targets.numel()

    all_losses = torch.cat(all_losses, dim=0)
    mean_loss = all_losses.mean(dim=0).numpy()
    first_half = mean_loss[:half_len - 1].mean()
    second_half = mean_loss[half_len:].mean()
    induction_score = float(first_half - second_half)
    accuracy = total_correct / total_tokens
    return induction_score, float(first_half), float(second_half), accuracy


# ---- Main ----

def main(batch_size=128, attn="bilinear"):
    n_layers = 2
    d_model = 256
    n_head = 8
    vocab_size = 4096

    train_ds = ConcatDataset("train", n_ctx=N_CTX)
    val_ds = ConcatDataset("test", n_ctx=N_CTX, max_samples=500)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    tokens_per_step = batch_size * N_CTX
    max_steps = TARGET_TOKENS // tokens_per_step
    steps_per_epoch = len(train_ds) // batch_size
    n_epochs = max_steps / steps_per_epoch

    print(f"Train windows: {len(train_ds):,}, Steps/epoch: {steps_per_epoch:,}")
    print(f"Total steps: {max_steps:,} ({n_epochs:.1f} epochs, {max_steps * tokens_per_step / 1e9:.1f}B tokens)")

    if attn == "bilinear":
        attn_type = "bilinear"
        name = "d256_2layer_bilinear_batchnorm_5B"
        use_rmsnorm_qk = False
    else:
        attn_type = "softmax"
        name = "d256_2layer_softmax_qknorm_5B"
        use_rmsnorm_qk = True

    cfg = {
        "name": name,
        "seed": 42,
        "model": {
            "vocab_size": vocab_size,
            "n_ctx": N_CTX,
            "d_model": d_model,
            "n_head": n_head,
            "n_layers": n_layers,
            "attn_type": attn_type,
            "attn_scale": 1.0,
            "rope_base": 10000,
            "norm_type": "layernorm",
            "norm_place": "pre_unembed",
            "use_rmsnorm_qk": use_rmsnorm_qk,
            "use_bias_qkv": True,
            "use_bias_o": True,
        },
        "init": {
            "std_embed": 0.02,
            "std_qkv": 0.02,
            "std_o": 0.01,
        },
        "train": {
            "batch_size": batch_size,
            "lr": 3e-4,
            "muon_lr": 0.02,
            "use_muon": True,
            "betas": [0.9, 0.95],
            "weight_decay": 0.1,
            "max_steps": max_steps,
            "warmup_steps": 1000,
            "lr_decay_frac": 0.1,
            "grad_clip": 1.0,
            "dtype": "bfloat16",
            "debug": False,
            "eval_every": 5000,
            "save_every": steps_per_epoch,
            "log_every": 1000,
        },
        "loss": {
            "type": "next_token_ce",
            "label_smoothing": 0.0,
        },
    }

    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.init()
        torch.cuda.empty_cache()

    print(f"=== {name} ===")
    print(f"d_model={d_model}, n_head={n_head}, d_head={d_model//n_head}, n_ctx={N_CTX}")

    model = AttentionLM.from_config(cfg)

    if attn == "bilinear":
        apply_batchnorm(model, d_model, n_head, N_CTX)
        apply_ortho_init_bilinear(model)
        print("Applied batchnorm + ortho_init")

    model = model.to(device)
    model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} (compiled)")

    # Induction test (half_len=512, full=1024)
    sequences, half_len = make_repeated_sequences(vocab_size, half_len=512, n_sequences=100)
    print(f"Induction test: {sequences.shape[0]} seqs, half_len={half_len}")

    def induction_callback(model, step):
        score, first, second, accuracy = eval_induction_full(
            model, sequences, half_len, device
        )
        print(f"  [step {step}] induction_score={score:.4f} "
              f"1st={first:.4f} 2nd={second:.4f} "
              f"toy_acc={accuracy:.6f} ({accuracy*100:.4f}%)")
        return {
            "induction_score": score,
            "induction_first_half_loss": first,
            "induction_second_half_loss": second,
            "toy_induction_accuracy": accuracy,
        }

    # Monkey-patch Trainer to log every 1000 steps instead of every 10
    original_train = Trainer.train

    def patched_train(self, eval_every=100, save_every=1000):
        """Override to log metrics every 1000 steps instead of 10."""
        import json
        from tqdm import tqdm
        from lib import compute_loss

        self.model.train()
        data_iter = iter(self.train_dataloader)
        pbar = tqdm(total=self.max_steps, desc="Training")

        while self.step < self.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(self.device)

            try:
                self.optimizer.zero_grad()

                if self.use_amp:
                    with torch.amp.autocast("cuda", dtype=self.pt_dtype):
                        logits = self.model(input_ids)
                        loss = compute_loss(
                            logits, input_ids,
                            loss_type=self.loss_type,
                            label_smoothing=self.label_smoothing,
                        )
                    loss.backward()
                    preclip_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                else:
                    logits = self.model(input_ids)
                    loss = compute_loss(
                        logits, input_ids,
                        loss_type=self.loss_type,
                        label_smoothing=self.label_smoothing,
                    )
                    loss.backward()
                    preclip_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                self.scheduler.step()
                self.step += 1

                # Log every 1000 steps
                if self.step % 1000 == 0:
                    self.log_metrics({"train_loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]})

                pbar.set_postfix(loss=f"{loss.item():.4f}")
                pbar.update(1)

                if self.step % eval_every == 0 and self.val_dataloader is not None:
                    from lib import evaluate
                    val_loss = evaluate(self.model, self.val_dataloader, self.device)
                    self.log_metrics({"val_loss": val_loss})
                    for cb in self.eval_callbacks:
                        cb_metrics = cb(self.model, self.step)
                        if cb_metrics:
                            self.log_metrics(cb_metrics)
                    self.model.train()

                if self.step % save_every == 0:
                    self.save_checkpoint()

            except Exception as e:
                self.log_error(e, "training step")
                raise

        pbar.close()
        self.save_checkpoint("final")

    trainer = Trainer(
        model=model,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        cfg=cfg,
        device=device,
    )
    trainer.eval_callbacks.append(induction_callback)

    # Use patched train to log every 1k
    patched_train(
        trainer,
        eval_every=cfg["train"]["eval_every"],
        save_every=cfg["train"]["save_every"],
    )
    print(f"Done. Run dir: {trainer.run_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--attn", choices=["bilinear", "softmax"], required=True)
    args = parser.parse_args()
    main(args.batch_size, args.attn)
