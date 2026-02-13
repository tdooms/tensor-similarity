#!/usr/bin/env python3
"""Train 768-dim 12-head bilinear model with QK norm for 5 epochs with periodic induction eval."""
import yaml
import torch
import numpy as np
import torch.nn.functional as F
from lib import AttentionLM, create_dataloaders, Trainer


def make_repeated_sequences(vocab_size, half_len=256, n_sequences=50, seed=42):
    rng = np.random.RandomState(seed)
    seqs = rng.randint(1, vocab_size, size=(n_sequences, half_len))
    doubled = np.concatenate([seqs, seqs], axis=1)
    return torch.tensor(doubled, dtype=torch.long), half_len


@torch.no_grad()
def eval_induction(model, sequences, half_len, device):
    model.eval()
    batch = sequences.to(device)
    logits = model(batch)
    shift_logits = logits[:, :-1, :]
    shift_labels = batch[:, 1:]
    B, T, V = shift_logits.shape
    loss = F.cross_entropy(
        shift_logits.reshape(B * T, V),
        shift_labels.reshape(B * T),
        reduction="none",
    ).reshape(B, T)
    mean_loss = loss.mean(dim=0).cpu().numpy()
    first_half = mean_loss[:half_len - 1].mean()
    second_half = mean_loss[half_len:].mean()
    return float(first_half - second_half), float(first_half), float(second_half)


def main():
    config_path = "/workspace/tensor-mars/workspaces/logan/configs/main768_bilinear_12h_5epoch_qknorm.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.init()
        torch.cuda.empty_cache()

    model_cfg = cfg["model"]
    train_cfg = cfg.get("train", {})

    print("Creating dataloaders...")
    train_dl, val_dl = create_dataloaders(
        n_ctx=model_cfg["n_ctx"],
        batch_size=train_cfg.get("batch_size", 64),
        max_val_samples=1000,
    )

    print("Building model...")
    model = AttentionLM.from_config(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Pre-generate induction test sequences
    sequences, half_len = make_repeated_sequences(
        vocab_size=model_cfg["vocab_size"], half_len=256, n_sequences=50
    )
    print(f"Induction test: {sequences.shape[0]} sequences, half_len={half_len}")

    def induction_callback(model, step):
        score, first, second = eval_induction(model, sequences, half_len, device)
        print(f"  [step {step}] induction_score={score:.4f}  1st_half={first:.4f}  2nd_half={second:.4f}")
        return {
            "induction_score": score,
            "induction_first_half_loss": first,
            "induction_second_half_loss": second,
        }

    print("Starting training...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        cfg=cfg,
        device=device,
    )
    trainer.eval_callbacks.append(induction_callback)

    trainer.train(
        eval_every=train_cfg.get("eval_every", 5000),
        save_every=train_cfg.get("save_every", 33058),
    )
    print(f"Done. Run dir: {trainer.run_dir}")


if __name__ == "__main__":
    main()
