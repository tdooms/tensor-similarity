#!/usr/bin/env python3
"""Evaluate trained SS models on toy induction task — measuring ACCURACY.

For each model, generates repeated random sequences [s1..sL, s1..sL] using
the SS vocab (4096 tokens) and measures what fraction of second-half tokens
the model predicts correctly (argmax accuracy).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib import AttentionLM
from lib.attention import BilinearAttention
from train_induction_static_norm import BilinearBatchNorm, apply_spectral_norm
from train_ss_technique import WSLinear


def make_repeated_sequences(vocab_size, half_len=256, n_sequences=200, seed=42):
    rng = np.random.RandomState(seed)
    seqs = rng.randint(1, vocab_size, size=(n_sequences, half_len))
    doubled = np.concatenate([seqs, seqs], axis=1)
    return torch.tensor(doubled, dtype=torch.long), half_len


def build_model(technique, cfg, device="cuda"):
    """Build model with the right architecture for a given technique."""
    model = AttentionLM.from_config(cfg)

    if technique == "batchnorm":
        d_model = cfg["model"]["d_model"]
        n_head = cfg["model"]["n_head"]
        n_ctx = cfg["model"]["n_ctx"]
        for i, layer in enumerate(model.layers):
            new_layer = BilinearBatchNorm(
                d_model=d_model, n_head=n_head, n_ctx=n_ctx,
                scale=1.0, use_bias_qkv=True, use_bias_o=True,
            )
            model.layers[i] = new_layer
    elif technique == "specnorm":
        apply_spectral_norm(model)
    elif technique == "weight_std":
        for layer in model.layers:
            for attr in ['q', 'k', 'q2', 'k2']:
                old = getattr(layer, attr)
                new = WSLinear(old.in_features, old.out_features, bias=old.bias is not None)
                new.weight.data.copy_(old.weight.data)
                if old.bias is not None:
                    new.bias.data.copy_(old.bias.data)
                setattr(layer, attr, new)
    elif technique in ("muP_init", "ortho_init"):
        pass  # Just init changes, base architecture is the same
    elif technique == "qknorm":
        pass  # Uses use_rmsnorm_qk=True in config
    else:
        raise ValueError(f"Unknown technique: {technique}")

    return model.to(device)


@torch.no_grad()
def eval_induction_accuracy(model, sequences, half_len, device="cuda", batch_size=32):
    """Compute next-token accuracy on the second half of repeated sequences."""
    model.eval()
    n = sequences.shape[0]

    total_correct = 0
    total_tokens = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = sequences[start:end].to(device)

        logits = model(batch)  # (B, 2L, V)

        # Predictions for second half: positions half_len to 2*half_len-1
        # logits[:, t, :] predicts token at position t+1
        # For second half tokens (positions half_len..2*half_len-1),
        # we use logits[:, half_len-1..2*half_len-2, :]
        # But skip the boundary (position half_len predicting s[half_len] = s[0],
        # which requires "memory" not induction)
        # Induction positions: logits[:, half_len..2*half_len-2, :] predicting tokens at half_len+1..2*half_len-1
        pred_logits = logits[:, half_len:2*half_len-1, :]  # (B, L-1, V)
        targets = batch[:, half_len+1:2*half_len]           # (B, L-1)

        preds = pred_logits.argmax(dim=-1)  # (B, L-1)
        correct = (preds == targets).sum().item()
        total_correct += correct
        total_tokens += targets.numel()

    accuracy = total_correct / total_tokens
    return accuracy


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 4096
    half_len = 256
    n_sequences = 200

    print(f"Generating {n_sequences} repeated random sequences (vocab={vocab_size}, half_len={half_len})...")
    sequences, half_len = make_repeated_sequences(vocab_size, half_len, n_sequences)
    print(f"Sequence shape: {sequences.shape}")

    # Define models to evaluate
    models_to_eval = [
        {
            "name": "ortho_init",
            "technique": "ortho_init",
            "checkpoint": "runs/2026-02-13_091715_ss_1epoch_bilinear_ortho_init/checkpoints/final.pt",
            "use_rmsnorm_qk": False,
        },
        {
            "name": "muP_init",
            "technique": "muP_init",
            "checkpoint": "runs/2026-02-13_063949_ss_1epoch_bilinear_muP_init/checkpoints/final.pt",
            "use_rmsnorm_qk": False,
        },
        {
            "name": "batchnorm",
            "technique": "batchnorm",
            "checkpoint": "runs/2026-02-13_035809_main768_bilinear_12h_3epoch_batchnorm/checkpoints/final.pt",
            "use_rmsnorm_qk": False,
        },
    ]

    # Check for specnorm checkpoint
    import glob
    specnorm_ckpts = sorted(glob.glob("runs/*specnorm*/checkpoints/final.pt"))
    if specnorm_ckpts:
        models_to_eval.append({
            "name": "specnorm",
            "technique": "specnorm",
            "checkpoint": specnorm_ckpts[-1],
            "use_rmsnorm_qk": False,
        })
    else:
        # Use latest step checkpoint
        specnorm_steps = sorted(glob.glob("runs/*specnorm*/checkpoints/step_*.pt"))
        if specnorm_steps:
            models_to_eval.append({
                "name": "specnorm",
                "technique": "specnorm",
                "checkpoint": specnorm_steps[-1],
                "use_rmsnorm_qk": False,
            })

    # Check for weight_std
    ws_ckpts = sorted(glob.glob("runs/*weight_std*/checkpoints/final.pt"))
    if ws_ckpts:
        models_to_eval.append({
            "name": "weight_std",
            "technique": "weight_std",
            "checkpoint": ws_ckpts[-1],
            "use_rmsnorm_qk": False,
        })

    print(f"\nEvaluating {len(models_to_eval)} models on toy induction accuracy:")
    print("=" * 70)

    results = []
    for m in models_to_eval:
        cfg = {
            "model": {
                "vocab_size": vocab_size,
                "n_ctx": 512,
                "d_model": 768,
                "n_head": 12,
                "n_layers": 2,
                "attn_type": "bilinear",
                "attn_scale": 1.0,
                "rope_base": 10000,
                "norm_type": "layernorm",
                "norm_place": "pre_unembed",
                "use_rmsnorm_qk": m["use_rmsnorm_qk"],
                "use_bias_qkv": True,
                "use_bias_o": True,
            },
        }

        print(f"\n--- {m['name']} ---")
        print(f"  Loading: {m['checkpoint']}")

        try:
            model = build_model(m["technique"], cfg, device)
            ckpt = torch.load(m["checkpoint"], map_location=device, weights_only=False)
            state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state)
            step = ckpt.get("step", "?")
            print(f"  Checkpoint step: {step}")

            acc = eval_induction_accuracy(model, sequences, half_len, device)
            print(f"  Induction accuracy: {acc:.4f} ({acc*100:.2f}%)")
            results.append((m["name"], acc, step))

            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((m["name"], None, None))

    print("\n" + "=" * 70)
    print(f"{'Technique':<16} {'Accuracy':>10} {'Step':>8}")
    print("-" * 40)
    for name, acc, step in results:
        if acc is not None:
            print(f"{name:<16} {acc*100:>9.2f}% {step:>8}")
        else:
            print(f"{name:<16} {'ERROR':>10}")

    # Random baseline for vocab_size=4096
    random_acc = 1.0 / vocab_size
    print(f"\nRandom baseline: {random_acc*100:.4f}%")


if __name__ == "__main__":
    main()
