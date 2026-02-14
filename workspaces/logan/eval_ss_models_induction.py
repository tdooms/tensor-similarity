#!/usr/bin/env python3
"""Evaluate induction capability of pretrained SimpleStories HF models.

Tests each model on the toy repeated-sequence task:
  [16 random tokens][16 random tokens]
Reports 1st half CE, 2nd half CE, loss diff, and accuracy.
"""
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoConfig

DEVICE = "cuda:0"
N_SAMPLES = 100
HALF_LEN = 16


MODELS = [
    # V2 models (vocab=4019, Llama arch)
    "SimpleStories/SimpleStories-V2-1.25M",
    "SimpleStories/SimpleStories-V2-5M",
    "SimpleStories/SimpleStories-V2-11M",
    "SimpleStories/SimpleStories-V2-30M",
    "SimpleStories/SimpleStories-V2-35M",
    # V1 models (vocab=4096, Llama arch)
    "SimpleStories/SimpleStories-1.25M",
    "SimpleStories/SimpleStories-5M",
    "SimpleStories/SimpleStories-11M",
    "SimpleStories/SimpleStories-30M",
    "SimpleStories/SimpleStories-35M",
]


@torch.no_grad()
def eval_toy_repeated(model, vocab_size, device, n_samples=N_SAMPLES, half_len=HALF_LEN):
    """Evaluate on [half_len random tokens] repeated twice."""
    model.eval()
    rng = np.random.RandomState(42)
    seqs = rng.randint(1, vocab_size, size=(n_samples, half_len))
    doubled = np.concatenate([seqs, seqs], axis=1)
    sequences = torch.tensor(doubled, dtype=torch.long)

    batch_size = 32
    all_losses = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = sequences[start:end].to(device)
        logits = model(batch).logits
        shift_logits = logits[:, :-1, :]
        shift_labels = batch[:, 1:]
        B, T, V = shift_logits.shape
        loss = F.cross_entropy(
            shift_logits.reshape(B * T, V), shift_labels.reshape(B * T),
            reduction="none"
        ).reshape(B, T)
        all_losses.append(loss.cpu())
    all_losses = torch.cat(all_losses, dim=0)

    first_half_loss = all_losses[:, :half_len - 1].mean().item()
    second_half_loss = all_losses[:, half_len:].mean().item()

    # Per-position losses for detailed view
    pos_losses = all_losses.mean(dim=0).numpy()

    # Accuracy on second half
    total_correct, total_tokens = 0, 0
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = sequences[start:end].to(device)
        logits = model(batch).logits
        preds = logits[:, half_len:2*half_len-1, :].argmax(dim=-1)
        tgt = batch[:, half_len+1:2*half_len]
        total_correct += (preds == tgt).sum().item()
        total_tokens += tgt.numel()

    return {
        "first_half_ce": first_half_loss,
        "second_half_ce": second_half_loss,
        "loss_diff": first_half_loss - second_half_loss,
        "accuracy": total_correct / total_tokens,
        "pos_losses": pos_losses,
    }


results = {}
for model_id in MODELS:
    print(f"\nLoading {model_id}...")
    try:
        config = AutoConfig.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())

        r = eval_toy_repeated(model, config.vocab_size, DEVICE)
        r["n_params"] = n_params
        r["vocab_size"] = config.vocab_size
        r["hidden_size"] = config.hidden_size
        r["n_layers"] = config.num_hidden_layers
        results[model_id] = r

        print(f"  {n_params/1e6:.1f}M params | d={config.hidden_size} L={config.num_hidden_layers}")
        print(f"  1st half CE: {r['first_half_ce']:.4f}")
        print(f"  2nd half CE: {r['second_half_ce']:.4f}")
        print(f"  Loss diff:   {r['loss_diff']:.4f}")
        print(f"  Accuracy:    {r['accuracy']:.4f}")

        del model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  FAILED: {e}")

# Save raw results (without numpy arrays)
save_results = {}
for k, v in results.items():
    save_results[k] = {kk: vv for kk, vv in v.items() if kk != "pos_losses"}
with open("induction_results/ss_models_induction.json", "w") as f:
    json.dump(save_results, f, indent=2)

# =====================================================================
# Plotting
# =====================================================================

# Separate V1 and V2
v2_models = [m for m in MODELS if "V2" in m and m in results]
v1_models = [m for m in MODELS if "V2" not in m and m in results]

def short_name(model_id):
    return model_id.split("/")[1]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("SimpleStories Pretrained Models: Toy Induction Eval\n([16 random tokens] repeated x2, 100 samples)", fontsize=13, fontweight="bold")

all_models = v2_models + v1_models
names = [short_name(m) for m in all_models]
params = [results[m]["n_params"] / 1e6 for m in all_models]
colors = ["#2196F3"] * len(v2_models) + ["#FF9800"] * len(v1_models)

# Bar chart: 1st and 2nd half CE
ax = axes[0, 0]
x = np.arange(len(all_models))
w = 0.35
first_ces = [results[m]["first_half_ce"] for m in all_models]
second_ces = [results[m]["second_half_ce"] for m in all_models]
bars1 = ax.bar(x - w/2, first_ces, w, label="1st half CE", color="cyan", edgecolor="gray")
bars2 = ax.bar(x + w/2, second_ces, w, label="2nd half CE", color="magenta", edgecolor="gray")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Cross-Entropy Loss")
ax.set_title("1st vs 2nd Half CE")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

# Bar chart: Loss diff
ax = axes[0, 1]
diffs = [results[m]["loss_diff"] for m in all_models]
ax.bar(x, diffs, color=colors, edgecolor="gray")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Loss Diff (1st - 2nd)")
ax.set_title("Toy Loss Difference")
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
ax.grid(True, alpha=0.3, axis="y")

# Bar chart: Accuracy
ax = axes[1, 0]
accs = [results[m]["accuracy"] for m in all_models]
ax.bar(x, accs, color=colors, edgecolor="gray")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Accuracy")
ax.set_title("2nd Half Token Accuracy")
ax.grid(True, alpha=0.3, axis="y")

# Per-position loss curves for all models
ax = axes[1, 1]
for i, m in enumerate(all_models):
    r = results[m]
    pos = r["pos_losses"]
    label = short_name(m)
    ax.plot(range(len(pos)), pos, label=label, alpha=0.8, linewidth=1.5)
ax.axvline(x=HALF_LEN - 1, color="red", linestyle="--", alpha=0.5, label="repeat starts")
ax.set_xlabel("Position")
ax.set_ylabel("CE Loss")
ax.set_title("Per-Position Loss (avg over 100 samples)")
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("induction_results/ss_models_induction.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: induction_results/ss_models_induction.png")
