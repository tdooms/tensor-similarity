#!/usr/bin/env python3
"""Plot induction metrics for Run D."""
import json
import matplotlib.pyplot as plt

metrics_path = "runs/2026-02-13_211725_bilinear_bn_nobias_ctx512_d512/metrics.jsonl"

# Parse all metrics
toy_steps, toy_loss_diff, toy_acc = [], [], []
indist_steps, indist_loss_diff, indist_frac_pos = [], [], []
train_steps, train_loss = [], []

with open(metrics_path) as f:
    for line in f:
        d = json.loads(line)
        step = d["step"]
        if "toy_loss_diff" in d:
            toy_steps.append(step)
            toy_loss_diff.append(d["toy_loss_diff"])
            toy_acc.append(d["toy_accuracy"])
        if "indist_loss_diff" in d:
            indist_steps.append(step)
            indist_loss_diff.append(d["indist_loss_diff"])
            indist_frac_pos.append(d["indist_frac_positive"])
        if "train_loss" in d:
            train_steps.append(step)
            train_loss.append(d["train_loss"])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Run D: Bilinear+BN, No Bias, DSIR Pile (d=512, ctx=512)", fontsize=14, fontweight="bold")

# Toy loss diff
ax = axes[0, 0]
ax.plot(toy_steps, toy_loss_diff, "b-o", markersize=3)
ax.set_xlabel("Step")
ax.set_ylabel("Loss Diff (1st half - 2nd half)")
ax.set_title("Toy Repeated Sequences: Loss Difference")
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
ax.grid(True, alpha=0.3)

# In-dist bigram loss diff
ax = axes[0, 1]
ax.plot(indist_steps, indist_loss_diff, "r-o", markersize=3)
ax.set_xlabel("Step")
ax.set_ylabel("Loss Diff (1st - 2nd occurrence)")
ax.set_title("In-Distribution Repeated Bigrams: Loss Difference")
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
ax.grid(True, alpha=0.3)

# Frac positive
ax = axes[1, 0]
ax.plot(indist_steps, indist_frac_pos, "g-o", markersize=3)
ax.set_xlabel("Step")
ax.set_ylabel("Fraction")
ax.set_title("Fraction of Bigrams with Positive Loss Diff")
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
ax.grid(True, alpha=0.3)

# Train loss
ax = axes[1, 1]
# Subsample train loss for readability
step_interval = max(1, len(train_steps) // 500)
ax.plot(train_steps[::step_interval], train_loss[::step_interval], "k-", alpha=0.7, linewidth=0.8)
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.set_title("Training Loss")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("induction_results/run_D_induction_plot.png", dpi=150, bbox_inches="tight")
print(f"Saved: induction_results/run_D_induction_plot.png")
print(f"Toy evals: {len(toy_steps)} points, last step={toy_steps[-1] if toy_steps else 'N/A'}")
print(f"Indist evals: {len(indist_steps)} points, last step={indist_steps[-1] if indist_steps else 'N/A'}")
print(f"Latest toy_loss_diff={toy_loss_diff[-1]:.4f}, indist_loss_diff={indist_loss_diff[-1]:.4f}, frac+={indist_frac_pos[-1]:.3f}")
