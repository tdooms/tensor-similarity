#!/usr/bin/env python3
"""Plot induction metrics for Run D and F with 1st/2nd half CE shown."""
import json
import matplotlib.pyplot as plt


def load_metrics(path):
    toy_steps, toy_first, toy_second, toy_diff = [], [], [], []
    indist_steps, indist_first, indist_second, indist_diff, indist_frac = [], [], [], [], []
    train_steps, train_loss = [], []

    with open(path) as f:
        for line in f:
            d = json.loads(line)
            step = d["step"]
            if "toy_loss_diff" in d:
                toy_steps.append(step)
                toy_first.append(d["toy_first_loss"])
                toy_second.append(d["toy_second_loss"])
                toy_diff.append(d["toy_loss_diff"])
            if "indist_loss_diff" in d:
                indist_steps.append(step)
                indist_first.append(d["indist_first_loss"])
                indist_second.append(d["indist_second_loss"])
                indist_diff.append(d["indist_loss_diff"])
                indist_frac.append(d["indist_frac_positive"])
            if "train_loss" in d:
                train_steps.append(step)
                train_loss.append(d["train_loss"])

    return dict(
        toy_steps=toy_steps, toy_first=toy_first, toy_second=toy_second, toy_diff=toy_diff,
        indist_steps=indist_steps, indist_first=indist_first, indist_second=indist_second,
        indist_diff=indist_diff, indist_frac=indist_frac,
        train_steps=train_steps, train_loss=train_loss,
    )


def plot_run(m, title, out_path):
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Row 0: 1st / 2nd half CE for toy and indist
    ax = axes[0, 0]
    ax.plot(m["toy_steps"], m["toy_first"], "c-o", markersize=2, label="1st half CE")
    ax.plot(m["toy_steps"], m["toy_second"], "m-o", markersize=2, label="2nd half CE")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Toy Repeated: 1st vs 2nd Half CE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(m["indist_steps"], m["indist_first"], "c-o", markersize=2, label="1st occurrence CE")
    ax.plot(m["indist_steps"], m["indist_second"], "m-o", markersize=2, label="2nd occurrence CE")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("In-Dist Bigrams: 1st vs 2nd Occurrence CE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 1: Loss diffs
    ax = axes[1, 0]
    ax.plot(m["toy_steps"], m["toy_diff"], "b-o", markersize=3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss Diff (1st - 2nd)")
    ax.set_title("Toy Repeated: Loss Difference")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(m["indist_steps"], m["indist_diff"], "r-o", markersize=3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss Diff (1st - 2nd)")
    ax.set_title("In-Dist Bigrams: Loss Difference")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Row 2: Frac positive + train loss
    ax = axes[2, 0]
    ax.plot(m["indist_steps"], m["indist_frac"], "g-o", markersize=3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Fraction")
    ax.set_title("Fraction of Bigrams with Positive Loss Diff")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    step_interval = max(1, len(m["train_steps"]) // 500)
    ax.plot(m["train_steps"][::step_interval], m["train_loss"][::step_interval], "k-", alpha=0.7, linewidth=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


# Run D
m = load_metrics("runs/2026-02-13_211725_bilinear_bn_nobias_ctx512_d512/metrics.jsonl")
plot_run(m, "Run D: Bilinear+BN, No Bias, DSIR Pile (d=512, ctx=512)", "induction_results/run_D_induction_detailed.png")
print(f"  Last toy: 1st={m['toy_first'][-1]:.4f}  2nd={m['toy_second'][-1]:.4f}  diff={m['toy_diff'][-1]:.4f}")
print(f"  Last indist: 1st={m['indist_first'][-1]:.4f}  2nd={m['indist_second'][-1]:.4f}  diff={m['indist_diff'][-1]:.4f}")

# Run F
m = load_metrics("runs/2026-02-13_221438_bilinear_bn_nobias_ss_ctx512_d512/metrics.jsonl")
plot_run(m, "Run F: Bilinear+BN, No Bias, SimpleStories (d=512, ctx=512)", "induction_results/run_F_induction_detailed.png")
print(f"  Last toy: 1st={m['toy_first'][-1]:.4f}  2nd={m['toy_second'][-1]:.4f}  diff={m['toy_diff'][-1]:.4f}")
print(f"  Last indist: 1st={m['indist_first'][-1]:.4f}  2nd={m['indist_second'][-1]:.4f}  diff={m['indist_diff'][-1]:.4f}")
