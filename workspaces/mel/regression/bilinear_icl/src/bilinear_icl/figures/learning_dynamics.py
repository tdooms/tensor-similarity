import matplotlib.pyplot as plt

from ._utils import save_fig


def make(metrics_df, run_dir):
    metrics_df = metrics_df.sort_values("step").reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    x = metrics_df["step"].to_numpy() + 1

    axes[0].plot(x, metrics_df["test_loss"], label="mean test loss")
    label_map = {0: "ℓ1", 2: "ℓ3", 4: "ℓ5", 6: "ℓ7"}
    for k in (0, 2, 4, 6):
        c = f"loss_pos_{k}"
        if c in metrics_df:
            axes[0].plot(x, metrics_df[c], label=label_map[k])
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Step (+1)")
    axes[0].set_ylabel("MSE")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Test Loss")

    axes[1].plot(x, metrics_df["icl_1_4"], label="ICL 1→4")
    axes[1].plot(x, metrics_df["icl_4_8"], label="ICL 4→8")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Step (+1)")
    axes[1].set_ylabel("Loss difference")
    axes[1].legend(fontsize=7)
    axes[1].set_title("ICL Deltas")

    axes[2].plot(x, metrics_df["pred_sq_magnitude"], label="E[y_hat^2]")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Step (+1)")
    axes[2].set_ylabel("Magnitude")
    axes[2].legend(fontsize=7)
    axes[2].set_title("Prediction Magnitude")

    save_fig(fig, run_dir, "learning_dynamics")
