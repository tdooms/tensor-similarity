import matplotlib.pyplot as plt

from ._utils import save_fig


def make(metrics_df, run_dir):
    metrics_df = metrics_df.sort_values("step").reset_index(drop=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    x = metrics_df["step"].to_numpy() + 1

    axes[0, 0].plot(x, metrics_df["rav"])
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("Step (+1)")
    axes[0, 0].set_ylabel("Fraction")
    axes[0, 0].set_title("RAV")

    axes[0, 1].plot(x, metrics_df["erank"])
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("Step (+1)")
    axes[0, 1].set_ylabel("Rank")
    axes[0, 1].set_title("Effective rank")

    axes[1, 0].plot(x, metrics_df["rav_num"])
    axes[1, 0].set_xlabel("Step (+1)")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].set_title("RAV numerator")

    axes[1, 1].plot(x, metrics_df["rav_den"])
    axes[1, 1].set_xlabel("Step (+1)")
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].set_title("RAV denominator")

    save_fig(fig, run_dir, "residual")
