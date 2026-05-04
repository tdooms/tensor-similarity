import matplotlib.pyplot as plt

from ._utils import save_fig


def make(metrics_df, run_dir):
    metrics_df = metrics_df.sort_values("step").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = metrics_df["step"].to_numpy() + 1
    cols = [c for c in metrics_df.columns if c.startswith("embed_sv_")]
    for c in cols:
        ax.plot(x, metrics_df[c], label=c)
    ax.set_xscale("log")
    ax.set_title("Embedding singular values")
    ax.set_xlabel("Step (+1)")
    ax.set_ylabel("Singular value")
    ax.legend(fontsize=6, ncol=2)
    save_fig(fig, run_dir, "embedding")
