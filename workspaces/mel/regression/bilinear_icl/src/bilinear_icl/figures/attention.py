import matplotlib.pyplot as plt
import numpy as np

from ._utils import save_fig


def _matrix(df, suffix):
    cols = [c for c in df.columns if c.startswith("attn_L") and c.endswith(suffix)]
    vals = np.stack([df[c].to_numpy() for c in cols], axis=0) if cols else np.zeros((1, len(df)))
    return cols, vals


def make(metrics_df, run_dir):
    metrics_df = metrics_df.sort_values("step").reset_index(drop=True)
    steps = metrics_df["step"].to_numpy() + 1
    eps = 1e-12

    x_cols = [c for c in metrics_df.columns if c.startswith("attn_L") and c.endswith("_total_x")]
    for c in x_cols:
        y_col = c.replace("_total_x", "_total_y")
        if y_col in metrics_df:
            out_col = c.replace("_total_x", "_x_mass_frac")
            denom = metrics_df[c] + metrics_df[y_col] + eps
            metrics_df[out_col] = metrics_df[c] / denom

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    specs = [
        (axes[0, 0], "_entropy_norm", "Entropy (normalized)"),
        (axes[0, 1], "_variability", "Variability"),
        (axes[0, 2], "_x_mass_frac", "X mass fraction"),
        (axes[1, 0], "_prev_x", "Prev-X mass"),
        (axes[1, 1], "_prev_y", "Prev-Y mass"),
    ]

    axes[1, 2].axis("off")

    for ax, suffix, title in specs:
        labels, vals = _matrix(metrics_df, suffix)
        im = ax.imshow(vals, aspect="auto", origin="lower")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("Step (+1)")
        ax.set_ylabel("Head (layer, head)")
        ax.set_title(title)
        xt = np.linspace(0, max(0, len(steps) - 1), num=min(5, len(steps)), dtype=int)
        ax.set_xticks(xt)
        ax.set_xticklabels([str(int(steps[i])) for i in xt], fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)

    save_fig(fig, run_dir, "attention")
