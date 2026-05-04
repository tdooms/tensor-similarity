import matplotlib.pyplot as plt
import numpy as np

from ._utils import save_fig


def _collect(df, prefix, suffix):
    cols = [c for c in df.columns if c.startswith(prefix) and c.endswith(suffix)]
    cols = sorted(cols, key=lambda c: float(c[len(prefix) + 1 : -len(suffix) - 1].replace("m", "-").replace("p", ".")))
    tags = [c[len(prefix) + 1 : -len(suffix) - 1] for c in cols]
    vals = np.stack([df[c].to_numpy() for c in cols], axis=0) if cols else np.zeros((1, len(df)))
    return tags, vals


def make(metrics_df, run_dir):
    metrics_df = metrics_df.sort_values("step").reset_index(drop=True)
    steps = metrics_df["step"].to_numpy() + 1
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, suffix, title in [
        (axes[0, 0], "norm_mse", "OOD-T Norm MSE"),
        (axes[0, 1], "icl_1_4", "OOD-T ICL 1→4"),
        (axes[1, 0], "icl_4_8", "OOD-T ICL 4→8"),
        (axes[1, 1], "pred_abs_magnitude", "OOD-T |pred|"),
    ]:
        tags, vals = _collect(metrics_df, "ood_t", suffix)
        im = ax.imshow(vals, aspect="auto", origin="lower")
        ax.set_yticks(range(len(tags)))
        ax.set_yticklabels(tags, fontsize=7)
        ax.set_xlabel("Step (+1)")
        ax.set_ylabel("log10(g)")
        ax.set_title(title)
        xt = np.linspace(0, max(0, len(steps) - 1), num=min(5, len(steps)), dtype=int)
        ax.set_xticks(xt)
        ax.set_xticklabels([str(int(steps[i])) for i in xt], fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)

    save_fig(fig, run_dir, "ood_task")
