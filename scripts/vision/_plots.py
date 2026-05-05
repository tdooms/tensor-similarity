"""Shared plotting utilities for scripts/vision/ experiments."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

FIGSIZE  = (14, 5)
FONTSIZE = 16


def line_plot(series, spans, xlabel, ylabel,
              ylim=None, xlim=None, log_y=False, legend=None,
              figsize=FIGSIZE, fontsize=FONTSIZE):
    """Progressive training line plot.

    series: list of (x, y, plot_kwargs) tuples.
    spans:  list of (x_start, x_end, stage_name) from get_stage_spans().
    Returns (fig, ax).
    """
    fig, ax = plt.subplots(figsize=figsize)
    for x, y, kw in series:
        ax.plot(x, y, **kw)

    transform = blended_transform_factory(ax.transData, ax.transAxes)
    for i, (x0, x1, name) in enumerate(spans):
        if i > 0:
            ax.axvline(x0, color='black', linestyle=':', linewidth=1.5, alpha=0.6)
        ax.text((x0 + x1) / 2, 0.97, name, transform=transform,
                ha='center', va='top', fontsize=fontsize - 2, fontstyle='italic', color='black')

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    if log_y:
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
    else:
        ax.grid(True, alpha=0.3)
    if legend is not None:
        ax.legend(handles=legend, fontsize=fontsize)
    fig.tight_layout()
    return fig, ax


HEATMAP_FIGSIZE = (8, 8)
HEATMAP_FIGSIZE = FIGSIZE


def heatmap_plot(matrix, heatmap_cps, title='',
                 vmin=0, batch_ticks=False, n_batch_ticks=6, linewidth=2.5, figsize=HEATMAP_FIGSIZE, fontsize=14):
    """Heatmap with stage labels on x/y (default) or batch-number ticks.

    Returns (fig, ax).
    """
    N = len(heatmap_cps)

    stage_spans = {}
    for i, cp in enumerate(heatmap_cps):
        stage_spans.setdefault(cp['stage'], []).append(i)
    spans  = [(idxs[0], idxs[-1], name) for name, idxs in stage_spans.items()]
    bounds = [idxs[0] - 0.5 for idxs in stage_spans.values() if idxs[0] > 0]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap='RdYlBu_r', vmin=vmin, vmax=1, aspect='auto')
    cb = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.08)
    cb.set_label(title, fontsize=fontsize)

    if batch_ticks:
        tick_idx    = np.linspace(0, N - 1, n_batch_ticks, dtype=int)
        tick_labels = [str(heatmap_cps[i]['batch']) for i in tick_idx]
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=fontsize - 2)
        ax.set_yticks(tick_idx)
        ax.set_yticklabels(tick_labels, fontsize=fontsize - 2)
        transform = blended_transform_factory(ax.transData, ax.transAxes)
        for i_start, i_end, name in spans:
            ax.text((i_start + i_end) / 2, 1.02, name, transform=transform,
                    ha='center', va='bottom', fontsize=fontsize, fontstyle='italic')
    else:
        stage_mid   = [(i0 + i1) / 2 for i0, i1, _ in spans]
        stage_names = [name for _, _, name in spans]
        ax.set_xticks(stage_mid)
        ax.set_xticklabels(stage_names, rotation=0, ha='center', fontsize=fontsize - 2, fontstyle='italic')
        ax.set_yticks(stage_mid)
        ax.set_yticklabels(stage_names, fontsize=fontsize - 2, fontstyle='italic')

    for b in bounds:
        ax.axvline(b, color='black', linestyle=':', linewidth=linewidth, alpha=0.8)
        ax.axhline(b, color='black', linestyle=':', linewidth=linewidth, alpha=0.8)

    fig.tight_layout()
    return fig, ax


def per_digit_line_plot(digit_arrays, batch, spans, ylabel,
                        ylim=None, figsize=(22, 9), fontsize=13):
    """2×5 grid of line plots, one per digit 0-9.

    digit_arrays: dict {digit: 1D array aligned with batch}.
    spans: from get_stage_spans().
    Returns fig.
    """
    fig, axes = plt.subplots(2, 5, figsize=figsize, sharey=True)
    xlim = (batch[0], batch[-1])
    for d, ax in enumerate(axes.flat):
        ax.plot(batch, digit_arrays[d], color='steelblue', linewidth=1.5)
        ax.set_title(f'digit {d}', fontsize=fontsize, fontstyle='italic')
        ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        for i, (x0, _, _) in enumerate(spans):
            if i > 0:
                ax.axvline(x0, color='black', linestyle=':', linewidth=1, alpha=0.6)
        if d % 5 == 0:
            ax.set_ylabel(ylabel, fontsize=fontsize - 3)
        if d >= 5:
            ax.set_xlabel("Batch", fontsize=fontsize - 3)
    fig.tight_layout()
    return fig


def slice_heatmap_plot(slice_heatmaps, heatmap_cps, label='Tensor slice similarity',
                       figsize=(22, 9), fontsize=13, linewidth=1.5):
    """2×5 grid of per-digit slice similarity heatmaps.

    slice_heatmaps: dict {digit: NxN ndarray} for digits 0-9.
    heatmap_cps:    list of checkpoint dicts with 'stage' key.
    Returns fig.
    """
    stage_spans = {}
    for i, cp in enumerate(heatmap_cps):
        stage_spans.setdefault(cp['stage'], []).append(i)
    stage_mid   = [(idxs[0] + idxs[-1]) / 2 for idxs in stage_spans.values()]
    stage_names = list(stage_spans.keys())
    bounds      = [idxs[0] - 0.5 for idxs in stage_spans.values() if idxs[0] > 0]

    fig, axes = plt.subplots(2, 5, figsize=figsize)
    for d, ax in enumerate(axes.flat):
        im = ax.imshow(slice_heatmaps[d], cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
        ax.set_title(f'digit {d}', fontsize=fontsize, fontstyle='italic')
        ax.set_xticks(stage_mid)
        ax.set_yticks(stage_mid)
        ax.set_xticklabels(stage_names, rotation=45, ha='right', fontsize=fontsize - 6, fontstyle='italic')
        ax.set_yticklabels(stage_names, fontsize=fontsize - 6, fontstyle='italic')
        for b in bounds:
            ax.axvline(b, color='black', linestyle=':', linewidth=linewidth, alpha=0.8)
            ax.axhline(b, color='black', linestyle=':', linewidth=linewidth, alpha=0.8)

    fig.subplots_adjust(bottom=0.18, hspace=0.45, wspace=0.3)
    cax = fig.add_axes([0.15, 0.05, 0.7, 0.025])
    fig.colorbar(im, cax=cax, orientation='horizontal', label=label)
    return fig


def save_show(fig, path, dpi=150, save=True):
    if save:
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.show()
