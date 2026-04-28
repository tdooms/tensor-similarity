"""Grokking summary: train/val accuracy + loss + TN similarity + frequency marginals.

Six DP-optimal phases (start → memorize start → memorize end → grok start →
grok end → consolidate) are rendered as subtle background fills on the line
plots and as dotted vertical lines through the heatmaps. Train/val are drawn
in a single dark hue, distinguished by line style (solid / dashed). The
heatmaps use a neutral grayscale to keep attention on the phase structure
rather than a rainbow gradient.
"""
import json
import math

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from src.figures.grokking.prepare import CACHE
from src.figures.style import apply_style, save_figure

LINE_COLOR = "#0F172A"

PHASE_FILLS_LINE = {
    "start":       "rgba(241,245,249,0.7)",   # slate-100
    "memorize":    "rgba(219,234,254,0.65)",  # blue-100
    "grok":        "rgba(254,215,170,0.6)",   # orange-200
    "consolidate": "rgba(204,251,241,0.65)",  # teal-100
}
PHASE_FILLS_HEAT = {
    "start":       "rgba(100,116,139,0.18)",  # slate-500
    "memorize":    "rgba(59,130,246,0.18)",   # blue-500
    "grok":        "rgba(249,115,22,0.18)",   # orange-500
    "consolidate": "rgba(20,184,166,0.18)",   # teal-500
}


def main():
    meta = json.loads((CACHE / "metadata.json").read_text(encoding="utf-8"))
    steps = [s for s in meta["steps"] if s > 0]
    n = len(steps)
    phases = meta["phases"]

    history = pl.read_ipc(CACHE / "history.feather").filter(pl.col("step") > 0)
    matrix = (pl.read_ipc(CACHE / "similarity.feather")
                .filter((pl.col("metric") == "tn_similarity")
                        & (pl.col("step_i") > 0) & (pl.col("step_j") > 0))
                .sort(["step_i", "step_j"])["value"]
                .to_numpy().reshape(n, n).tolist())
    freq = (pl.read_ipc(CACHE / "freq_marginals.feather")
              .filter(pl.col("step") > 0)
              .with_columns(value=pl.col("value") / pl.col("value").sum().over("step"))
              .sort(["freq_idx", "step"]))
    n_freqs = freq["freq_idx"].n_unique()
    freq_z = freq["value"].to_numpy().reshape(n_freqs, n).tolist()

    log_x = [math.log10(s) for s in steps]
    log_range = [log_x[0] - (log_x[1] - log_x[0]) / 2,
                 log_x[-1] + (log_x[-1] - log_x[-2]) / 2]

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        row_heights=[0.13, 0.13, 0.50, 0.24],
                        vertical_spacing=0.04)

    # Cell-edge phase boundaries: geometric mean of the last step of phase i
    # and the first step of phase i+1 (so the band edge falls between cells,
    # not through the middle of one). The first phase extends to the left
    # edge of the x axis; the last phase to the right edge.
    axis_lo, axis_hi = 10 ** log_range[0], 10 ** log_range[1]
    bounds = []
    left = axis_lo
    for i, phase in enumerate(phases):
        right = (math.sqrt(phase["last_step"] * phases[i + 1]["first_step"])
                 if i < len(phases) - 1 else axis_hi)
        bounds.append((left, right))
        left = right

    # Named phases only: colored band in the gap between row 1 and row 2 (so
    # the line panels themselves stay clean) + low-opacity tints on each
    # heatmap. Unnamed middle phase stays white.
    for (x0, x1), phase in zip(bounds, phases):
        if not phase["name"]:
            continue
        fig.add_shape(type="rect", xref="x", yref="paper",
                      x0=x0, x1=x1, y0=0.74, y1=0.79,
                      fillcolor=PHASE_FILLS_LINE[phase["name"]], line_width=0, layer="below")
        for ax in ("3", "4"):
            fig.add_shape(type="rect", xref=f"x{ax}", yref=f"y{ax} domain",
                          x0=x0, x1=x1, y0=0, y1=1,
                          fillcolor=PHASE_FILLS_HEAT[phase["name"]], line_width=0, layer="above")
    train_acc = history.filter(pl.col("metric") == "train_acc").sort("step")
    val_acc = history.filter(pl.col("metric") == "val_acc").sort("step")
    fig.add_trace(go.Scatter(x=train_acc["step"], y=train_acc["value"], mode="lines+markers",
                             name="Train", legendgroup="train",
                             line=dict(color=LINE_COLOR, width=2.5),
                             marker=dict(size=5, color=LINE_COLOR)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=val_acc["step"], y=val_acc["value"], mode="lines+markers",
                             name="Validation", legendgroup="val",
                             line=dict(color=LINE_COLOR, width=2.5, dash="dash"),
                             marker=dict(size=5, color=LINE_COLOR, symbol="diamond-open")),
                  row=1, col=1)

    train_loss = history.filter(pl.col("metric") == "train_loss").sort("step")
    val_loss = history.filter(pl.col("metric") == "val_loss").sort("step")
    fig.add_trace(go.Scatter(x=train_loss["step"], y=train_loss["value"], mode="lines+markers",
                             name="Train", legendgroup="train", showlegend=False,
                             line=dict(color=LINE_COLOR, width=2.5),
                             marker=dict(size=5, color=LINE_COLOR)),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=val_loss["step"], y=val_loss["value"], mode="lines+markers",
                             name="Validation", legendgroup="val", showlegend=False,
                             line=dict(color=LINE_COLOR, width=2.5, dash="dash"),
                             marker=dict(size=5, color=LINE_COLOR, symbol="diamond-open")),
                  row=2, col=1)

    fig.add_trace(go.Heatmap(z=matrix, x=steps, y=steps, zmin=0, zmax=1,
                             colorscale="Greys",
                             colorbar=dict(y=0.385, len=0.4, thickness=12, xpad=10)),
                  row=3, col=1)

    fig.add_trace(go.Heatmap(z=freq_z, x=steps, y=list(range(n_freqs)),
                             colorscale="Greys",
                             colorbar=dict(y=0.085, len=0.18, thickness=12, xpad=10)),
                  row=4, col=1)

    plot_left = 90 / 900
    plot_right = 1 - 120 / 900
    span = plot_right - plot_left
    log_span = log_range[1] - log_range[0]
    for (x0, x1), phase in zip(bounds, phases):
        if not phase["name"]:
            continue
        log_mid = (math.log10(max(x0, axis_lo)) + math.log10(x1)) / 2
        x_paper = plot_left + (log_mid - log_range[0]) / log_span * span
        fig.add_annotation(x=x_paper, y=0.766, xref="paper", yref="paper",
                           xanchor="center", yanchor="middle",
                           text=f"<b>{phase['name']}</b>", showarrow=False,
                           font=dict(size=12, color="#334155"))

    for row in (1, 2, 3, 4):
        fig.update_xaxes(type="log", range=log_range, row=row, col=1)
    fig.update_xaxes(title="<b>Training step</b>", row=4, col=1)
    fig.update_yaxes(automargin=False, title_standoff=14)
    fig.update_yaxes(title="<b>Accuracy</b>", range=[0, 1.05],
                     domain=[0.80, 0.91], row=1, col=1)
    fig.update_yaxes(type="log", title="<b>Loss</b>",
                     domain=[0.62, 0.73], row=2, col=1)
    fig.update_yaxes(type="log", range=log_range, title="<b>Step</b>",
                     domain=[0.18, 0.58], row=3, col=1)
    fig.update_yaxes(title="<b>Frequency</b>",
                     domain=[0.00, 0.16], row=4, col=1)

    apply_style(fig, title=f"Grokking on modular addition (P={meta['config']['P']})",
                width=900, height=1620)
    fig.update_layout(
        margin=dict(l=90, r=120, t=160, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=0.985,
                    xanchor="center", x=0.5,
                    bgcolor="rgba(255,255,255,0.85)"),
    )
    save_figure(fig, "grokking")
