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

# One base color per phase; same RGB used in both contexts (line panel
# background AND heatmap tint), just different alpha — so the colors
# read as identical across panels.
PHASE_RGB = {
    "start":       "148,163,184",  # slate-400
    "memorize":    "96,165,250",   # blue-400
    "grok":        "251,146,60",   # orange-400
    "consolidate": "45,212,191",   # teal-400
}
PHASE_FILLS_LINE = {k: f"rgba({rgb},0.35)" for k, rgb in PHASE_RGB.items()}
PHASE_FILLS_HEAT = {k: f"rgba({rgb},0.30)" for k, rgb in PHASE_RGB.items()}


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

    # Named phases only: colored fill behind row 1 (acc) and row 2 (loss)
    # panels separately — the gap between them stays white so the labels
    # have a clean uncolored band to live in. Heatmaps get low-opacity
    # tints. Unnamed middle phase stays white throughout.
    for (x0, x1), phase in zip(bounds, phases):
        if not phase["name"]:
            continue
        line_fill = PHASE_FILLS_LINE[phase["name"]]
        heat_fill = PHASE_FILLS_HEAT[phase["name"]]
        for ax in ("", "2"):
            fig.add_shape(type="rect", xref=f"x{ax}", yref=f"y{ax} domain",
                          x0=x0, x1=x1, y0=0, y1=1,
                          fillcolor=line_fill, line_width=0, layer="below")
        for ax in ("3", "4"):
            fig.add_shape(type="rect", xref=f"x{ax}", yref=f"y{ax} domain",
                          x0=x0, x1=x1, y0=0, y1=1,
                          fillcolor=heat_fill, line_width=0, layer="above")
    train_acc = history.filter(pl.col("metric") == "train_acc").sort("step")
    val_acc = history.filter(pl.col("metric") == "val_acc").sort("step")
    fig.add_trace(go.Scatter(x=train_acc["step"], y=train_acc["value"], mode="lines+markers",
                             name="Train", legendgroup="train",
                             line=dict(color=LINE_COLOR, width=2.5),
                             marker=dict(size=5, color=LINE_COLOR)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=val_acc["step"], y=val_acc["value"], mode="lines+markers",
                             name="Validation", legendgroup="val",
                             line=dict(color=LINE_COLOR, width=2.5, dash="dot"),
                             marker=dict(size=5, color=LINE_COLOR)),
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
                             line=dict(color=LINE_COLOR, width=2.5, dash="dot"),
                             marker=dict(size=5, color=LINE_COLOR)),
                  row=2, col=1)

    fig.add_trace(go.Heatmap(z=matrix, x=steps, y=steps, zmin=0, zmax=1,
                             colorscale="Greys",
                             colorbar=dict(y=0.435, len=0.45, thickness=12, xpad=10)),
                  row=3, col=1)

    fig.add_trace(go.Heatmap(z=freq_z, x=steps, y=list(range(n_freqs)),
                             colorscale="Greys",
                             colorbar=dict(y=0.10, len=0.16, thickness=12, xpad=10)),
                  row=4, col=1)

    # Plotly "paper" x runs 0..1 across the plot area (between margins), and
    # the shared xaxis spans that whole domain. So a data value v maps to
    # paper x = (log10(v) - log_lo) / log_span — no margin offsets.
    log_span = log_range[1] - log_range[0]
    for phase in phases:
        if not phase["name"]:
            continue
        log_mid = (math.log10(max(phase["first_step"], steps[0])) +
                   math.log10(phase["last_step"])) / 2
        x_paper = (log_mid - log_range[0]) / log_span
        fig.add_annotation(x=x_paper, y=0.84, xref="paper", yref="paper",
                           xanchor="center", yanchor="middle",
                           text=f"<b>{phase['name']}</b>", showarrow=False,
                           font=dict(size=12, color="#334155"))

    for row in (1, 2, 3, 4):
        fig.update_xaxes(type="log", range=log_range, row=row, col=1)
    fig.update_xaxes(title="<b>Training step</b>", row=4, col=1)
    fig.update_yaxes(automargin=False, title=None)
    fig.update_yaxes(range=[0, 1.05], domain=[0.85, 0.95], row=1, col=1)
    fig.update_yaxes(type="log", domain=[0.71, 0.81], row=2, col=1)
    fig.update_yaxes(type="log", range=log_range, domain=[0.22, 0.68], row=3, col=1)
    fig.update_yaxes(domain=[0.03, 0.19], row=4, col=1)

    # Y-axis titles as fixed paper-x annotations so they align across rows
    # regardless of tick-label widths.
    for label, mid_y in (("Accuracy",  0.90),
                         ("Loss",      0.76),
                         ("Step",      0.45),
                         ("Frequency", 0.11)):
        fig.add_annotation(text=f"<b>{label}</b>", xref="paper", yref="paper",
                           x=-0.06, y=mid_y, xanchor="center", yanchor="middle",
                           textangle=-90, showarrow=False,
                           font=dict(size=14, color="#0F172A"))

    apply_style(fig, title=f"Grokking on modular addition (P={meta['config']['P']})",
                width=900, height=1620, legend=False)
    fig.update_layout(
        margin=dict(l=90, r=120, t=50, b=60),
        title=dict(y=0.985),
    )
    save_figure(fig, "grokking")
