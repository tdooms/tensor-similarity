"""Grokking summary on a checkpoint-INDEX axis (not log-step).

Each checkpoint occupies one unit on the x axis, so the heatmap cells are
uniform 1×1 squares (Logan's aesthetic). The bottom axis labels show step
values at hand-picked sparse indices instead of plotly's automatic log
ticks. Phase regions and labels are computed in index space — boundaries
fall on cell edges (i + 0.5).

Five DP-optimal phases (`start → memorize → [stuck plateau] → grok →
consolidate`); the unnamed middle segment stays uncolored so the gap
between memorize and grok reads as the plateau where train acc has
saturated but val hasn't yet. Train/val drawn in a single dark hue,
solid for train, dotted for validation. Heatmaps grayscale; phase color
carries through every panel via the same RGB at different opacity.
"""
import json

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from src.figures.grokking_similarity.prepare import CACHE
from src.figures.style import apply_style, save_figure

LINE_COLOR = "#0F172A"

PHASE_RGB = {
    "start":       "148,163,184",  # slate-400
    "memorize":    "96,165,250",   # blue-400
    "grok":        "251,146,60",   # orange-400
    "consolidate": "45,212,191",   # teal-400
}
PHASE_LABEL_COLOR = {
    "start":       "#475569",  # slate-600
    "memorize":    "#2563eb",  # blue-600
    "grok":        "#ea580c",  # orange-600
    "consolidate": "#0d9488",  # teal-600
}
PHASE_FILLS_LINE = {k: f"rgba({rgb},0.35)" for k, rgb in PHASE_RGB.items()}
PHASE_FILLS_HEAT = {k: f"rgba({rgb},0.45)" for k, rgb in PHASE_RGB.items()}

# Heatmap colorscale: transparent at value 0, opaque black at value 1.
# The tinted phase rectangles sit BEHIND the heatmap and show through
# wherever cell value is low. High-similarity cells render as solid
# black with no phase tint mixing in — so the diagonal stays pure 1.
HEAT_COLORSCALE = [[0, "rgba(15,23,42,0)"], [1, "rgba(15,23,42,1)"]]

# Sparse decade ticks. Each target is snapped to the nearest actual
# checkpoint index, but labeled with the clean decade value (so the
# rendered label reads "10k" even when the closest checkpoint is at
# step 9212 etc).
TICK_TARGETS = (1, 10, 100, 1_000, 10_000, 100_000)
TICK_LABELS = ("1", "10", "100", "1k", "10k", "100k")


def main():
    meta = json.loads((CACHE / "metadata.json").read_text(encoding="utf-8"))
    steps = [s for s in meta["steps"] if s > 0]
    n = len(steps)
    step_to_idx = {s: i for i, s in enumerate(steps)}
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

    # Phase index spans (skip the dropped step 0 by clamping at 0).
    phase_idx = [(step_to_idx.get(p["first_step"], 0),
                  step_to_idx[p["last_step"]],
                  p["name"])
                 for p in phases]

    # Cell-edge phase boundaries in index space: midpoint between the last
    # index of phase i and the first index of phase i+1. First phase extends
    # to -0.5 (left cell edge); last phase to n-0.5.
    bounds = []
    left = -0.5
    for i, (_, last, _) in enumerate(phase_idx):
        right = (last + phase_idx[i + 1][0]) / 2 if i < len(phase_idx) - 1 else n - 0.5
        bounds.append((left, right))
        left = right

    indices = list(range(n))
    indices_freq = list(range(n_freqs))

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.012)

    for (x0, x1), (_, _, name) in zip(bounds, phase_idx):
        if not name:
            continue
        line_fill = PHASE_FILLS_LINE[name]
        heat_fill = PHASE_FILLS_HEAT[name]
        for ax in ("", "2"):
            fig.add_shape(type="rect", xref=f"x{ax}", yref=f"y{ax} domain",
                          x0=x0, x1=x1, y0=0, y1=1,
                          fillcolor=line_fill, line_width=0, layer="below")
        for ax in ("3", "4"):
            fig.add_shape(type="rect", xref=f"x{ax}", yref=f"y{ax} domain",
                          x0=x0, x1=x1, y0=0, y1=1,
                          fillcolor=heat_fill, line_width=0, layer="below")

    for metric, name, dash, show in (("train_acc", "Train", None, True),
                                     ("val_acc",   "Validation", "dot", True)):
        s = history.filter(pl.col("metric") == metric).sort("step")
        x = [step_to_idx[step] for step in s["step"].to_list()]
        fig.add_trace(go.Scatter(x=x, y=s["value"].to_list(), mode="lines+markers",
                                 name=name, legendgroup=name.lower(), showlegend=show,
                                 line=dict(color=LINE_COLOR, width=2.5, dash=dash),
                                 marker=dict(size=5, color=LINE_COLOR)),
                      row=1, col=1)

    for metric, name, dash in (("train_loss", "Train", None),
                               ("val_loss",   "Validation", "dot")):
        s = history.filter(pl.col("metric") == metric).sort("step")
        x = [step_to_idx[step] for step in s["step"].to_list()]
        fig.add_trace(go.Scatter(x=x, y=s["value"].to_list(), mode="lines+markers",
                                 name=name, legendgroup=name.lower(), showlegend=False,
                                 line=dict(color=LINE_COLOR, width=2.5, dash=dash),
                                 marker=dict(size=5, color=LINE_COLOR)),
                      row=2, col=1)

    fig.add_trace(go.Heatmap(z=matrix, x=indices, y=indices, zmin=0, zmax=1,
                             colorscale=HEAT_COLORSCALE, showscale=False),
                  row=3, col=1)

    fig.add_trace(go.Heatmap(z=freq_z, x=indices, y=indices_freq,
                             colorscale=HEAT_COLORSCALE, showscale=False),
                  row=4, col=1)

    # Phase labels above row 1 — paper x = (mid_idx + 0.5) / n since the
    # xaxis range is [-0.5, n-0.5] and the xaxis domain spans paper [0, 1].
    for first, last, name in phase_idx:
        if not name:
            continue
        x_paper = ((first + last) / 2 + 0.5) / n
        fig.add_annotation(x=x_paper, y=0.925, xref="paper", yref="paper",
                           xanchor="center", yanchor="middle",
                           text=f"<b>{name}</b>", showarrow=False,
                           font=dict(size=14, color=PHASE_LABEL_COLOR[name]))

    # Snap each target to the nearest checkpoint index; keep the clean
    # decade label regardless of how close the actual checkpoint is.
    tick_indices = [min(range(n), key=lambda i: abs(steps[i] - t)) for t in TICK_TARGETS]

    for row in (1, 2, 3, 4):
        fig.update_xaxes(range=[-0.5, n - 0.5], showgrid=False, row=row, col=1)
    fig.update_xaxes(tickmode="array", tickvals=tick_indices, ticktext=list(TICK_LABELS),
                     title="<b>Training step</b>", row=4, col=1)

    fig.update_xaxes(showline=False, zeroline=False, mirror=False)
    fig.update_yaxes(automargin=False, title=None,
                     showline=False, zeroline=False, mirror=False)
    fig.update_yaxes(range=[0, 1.05], domain=[0.762, 0.91], row=1, col=1)
    fig.update_yaxes(type="log", domain=[0.602, 0.75], row=2, col=1)
    fig.update_yaxes(range=[-0.5, n - 0.5], domain=[0.249, 0.59],
                     showticklabels=False, ticks="", showgrid=False, row=3, col=1)
    fig.update_yaxes(domain=[0.01, 0.237], showgrid=False, row=4, col=1)
    fig.update_xaxes(ticks="", showgrid=False, row=3, col=1)
    fig.update_xaxes(ticks="", showgrid=False, row=4, col=1)

    for label, mid_y in (("Accuracy",   0.836),
                         ("Loss",       0.676),
                         ("Similarity", 0.4195),
                         ("Frequency",  0.1235)):
        fig.add_annotation(text=f"<b>{label}</b>", xref="paper", yref="paper",
                           x=-0.07, y=mid_y, xanchor="center", yanchor="middle",
                           textangle=-90, showarrow=False,
                           font=dict(size=18, color="#0F172A"))

    apply_style(fig, title=f"Grokking on modular addition (P={meta['config']['P']})",
                width=900, height=880)
    fig.update_layout(
        margin=dict(l=85, r=30, t=40, b=8),
        title=dict(y=0.978),
        legend=dict(orientation="h", yanchor="middle", y=0.95,
                    xanchor="center", x=0.5,
                    bgcolor="rgba(255,255,255,0)",
                    font=dict(size=14)),
    )
    save_figure(fig, "grokking_similarity")
