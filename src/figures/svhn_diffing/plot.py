"""SVHN diffing: tracking the digit-9 fine-tune direction through training.

Same 9-stage progressive curriculum as `svhn-forgetting`. We hold a fixed
`M_diff = M_B_ft − M_B_base` (digit-9 fine-tune signal from a held-out
seed) and watch how strongly the progressive model's interaction matrix
projects onto it over training.

Two views per checkpoint:

    slice (class k) — Isserlis cosine of model_A(t)[k] vs M_diff_full[k],
                      for each of the 10 output classes. Class 9 is the
                      on-target signal; all others serve as controls.
    global          — Isserlis cosine of the full 10-class matrix vs
                      M_diff_full, checking that any rise isn't generic.

The diffing signal lives in the class-9 slice during *add 9* and *re-add 9*.
"""
import json

import plotly.graph_objects as go
import polars as pl

from src.figures.style import apply_style, save_figure
from src.figures.svhn_diffing.prepare import CACHE

BG       = "#FFFFFF"
LABEL    = "#0f172a"
MUTED    = "#64748b"
BOUNDARY = "#0f172a"

# Slice/global colors mirror the svhn_forgetting palette family: deep slate
# for the on-target signal, warm rose for the contrastive global control.
SLICE_COLOR  = "#1c3a72"
GLOBAL_COLOR = "#a8453a"

STAGE_DISPLAY = {"repeat": "control"}


def main():
    meta = json.loads((CACHE / "metadata.json").read_text(encoding="utf-8"))
    target_digit = meta["target_digit"]
    spans        = meta["spans"]
    cum_lim      = meta["cum_xlim"]

    progress  = pl.read_ipc(CACHE / "progress.feather")
    per_class = pl.read_ipc(CACHE / "per_class.feather")

    stage_boundaries = [s[0] for s in spans[1:]]
    stage_names      = [STAGE_DISPLAY.get(s[2], s[2]) for s in spans]

    fig = go.Figure()

    # Per-class slice traces: faint for controls, full opacity + thicker for target.
    for digit in range(10):
        s = per_class.filter(pl.col("class_idx") == digit).sort("batch")
        is_target = digit == target_digit
        fig.add_trace(go.Scatter(
            x=s["batch"].to_list(), y=s["value"].to_list(),
            mode="lines",
            line=dict(color=SLICE_COLOR, width=2.4 if is_target else 1.2),
            opacity=1.0 if is_target else 0.25,
            name=f"Tensor slices" if is_target else f"class {digit}",
            showlegend=is_target,
        ))

    # Global trace.
    s = progress.filter(pl.col("metric") == "global").sort("batch")
    fig.add_trace(go.Scatter(
        x=s["batch"].to_list(), y=s["value"].to_list(),
        mode="lines", line=dict(color=GLOBAL_COLOR, width=2.4),
        name="Tensor",
    ))

    # Stage boundary lines.
    for xb in stage_boundaries:
        fig.add_shape(type="line", xref="x", yref="y domain",
                      x0=xb, x1=xb, y0=0, y1=1,
                      line_color=BOUNDARY, line_width=1.0)

    # Stage name labels above the panel (rotated), legend sits above them.
    for x0, x1, name in zip([s[0] for s in spans], [s[1] for s in spans], stage_names):
        fig.add_annotation(x=(x0 + x1) / 2, y=0.905, xref="x", yref="paper",
                           text=name, showarrow=False,
                           xanchor="center", yanchor="bottom",
                           textangle=-90,
                           font=dict(size=12, color=LABEL))

    fig.add_annotation(text="<b>Similarity to diff</b>", xref="paper", yref="paper",
                       x=-0.005, y=0.45,
                       xanchor="right", yanchor="middle",
                       textangle=-90, showarrow=False,
                       font=dict(size=14, color=LABEL))

    tick_kw = dict(showticklabels=True, ticks="outside",
                   tickcolor="#cbd5e1", tickfont=dict(color=MUTED, size=11))
    apply_style(fig, title=None, width=1100, height=420, legend=True)
    fig.update_layout(
        xaxis=dict(domain=[0.06, 0.995], anchor="y", range=cum_lim,
                   showline=False, zeroline=False, showgrid=False, ticks="",
                   tickfont=dict(color=MUTED, size=13),
                   title=dict(text="<b>Cumulative batch</b>",
                              font=dict(size=15, color=LABEL))),
        yaxis=dict(domain=[0.10, 0.80], anchor="x", range=[-1.05, 1.05],
                   showline=False, zeroline=True,
                   zerolinecolor=MUTED, zerolinewidth=1,
                   showgrid=True, gridcolor="#e2e8f0",
                   tickmode="array", tickvals=[-1.0, 0.0, 1.0],
                   ticktext=["−1", "0", "1"], **tick_kw),
        margin=dict(l=88, r=24, t=64, b=58),
        paper_bgcolor=BG, plot_bgcolor=BG,
        legend=dict(orientation="h", yanchor="bottom", y=1.08,
                    xanchor="center", x=0.5,
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(size=13, color=LABEL)),
    )
    save_figure(fig, "svhn_diffing")
