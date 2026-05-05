"""SVHN diffing: tracking the digit-9 fine-tune direction through training.

Same 9-stage progressive curriculum as `svhn-forgetting`. We hold a fixed
`M_diff = M_B_ft − M_B_base` (digit-9 fine-tune signal from a held-out
seed) and watch how strongly the progressive model's interaction matrix
projects onto it over training.

Two views:

    progress  — Gaussian cosine of model_A(t) vs M_diff, both as the
                full 10-class interaction matrix (`global`) and restricted
                to the digit-9 slice (`slice`). The slice is the on-target
                signal; global lets us check it isn't a generic alignment
                of every output direction.
    heatmap   — N×N pairwise digit-9 slice similarity across the same
                checkpoints, showing block structure across stages.

The "diffing" signal lives in *slice* during *add 9* and *re-add 9*: the
projection rises sharply when the model first learns digit 9 and again when
it's re-introduced after removal, while *global* stays flat throughout.
"""
import json

import plotly.graph_objects as go
import polars as pl

from src.figures.style import apply_style, save_figure
from src.figures.svhn_diffing.prepare import CACHE

BG       = "#FAFAF7"
LABEL    = "#0f172a"
MUTED    = "#64748b"
BOUNDARY = "#0f172a"

HEAT_COLORSCALE = [
    [0.000, "#7c2d2d"],
    [0.150, "#a8453a"],
    [0.300, "#cf7f70"],
    [0.500, BG],
    [0.700, "#6b8ec0"],
    [0.850, "#33588f"],
    [1.000, "#1c3a72"],
]

# Slice/global colors mirror the svhn_forgetting palette family: deep slate
# for the on-target signal, warm rose for the contrastive global control.
SLICE_COLOR  = "#1c3a72"
GLOBAL_COLOR = "#a8453a"


def _bounds(values):
    zmin = min(float(values.quantile(0.01)), -0.01)
    zmax = max(float(values.quantile(0.99)),  0.01)
    return zmin, zmax


def _stage_boundaries(heatmap_steps):
    bounds_x, spans = [], []
    left = -0.5
    cur_stage = heatmap_steps[0]["stage"]
    for i, cp in enumerate(heatmap_steps[1:], start=1):
        if cp["stage"] != cur_stage:
            mid = i - 0.5
            spans.append((left, mid, cur_stage))
            bounds_x.append(mid)
            left = mid
            cur_stage = cp["stage"]
    spans.append((left, len(heatmap_steps) - 0.5, cur_stage))
    return bounds_x, spans


def main():
    meta = json.loads((CACHE / "metadata.json").read_text(encoding="utf-8"))
    heatmap_steps = meta["heatmap_steps"]
    target_digit = meta["target_digit"]
    n = len(heatmap_steps)
    indices = list(range(n))

    sim = pl.read_ipc(CACHE / "similarity.feather")
    progress = pl.read_ipc(CACHE / "progress.feather")

    step_to_idx = {cp["batch"]: i for i, cp in enumerate(heatmap_steps)}
    z = (sim.with_columns(
                i=pl.col("step_i").replace_strict(step_to_idx, return_dtype=pl.Int64),
                j=pl.col("step_j").replace_strict(step_to_idx, return_dtype=pl.Int64),
            ).sort(["i", "j"])["value"]
              .to_numpy().reshape(n, n).tolist())
    zmin, zmax = _bounds(sim.filter(pl.col("step_i") != pl.col("step_j"))["value"])

    bounds_x, span_blocks = _stage_boundaries(heatmap_steps)
    spans = meta["spans"]
    cum_lim = meta["cum_xlim"]

    fig = go.Figure()

    fig.add_trace(go.Heatmap(z=z, x=indices, y=indices,
                             zmin=zmin, zmax=zmax, zmid=0,
                             colorscale=HEAT_COLORSCALE, showscale=False,
                             xaxis="x", yaxis="y"))
    for x in bounds_x:
        fig.add_shape(type="line", xref="x", yref="y",
                      x0=x, x1=x, y0=-0.5, y1=n - 0.5,
                      line_color=BOUNDARY, line_width=1.6)
        fig.add_shape(type="line", xref="x", yref="y",
                      x0=-0.5, x1=n - 0.5, y0=x, y1=x,
                      line_color=BOUNDARY, line_width=1.6)

    for metric, color, name in (("slice",  SLICE_COLOR,  f"Slice (class {target_digit})"),
                                ("global", GLOBAL_COLOR, "Global")):
        s = progress.filter(pl.col("metric") == metric).sort("batch")
        fig.add_trace(go.Scatter(
            x=s["batch"].to_list(), y=s["value"].to_list(),
            mode="lines", line=dict(color=color, width=2.4),
            name=name, xaxis="x2", yaxis="y2",
        ))

    for x0, _, _ in spans[1:]:
        fig.add_shape(type="line", xref="x2", yref="y2 domain",
                      x0=x0, x1=x0, y0=0, y1=1,
                      line_color=BOUNDARY, line_width=1.4)

    layout_axes = dict(
        xaxis=dict(domain=[0.32, 0.72], anchor="y", range=[-0.5, n - 0.5],
                   showticklabels=False, ticks="",
                   showline=False, zeroline=False, mirror=False, showgrid=False),
        yaxis=dict(domain=[0.470, 0.890], anchor="x", range=[-0.5, n - 0.5],
                   showticklabels=False, ticks="",
                   showline=False, zeroline=False, mirror=False, showgrid=False),
        xaxis2=dict(domain=[0.0, 1.0], anchor="y2", range=cum_lim,
                    showline=False, zeroline=False, showgrid=False, ticks="",
                    tickfont=dict(color=MUTED, size=13),
                    title=dict(text="<b>Cumulative batch</b>",
                               font=dict(size=15, color=LABEL))),
        yaxis2=dict(domain=[0.075, 0.345], anchor="x2", range=[-1.05, 1.05],
                    showline=False, zeroline=True,
                    zerolinecolor=MUTED, zerolinewidth=1,
                    showgrid=False, ticks="", showticklabels=False),
    )

    # Heatmap title above the panel.
    fig.add_annotation(
        x=0.52, y=0.895, xref="paper", yref="paper",
        text=f"<b>Slice similarity (class {target_digit})</b>",
        showarrow=False, xanchor="center", yanchor="bottom",
        font=dict(size=15, color=LABEL),
    )

    # Stage labels above the heatmap, vertical text per span.
    heat_left, heat_right = 0.32, 0.72
    heat_w = heat_right - heat_left
    for left_x, right_x, name in span_blocks:
        xp = heat_left + heat_w * (left_x + right_x + 1) / (2 * n)
        fig.add_annotation(x=xp, y=0.955, xref="paper", yref="paper",
                           text=name, showarrow=False,
                           xanchor="center", yanchor="middle",
                           textangle=-90,
                           font=dict(size=11, color=LABEL))

    # Stage labels above the line plot, in cum_batch coords.
    for x0, x1, name in spans:
        fig.add_annotation(x=(x0 + x1) / 2, y=0.395, xref="x2", yref="paper",
                           text=name, showarrow=False,
                           xanchor="center", yanchor="middle",
                           textangle=-90,
                           font=dict(size=12, color=LABEL))

    fig.add_annotation(text="<b>Similarity to diff</b>", xref="paper", yref="paper",
                       x=-0.005, y=0.210, xanchor="right", yanchor="middle",
                       showarrow=False, font=dict(size=14, color=LABEL))

    apply_style(fig, title=None, width=1100, height=900, legend=True)
    fig.update_layout(
        layout_axes,
        margin=dict(l=150, r=24, t=32, b=58),
        paper_bgcolor=BG, plot_bgcolor=BG,
        legend=dict(orientation="h", yanchor="bottom", y=0.420,
                    xanchor="center", x=0.5,
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(size=13, color=LABEL)),
    )
    save_figure(fig, "svhn_diffing")
