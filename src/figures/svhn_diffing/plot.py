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


def main():
    meta = json.loads((CACHE / "metadata.json").read_text(encoding="utf-8"))
    heatmap_steps = meta["heatmap_steps"]
    heatmap_batches = [cp["batch"] for cp in heatmap_steps]
    target_digit = meta["target_digit"]
    n = len(heatmap_steps)

    sim = pl.read_ipc(CACHE / "similarity.feather")
    progress = pl.read_ipc(CACHE / "progress.feather")

    step_to_idx = {b: i for i, b in enumerate(heatmap_batches)}
    z = (sim.with_columns(
                i=pl.col("step_i").replace_strict(step_to_idx, return_dtype=pl.Int64),
                j=pl.col("step_j").replace_strict(step_to_idx, return_dtype=pl.Int64),
            ).sort(["i", "j"])["value"]
              .to_numpy().reshape(n, n).tolist())
    zmin, zmax = _bounds(sim.filter(pl.col("step_i") != pl.col("step_j"))["value"])

    spans = meta["spans"]
    cum_lim = meta["cum_xlim"]

    # Heatmap and line plot share the cum_batch x-axis: heatmap cells live at
    # actual checkpoint batch numbers, line plot at sim_batch values. Both
    # span the same x_dom on paper, so vertical features (stage boundaries,
    # spikes at stage transitions) line up across the two panels.
    stage_boundaries_batch = [s[0] for s in spans[1:]]
    stage_midpoints_batch = [(s[0] + s[1]) / 2 for s in spans]
    stage_names = [s[2] for s in spans]

    fig = go.Figure()

    fig.add_trace(go.Heatmap(z=z, x=heatmap_batches, y=heatmap_batches,
                             zmin=zmin, zmax=zmax, zmid=0,
                             colorscale=HEAT_COLORSCALE, showscale=False,
                             xaxis="x", yaxis="y"))
    for xb in stage_boundaries_batch:
        fig.add_shape(type="line", xref="x", yref="y",
                      x0=xb, x1=xb, y0=cum_lim[0], y1=cum_lim[1],
                      line_color=BOUNDARY, line_width=0.7)
        fig.add_shape(type="line", xref="x", yref="y",
                      x0=cum_lim[0], x1=cum_lim[1], y0=xb, y1=xb,
                      line_color=BOUNDARY, line_width=0.7)

    for metric, color, name in (("slice",  SLICE_COLOR,  f"Slice (class {target_digit})"),
                                ("global", GLOBAL_COLOR, "Global")):
        s = progress.filter(pl.col("metric") == metric).sort("batch")
        fig.add_trace(go.Scatter(
            x=s["batch"].to_list(), y=s["value"].to_list(),
            mode="lines", line=dict(color=color, width=2.4),
            name=name, xaxis="x2", yaxis="y2",
        ))

    for xb in stage_boundaries_batch:
        fig.add_shape(type="line", xref="x2", yref="y2 domain",
                      x0=xb, x1=xb, y0=0, y1=1,
                      line_color=BOUNDARY, line_width=1.0)

    # Single shared x_dom keeps the two panels aligned. Heatmap top-axis
    # carries the stage names; line-plot bottom-axis carries the cum_batch
    # tick labels — same range, same paper-x extent, no double-labeling.
    SHARED_X = [0.06, 0.985]
    layout_axes = dict(
        xaxis=dict(domain=SHARED_X, anchor="y", range=cum_lim,
                   side="top",
                   tickmode="array", tickvals=stage_midpoints_batch,
                   ticktext=stage_names,
                   ticks="", tickfont=dict(color=LABEL, size=12),
                   showline=False, zeroline=False, mirror=False, showgrid=False),
        yaxis=dict(domain=[0.435, 0.895], anchor="x", range=cum_lim,
                   showticklabels=False, ticks="",
                   showline=False, zeroline=False, mirror=False, showgrid=False),
        xaxis2=dict(domain=SHARED_X, anchor="y2", range=cum_lim,
                    showline=False, zeroline=False, showgrid=False, ticks="",
                    tickfont=dict(color=MUTED, size=12),
                    title=dict(text="<b>Cumulative batch</b>",
                               font=dict(size=15, color=LABEL))),
        yaxis2=dict(domain=[0.075, 0.345], anchor="x2", range=[-1.05, 1.05],
                    showline=False, zeroline=True,
                    zerolinecolor=MUTED, zerolinewidth=1,
                    showgrid=True, gridcolor="#e2e8f0",
                    tickmode="array", tickvals=[-1.0, 0.0, 1.0],
                    ticktext=["−1", "0", "1"],
                    ticks="outside", tickcolor="#cbd5e1",
                    tickfont=dict(color=MUTED, size=11),
                    showticklabels=True),
    )

    # Y-axis labels — one per panel, in the left margin.
    for label, mid_y in ((f"<b>Slice similarity (class {target_digit})</b>", 0.665),
                         ("<b>Similarity to diff</b>",                       0.210)):
        fig.add_annotation(text=label, xref="paper", yref="paper",
                           x=-0.005, y=mid_y,
                           xanchor="right", yanchor="middle",
                           textangle=-90,
                           showarrow=False,
                           font=dict(size=13, color=LABEL))

    apply_style(fig, title=None, width=1300, height=820, legend=True)
    fig.update_layout(
        layout_axes,
        margin=dict(l=46, r=18, t=44, b=58),
        paper_bgcolor=BG, plot_bgcolor=BG,
        legend=dict(orientation="h", yanchor="middle", y=0.395,
                    xanchor="center", x=0.5,
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(size=13, color=LABEL)),
    )
    save_figure(fig, "svhn_diffing")
