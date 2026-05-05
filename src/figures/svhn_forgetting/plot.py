"""SVHN forgetting: which similarities track digit-level capability loss.

A DeepMLP is trained through a 9-stage curriculum that progressively expands
from digits 0–4 to 0–9, repeats, removes digit 9, then re-adds it. Three
metrics summarise the run as N×N pairwise heatmaps across 80 subsampled
checkpoints:

    Tensor  — Gaussian functional similarity (gauge-invariant, data-free)
    CKA     — linear CKA on logits (data-driven, per-sample)
    Weight  — naive weight cosine

A 2×5 grid below shows the per-digit Tensor slice. The forgetting signal lives
in class 9: its slice collapses during *remove 9* and recovers during *re-add
9*, while the other nine digits keep high pairwise similarity throughout.

Layout follows `svhn_backdoor`: warm bg, brick↔slate diverging palette,
square panels via manual axis-domain placement, slate stage-boundary lines
across every panel, stage names as vertical labels above the top row.
"""
import json

import plotly.graph_objects as go
import polars as pl

from src.figures.style import apply_style, save_figure
from src.figures.svhn_forgetting.prepare import CACHE

BG       = "#FAFAF7"
LABEL    = "#0f172a"
MUTED    = "#64748b"
BOUNDARY = "#0f172a"
TARGET_CLASS = 9
TARGET_COLOR = "#b91c1c"

HEAT_COLORSCALE = [
    [0.000, "#7c2d2d"],
    [0.150, "#a8453a"],
    [0.300, "#cf7f70"],
    [0.500, BG],
    [0.700, "#6b8ec0"],
    [0.850, "#33588f"],
    [1.000, "#1c3a72"],
]
BOUNDARY_KW = dict(line_color=BOUNDARY, line_width=0.7)

MAIN_HEATMAPS = (
    ("tn",     "Tensor"),
    ("cka",    "CKA · logits"),
    ("weight", "Weight cosine"),
)

# Layout: 1240 × 920 figure → plot area 1172 × 896 with margin l=66 (room
# for stage tick labels on the leftmost panel of each row), r=2, t=22, b=2.
# Main panels: 0.30 × 1172 = 351 px wide → matched by y span 0.392 × 896 ≈
# 351 px tall (square). Digit panels: 0.18 × 1172 = 211 px wide → matched by
# y span 0.235 × 896 ≈ 211 px tall (square).
MAIN_X = ((0.000, 0.300), (0.350, 0.650), (0.700, 1.000))
MAIN_Y = (0.580, 0.972)

DIGIT_X = ((0.000, 0.180), (0.205, 0.385), (0.410, 0.590),
           (0.615, 0.795), (0.820, 1.000))
DIGIT_Y_TOP = (0.305, 0.540)
DIGIT_Y_BOT = (0.020, 0.255)


def _bounds(values):
    zmin = min(float(values.quantile(0.01)), -0.01)
    zmax = max(float(values.quantile(0.99)),  0.01)
    return zmin, zmax


def _stage_boundaries(heatmap_steps):
    """Return (boundary_indices, [(left_idx, right_idx, stage_name)]).

    Indices live in heatmap-cell space [-0.5, n-0.5]: a boundary at i.5 falls
    between cells i and i+1 (i.e. between checkpoints i and i+1).
    """
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
    n = len(heatmap_steps)
    indices = list(range(n))

    sim = pl.read_ipc(CACHE / "similarity.feather")
    slice_sim = pl.read_ipc(CACHE / "slice_similarity.feather") \
                   .filter(pl.col("kind") == "tensor")

    step_to_idx = {cp["batch"]: i for i, cp in enumerate(heatmap_steps)}

    def _matrix(df, value_col="value"):
        return (df.with_columns(
                    i=pl.col("step_i").replace_strict(step_to_idx, return_dtype=pl.Int64),
                    j=pl.col("step_j").replace_strict(step_to_idx, return_dtype=pl.Int64),
                ).sort(["i", "j"])[value_col]
                  .to_numpy().reshape(n, n).tolist())

    main_bounds = {
        m: _bounds(sim.filter((pl.col("metric") == m)
                              & (pl.col("step_i") != pl.col("step_j")))["value"])
        for m, _ in MAIN_HEATMAPS
    }
    slice_bounds = _bounds(slice_sim.filter(pl.col("step_i") != pl.col("step_j"))["value"])

    bounds_x, span_blocks = _stage_boundaries(heatmap_steps)

    panels = []  # (axis_num, x_dom, y_dom, title, color, z, (zmin, zmax))
    for k, (metric, title) in enumerate(MAIN_HEATMAPS, start=1):
        z = _matrix(sim.filter(pl.col("metric") == metric))
        panels.append((k, MAIN_X[k - 1], MAIN_Y, title, LABEL, z, main_bounds[metric]))
    for digit in range(10):
        col = digit % 5
        y = DIGIT_Y_TOP if digit < 5 else DIGIT_Y_BOT
        is_target = (digit == TARGET_CLASS)
        color = TARGET_COLOR if is_target else LABEL
        z = _matrix(slice_sim.filter(pl.col("class_idx") == digit))
        panels.append((4 + digit, DIGIT_X[col], y, f"class {digit}", color,
                       z, slice_bounds))

    # Stage labels ride the y-axis of each row's leftmost panel (axes 1, 4, 9).
    # All panels in a row share y_dom and the same checkpoint range, so a
    # single yaxis tick set serves the whole row. tickvals are heatmap-index
    # midpoints; tick labels render horizontally in the figure left margin.
    stage_tickvals = [(left_x + right_x) / 2 for left_x, right_x, _ in span_blocks]
    stage_ticktext = [name for _, _, name in span_blocks]
    LEFT_OF_ROW = {1, 4, 9}

    fig = go.Figure()
    layout_axes = {}
    for axis_num, x_dom, y_dom, _, _, z, (zmin, zmax) in panels:
        suffix = "" if axis_num == 1 else str(axis_num)
        x_id, y_id = f"x{suffix}", f"y{suffix}"
        fig.add_trace(go.Heatmap(z=z, x=indices, y=indices,
                                 zmin=zmin, zmax=zmax, zmid=0,
                                 colorscale=HEAT_COLORSCALE, showscale=False,
                                 xaxis=x_id, yaxis=y_id))
        layout_axes[f"xaxis{suffix}"] = dict(
            domain=list(x_dom), anchor=y_id, range=[-0.5, n - 0.5],
            showticklabels=False, ticks="",
            showline=False, zeroline=False, mirror=False, showgrid=False,
        )

        y_axis_kw = dict(
            domain=list(y_dom), anchor=x_id, range=[-0.5, n - 0.5],
            showline=False, zeroline=False, mirror=False, showgrid=False,
        )
        if axis_num in LEFT_OF_ROW:
            y_axis_kw |= dict(
                tickmode="array", tickvals=stage_tickvals, ticktext=stage_ticktext,
                ticks="", tickfont=dict(color=LABEL, size=11),
                showticklabels=True,
            )
        else:
            y_axis_kw |= dict(showticklabels=False, ticks="")
        layout_axes[f"yaxis{suffix}"] = y_axis_kw

        for x in bounds_x:
            fig.add_shape(type="line", xref=f"x{suffix}", yref=f"y{suffix}",
                          x0=x, x1=x, y0=-0.5, y1=n - 0.5, **BOUNDARY_KW)
            fig.add_shape(type="line", xref=f"x{suffix}", yref=f"y{suffix}",
                          x0=-0.5, x1=n - 0.5, y0=x, y1=x, **BOUNDARY_KW)

    for axis_num, x_dom, y_dom, title, color, _, _ in panels:
        x_paper = (x_dom[0] + x_dom[1]) / 2
        title_size = 16 if axis_num <= 3 else 13
        fig.add_annotation(x=x_paper, y=y_dom[1] + 0.005, xref="paper", yref="paper",
                           text=f"<b>{title}</b>", showarrow=False,
                           xanchor="center", yanchor="bottom",
                           font=dict(size=title_size, color=color))

    apply_style(fig, title=None, width=1240, height=920, legend=False)
    fig.update_layout(
        layout_axes,
        margin=dict(l=66, r=2, t=22, b=2),
        paper_bgcolor=BG, plot_bgcolor=BG,
    )
    save_figure(fig, "svhn_forgetting")

    # ── companion: progress curves ───────────────────────────────────────
    # 3 stacked panels share the cum_batch axis: train/val accuracy,
    # train/val loss (log), and similarity-to-reference (3 metrics).
    # Stage spans are drawn as vertical boundary lines across all rows.
    _plot_progress(meta)


# Neutral train/val: medium-gray vs near-black slate. Reads as "two flavours
# of the same baseline" rather than competing primaries; the curves carry the
# information, the legend pulls them apart by name not by hue.
TRAIN = "#94a3b8"   # slate-400
VAL   = "#1e293b"   # slate-800
# Sim metrics carry the figure's narrative — they get the saturated palette.
SIM_COLORS = {
    "tn":     "#1c3a72",   # deep slate-navy (matches heatmap deep blue)
    "cka":    "#6b8ec0",   # mid slate-blue
    "weight": "#a8453a",   # warm rose
}
SIM_LABELS = {"tn": "Tensor", "cka": "CKA · logits", "weight": "Weight cosine"}


def _plot_progress(meta):
    history = pl.read_ipc(CACHE / "history.feather")
    progress = pl.read_ipc(CACHE / "progress.feather")
    spans = meta["spans"]

    cum_lim = meta["cum_xlim"]
    sim_lim = meta["sim_xlim"]

    fig = go.Figure()

    def _add(metric, df, color, name, axis, group, showlegend):
        s = df.filter(pl.col("metric") == metric).sort("batch")
        fig.add_trace(go.Scatter(
            x=s["batch"].to_list(), y=s["value"].to_list(),
            mode="lines", line=dict(color=color, width=2.2),
            name=name, legendgroup=group, showlegend=showlegend,
            xaxis=axis[0], yaxis=axis[1],
        ))

    # Row 1: accuracy — train/val legend entries declared here.
    _add("train_acc", history, TRAIN, "Train", ("x", "y"),  group="train", showlegend=True)
    _add("val_acc",   history, VAL,   "Val",   ("x", "y"),  group="val",   showlegend=True)
    # Row 2: loss (log) — same train/val groups, no duplicate legend entries.
    _add("train_loss", history, TRAIN, "Train", ("x", "y2"), group="train", showlegend=False)
    _add("val_loss",   history, VAL,   "Val",   ("x", "y2"), group="val",   showlegend=False)
    # Row 3: similarity — three distinct metrics, each their own legend entry.
    for metric in ("tn", "cka", "weight"):
        _add(metric, progress, SIM_COLORS[metric], SIM_LABELS[metric],
             ("x", "y3"), group=metric, showlegend=True)

    tick_kw = dict(showticklabels=True, ticks="outside",
                   tickcolor="#cbd5e1", tickfont=dict(color=MUTED, size=11))
    layout_axes = dict(
        xaxis=dict(domain=[0.06, 0.995], anchor="y3", range=cum_lim,
                   showline=False, zeroline=False, showgrid=False, ticks="",
                   tickfont=dict(color=MUTED, size=13),
                   title=dict(text="<b>Cumulative batch</b>",
                              font=dict(size=15, color=LABEL))),
        yaxis=dict(domain=[0.640, 0.890], anchor="x", range=[0, 1.02],
                   showline=False, zeroline=False,
                   showgrid=True, gridcolor="#e2e8f0",
                   tickmode="array", tickvals=[0, 0.5, 1.0],
                   ticktext=["0", "0.5", "1"], **tick_kw),
        yaxis2=dict(domain=[0.345, 0.595], anchor="x", type="log",
                    range=[-1.05, 0.4],
                    showline=False, zeroline=False,
                    showgrid=True, gridcolor="#e2e8f0",
                    tickmode="array", tickvals=[-1, 0],
                    ticktext=["10⁻¹", "10⁰"], **tick_kw),
        yaxis3=dict(domain=[0.060, 0.310], anchor="x", range=[-0.05, 1.05],
                    showline=False, zeroline=True,
                    zerolinecolor=MUTED, zerolinewidth=1,
                    showgrid=True, gridcolor="#e2e8f0",
                    tickmode="array", tickvals=[0, 0.5, 1.0],
                    ticktext=["0", "0.5", "1"], **tick_kw),
    )

    # Stage boundaries on every row (vertical lines + name annotations
    # above the figure, vertical text).
    for x0, _, _ in spans[1:]:
        for ax in ("y", "y2", "y3"):
            fig.add_shape(type="line", xref="x", yref=f"{ax} domain",
                          x0=x0, x1=x0, y0=0, y1=1,
                          line_color=BOUNDARY, line_width=1.4)

    for x0, x1, name in spans:
        fig.add_annotation(x=(x0 + x1) / 2, y=0.905, xref="x", yref="paper",
                           text=name, showarrow=False,
                           xanchor="center", yanchor="bottom",
                           textangle=-90,
                           font=dict(size=12, color=LABEL))

    for label, mid_y in (("Accuracy",   0.765),
                         ("Loss (log)", 0.470),
                         ("Similarity", 0.185)):
        fig.add_annotation(text=f"<b>{label}</b>", xref="paper", yref="paper",
                           x=-0.005, y=mid_y, xanchor="right", yanchor="middle",
                           showarrow=False, font=dict(size=14, color=LABEL))

    apply_style(fig, title=None, width=1100, height=720, legend=True)
    fig.update_layout(
        layout_axes,
        margin=dict(l=88, r=24, t=64, b=58),
        paper_bgcolor=BG, plot_bgcolor=BG,
        legend=dict(orientation="h", yanchor="bottom", y=0.985,
                    xanchor="center", x=0.5,
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(size=13, color=LABEL)),
    )
    save_figure(fig, "svhn_forgetting_progress")
