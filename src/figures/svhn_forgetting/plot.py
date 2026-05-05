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
BOUNDARY_KW = dict(line_color=BOUNDARY, line_width=1.6)

MAIN_HEATMAPS = (
    ("tn",     "Tensor"),
    ("cka",    "CKA · logits"),
    ("weight", "Weight cosine"),
)

# Layout: 1200 × 1240 figure (matches svhn_backdoor's 3+2×5 blueprint with a
# small extra header carved out for stage labels). Three main square panels
# top, 2×5 grid below. Stage labels live in the ~0.045 of paper above the
# panel titles, vertical text per stage span.
MAIN_X = ((0.000, 0.300), (0.350, 0.650), (0.700, 1.000))
MAIN_Y = (0.546, 0.882)

DIGIT_X = ((0.000, 0.180), (0.205, 0.385), (0.410, 0.590),
           (0.615, 0.795), (0.820, 1.000))
DIGIT_Y_TOP = (0.270, 0.474)
DIGIT_Y_BOT = (0.018, 0.222)
STAGE_LABEL_Y = 0.945


def _bounds(values):
    zmin = min(float(values.quantile(0.01)), -0.01)
    zmax = max(float(values.quantile(0.99)),  0.01)
    return zmin, zmax


def _stage_boundaries(heatmap_steps):
    """Return (boundary_x_positions, [(left_x, right_x, stage_name)])."""
    bounds_x, spans = [], []
    left = -0.5
    cur_stage = heatmap_steps[0]["stage"]
    cur_left_idx = 0
    for i, cp in enumerate(heatmap_steps[1:], start=1):
        if cp["stage"] != cur_stage:
            mid = (i - 1 + i) / 2
            spans.append((left, mid, cur_stage))
            bounds_x.append(mid)
            left = mid
            cur_stage = cp["stage"]
            cur_left_idx = i
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
        layout_axes[f"yaxis{suffix}"] = dict(
            domain=list(y_dom), anchor=x_id, range=[-0.5, n - 0.5],
            showticklabels=False, ticks="",
            showline=False, zeroline=False, mirror=False, showgrid=False,
        )
        for x in bounds_x:
            for direction in ("v", "h"):
                kw = (dict(x0=x, x1=x, y0=-0.5, y1=n - 0.5) if direction == "v"
                      else dict(x0=-0.5, x1=n - 0.5, y0=x, y1=x))
                fig.add_shape(type="line", xref=f"x{suffix}", yref=f"y{suffix}",
                              **kw, **BOUNDARY_KW)

    for axis_num, x_dom, y_dom, title, color, _, _ in panels:
        x_paper = (x_dom[0] + x_dom[1]) / 2
        title_size = 15 if axis_num <= 3 else 12
        fig.add_annotation(x=x_paper, y=y_dom[1] + 0.005, xref="paper", yref="paper",
                           text=f"<b>{title}</b>", showarrow=False,
                           xanchor="center", yanchor="bottom",
                           font=dict(size=title_size, color=color))

    # Stage labels span the full top-row width (one global label per stage,
    # not one per panel). The 3 main panels share the same checkpoint x-axis
    # so a global mapping is correct conceptually; the small inter-panel
    # gaps mean labels are very slightly offset from their per-panel boundary
    # lines, but cleaner than 27 cramped per-panel labels.
    full_left, full_right = MAIN_X[0][0], MAIN_X[-1][1]
    full_w = full_right - full_left
    for left_x, right_x, name in span_blocks:
        xp = full_left + full_w * (left_x + right_x + 1) / (2 * n)
        fig.add_annotation(x=xp, y=STAGE_LABEL_Y, xref="paper", yref="paper",
                           text=name, showarrow=False,
                           xanchor="center", yanchor="middle",
                           textangle=-90,
                           font=dict(size=12, color=LABEL))

    apply_style(fig, title=None, width=1200, height=1240, legend=False)
    fig.update_layout(
        layout_axes,
        margin=dict(l=2, r=2, t=2, b=2),
        paper_bgcolor=BG, plot_bgcolor=BG,
    )
    save_figure(fig, "svhn_forgetting")

    # ── companion: progress curves ───────────────────────────────────────
    # 3 stacked panels share the cum_batch axis: train/val accuracy,
    # train/val loss (log), and similarity-to-reference (3 metrics).
    # Stage spans are drawn as vertical boundary lines across all rows.
    _plot_progress(meta)


# Neutral train/val: muted vs dark slate. Reads as "two flavours of the same
# baseline" rather than competing primaries — the curves themselves carry the
# information, the legend pulls them apart by name not by hue saturation.
TRAIN = "#94a3b8"   # slate-400
VAL   = "#334155"   # slate-700
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

    layout_axes = dict(
        xaxis=dict(domain=[0.06, 0.995], anchor="y3", range=cum_lim,
                   showline=False, zeroline=False, showgrid=False, ticks="",
                   tickfont=dict(color=MUTED, size=13),
                   title=dict(text="<b>Cumulative batch</b>",
                              font=dict(size=15, color=LABEL))),
        yaxis=dict(domain=[0.640, 0.890], anchor="x", range=[0, 1.02],
                   showline=False, zeroline=False, showgrid=False, ticks="",
                   showticklabels=False),
        yaxis2=dict(domain=[0.345, 0.595], anchor="x", type="log",
                    showline=False, zeroline=False, showgrid=False, ticks="",
                    showticklabels=False),
        yaxis3=dict(domain=[0.060, 0.310], anchor="x", range=[-0.05, 1.05],
                    showline=False, zeroline=False, showgrid=False, ticks="",
                    showticklabels=False),
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
