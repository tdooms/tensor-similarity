"""SVHN backdoor: weight vs functional similarity through a mid-training trigger.

Two-phase training of a 1-layer bilinear model (Bilinear(1024 → 64) → Linear):
8000 clean steps, then 2000 steps with 10% per-batch poisoning (5×7 black
diamond top-right → label 9).

Layout: three full-eval similarity heatmaps in a row (behavioural-test,
behavioural-poison, tensor) above a 2×5 grid of per-class slice
similarities. Class 9 (the backdoor target) is the only digit whose
slice exhibits a sharp phase discontinuity at step 8000; the other nine
classes evolve smoothly.

Each panel carries a small phase-block "gap" annotation
    Δ = mean(sim within phase B) − mean(sim across phases)
which functions as an unsupervised detector of the backdoor target —
argmax_c Δ(c) over the per-class slices identifies the corrupted class
without ever having seen the trigger.

Manual axis-domain placement throughout: `make_subplots` with mixed
colspan widths under-allocates the first columns and over-allocates the
last when the row above spans (observed empirically with the 3-row
layout we tried first). All panels are square in pixel space — the
figure dimensions and per-panel domain ranges are tuned to that.
"""
import json

import plotly.graph_objects as go
import polars as pl

from src.figures.style import apply_style, save_figure
from src.figures.svhn_backdoor.prepare import CACHE

BG       = "#FAFAF7"
LABEL    = "#0f172a"
MUTED    = "#64748b"
BOUNDARY = "#0f172a"
TARGET_CLASS = 9
TARGET_COLOR = "#b91c1c"

# Diverging palette: rich brick ↔ deep slate-navy through the paper bg.
# Saturated enough at the extremes to carry contrast across the figure,
# but the hues are warmer/cooler (brick, slate) rather than pure
# Tailwind red-500 / blue-500 — easier on the eyes than fully primary
# diverging maps.
HEAT_COLORSCALE = [
    [0.000, "#7c2d2d"],   # deep brick
    [0.150, "#a8453a"],
    [0.300, "#cf7f70"],   # warm rose
    [0.500, BG],          # warm-white center
    [0.700, "#6b8ec0"],   # warm slate-blue
    [0.850, "#33588f"],
    [1.000, "#1c3a72"],   # deep slate-navy
]
BOUNDARY_KW = dict(line_color=BOUNDARY, line_width=3.0)

MAIN_HEATMAPS = (
    ("act_clean_full",    "Behavioural (test)"),
    ("act_poisoned_full", "Behavioural (poison)"),
    ("tn_sim",            "Tensor"),
)

# Layout: 1200 × 950 figure, margins 2 → plot area 1196 × 946.
# Figure height shrunk so the 3 square panels fill the vertical space
# tightly: top edge ~10px above the main-row title, bottom edge ~17px
# below the digit-row-2 panel. Inter-row gaps (~0.054 of paper height)
# carry the next row's title with breathing clear space.
MAIN_X = ((0.000, 0.300), (0.350, 0.650), (0.700, 1.000))
MAIN_Y = (0.582, 0.961)

DIGIT_X = ((0.000, 0.180), (0.205, 0.385), (0.410, 0.590),
           (0.615, 0.795), (0.820, 1.000))
DIGIT_Y_TOP = (0.300, 0.528)
DIGIT_Y_BOT = (0.018, 0.246)


def main():
    meta = json.loads((CACHE / "metadata.json").read_text(encoding="utf-8"))
    steps = meta["steps"]
    n = len(steps)
    step_to_idx = {s: i for i, s in enumerate(steps)}
    phases = meta["phases"]

    sim_long = pl.read_ipc(CACHE / "similarity.feather")
    slice_long = pl.read_ipc(CACHE / "slice_similarity.feather")

    def _matrix(metric):
        return (sim_long.filter(pl.col("metric") == metric)
                        .sort(["step_i", "step_j"])["value"]
                        .to_numpy().reshape(n, n).tolist())

    def _slice(class_idx):
        return (slice_long.filter(pl.col("class_idx") == class_idx)
                          .sort(["step_i", "step_j"])["value"]
                          .to_numpy().reshape(n, n).tolist())

    def _bounds(values):
        # Asymmetric data-range bounds with zmid=0 below: keeps white at
        # the true 0 in the diverging colormap (sign is honest), but lets
        # the visible color range stretch over where the data actually
        # lives. Symmetric [-vmax, vmax] would pack the [0.85, 1.0] cluster
        # into ~7% of the palette and render as uniform blue.
        zmin = min(float(values.quantile(0.01)), -0.01)
        zmax = max(float(values.quantile(0.99)),  0.01)
        return zmin, zmax

    main_bounds = {
        m: _bounds(sim_long.filter((pl.col("metric") == m)
                                   & (pl.col("step_i") != pl.col("step_j")))["value"])
        for m, _ in MAIN_HEATMAPS
    }
    slice_bounds = _bounds(slice_long.filter(pl.col("step_i") != pl.col("step_j"))["value"])

    # Phase indices (cell-centered). Boundary line falls cleanly between the
    # last clean checkpoint and the first poisoned one — half-step offset.
    clean_last = step_to_idx[phases[0]["last_step"]]
    poison_first = step_to_idx[phases[1]["first_step"]]
    boundary = (clean_last + poison_first) / 2

    # Phase-block gap detector: for any per-checkpoint NxN similarity matrix,
    # Δ = mean(within-B off-diag) − mean(across-A↔B). Higher Δ ⇒ sharper
    # phase transition. argmax_c Δ(c) over per-class slices identifies the
    # backdoor target unsupervised.
    a_idx = list(range(clean_last + 1))     # 0..clean_last inclusive
    b_idx = list(range(poison_first, n))    # poison_first..n-1
    a_set, b_set = set(a_idx), set(b_idx)

    def _gap_from_long(df, value_col="value"):
        # df has columns step_i, step_j, value. Returns Δ.
        df = df.with_columns(
            idx_i=pl.col("step_i").replace_strict(step_to_idx, return_dtype=pl.Int64),
            idx_j=pl.col("step_j").replace_strict(step_to_idx, return_dtype=pl.Int64),
        )
        within_b = df.filter(pl.col("idx_i").is_in(b_idx)
                             & pl.col("idx_j").is_in(b_idx)
                             & (pl.col("idx_i") != pl.col("idx_j")))[value_col].mean()
        across = df.filter(((pl.col("idx_i").is_in(a_idx) & pl.col("idx_j").is_in(b_idx))
                            | (pl.col("idx_i").is_in(b_idx) & pl.col("idx_j").is_in(a_idx))))[value_col].mean()
        return float(within_b) - float(across)

    main_gaps = {m: _gap_from_long(sim_long.filter(pl.col("metric") == m))
                 for m, _ in MAIN_HEATMAPS}
    slice_gaps = {c: _gap_from_long(slice_long.filter(pl.col("class_idx") == c))
                  for c in range(10)}

    indices = list(range(n))

    fig = go.Figure()

    panels = []  # (axis_num, x_dom, y_dom, title, label_color, z, (zmin, zmax), gap)
    for k, (metric, title) in enumerate(MAIN_HEATMAPS, start=1):
        panels.append((k, MAIN_X[k - 1], MAIN_Y, title, LABEL,
                       _matrix(metric), main_bounds[metric], main_gaps[metric]))
    for digit in range(10):
        col = digit % 5
        y = DIGIT_Y_TOP if digit < 5 else DIGIT_Y_BOT
        is_target = (digit == TARGET_CLASS)
        color = TARGET_COLOR if is_target else LABEL
        panels.append((4 + digit, DIGIT_X[col], y, f"class {digit}", color,
                       _slice(digit), slice_bounds, slice_gaps[digit]))

    layout_axes = {}
    for axis_num, x_dom, y_dom, _, _, z, (zmin, zmax), _ in panels:
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

    # Phase-boundary cross on every panel.
    for axis_num, *_ in panels:
        suffix = "" if axis_num == 1 else str(axis_num)
        fig.add_shape(type="line", xref=f"x{suffix}", yref=f"y{suffix}",
                      x0=boundary, x1=boundary, y0=-0.5, y1=n - 0.5,
                      **BOUNDARY_KW)
        fig.add_shape(type="line", xref=f"x{suffix}", yref=f"y{suffix}",
                      x0=-0.5, x1=n - 0.5, y0=boundary, y1=boundary,
                      **BOUNDARY_KW)

    # Panel titles with Δ inline. Title bold + Δ in muted regular weight
    # gives a clear hierarchy (panel name first, score after) without
    # adding a separate annotation row below the panel.
    for axis_num, x_dom, y_dom, title, color, _, _, gap in panels:
        x_paper = (x_dom[0] + x_dom[1]) / 2
        title_size = 15 if axis_num <= 3 else 12
        text = (f"<b>{title}</b>"
                f"<span style='color:{MUTED}'>  ·  Δ = {gap:.2f}</span>")
        fig.add_annotation(x=x_paper, y=y_dom[1] + 0.005, xref="paper", yref="paper",
                           text=text, showarrow=False,
                           xanchor="center", yanchor="bottom",
                           font=dict(size=title_size, color=color))

    apply_style(fig, title=None, width=1200, height=950, legend=False)
    fig.update_layout(
        layout_axes,
        margin=dict(l=2, r=2, t=2, b=2),
        paper_bgcolor=BG, plot_bgcolor=BG,
    )
    save_figure(fig, "svhn_backdoor")
