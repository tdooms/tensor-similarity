"""SVHN backdoor — focused 2×4 walkthrough, story-driven.

Four columns trace a four-step argument; each column has one story.

  1. We inject a backdoor (col 1): clean class-9 sample on top, the
     same sample with the 5×7 black-diamond trigger on the bottom.
  2. The backdoor is learned (col 2): accuracy + ASR curves above,
     behavioural similarity on the all-stamped eval below — sharp
     phase block at the boundary.
  3. Behavioural similarity on clean data is blind to it (col 3):
     pairwise similarity on the full clean test set, and restricted
     to clean class-9 inputs only — both stay near-uniform blue,
     Δ ≈ 0.09.
  4. Tensor similarity catches it AND localizes (col 4): TN sim
     across full weights (Δ ≈ 0.24) and the per-class slice for
     class 9 (Δ ≈ 0.43) — the slice surfaces a sharp phase break
     while other classes evolve smoothly.

Same blueprint as the rest of the family: gentle terracotta↔slate
diverging palette on similarity panels, asymmetric data-range bounds
with `zmid=0`, square cells. Story headers run across the top.
"""
import json

import plotly.graph_objects as go
import polars as pl

from src.figures.style import apply_style, save_figure
from src.figures.svhn_backdoor.prepare import CACHE

BG       = "#FFFFFF"
LABEL    = "#0f172a"
MUTED    = "#64748b"
BOUNDARY = "#0f172a"
TARGET_CLASS = 9
TARGET_COLOR = "#b91c1c"

# Train/test acc are context, ASR is the signal — muted slate for the
# former, rose for the latter, so the eye lands on the ASR jump.
TRAIN_COLOR = "#475569"   # slate-600
TEST_COLOR  = "#94a3b8"   # slate-400
ASR_COLOR   = TARGET_COLOR

HEAT_COLORSCALE = [
    [0.000, "#7c2d2d"], [0.150, "#a8453a"], [0.300, "#cf7f70"],
    [0.500, BG],
    [0.700, "#6b8ec0"], [0.850, "#33588f"], [1.000, "#1c3a72"],
]
GRAY_COLORSCALE = [[0.0, "#000000"], [1.0, "#FFFFFF"]]
BOUNDARY_KW = dict(line_color=BOUNDARY, line_width=3.0)

# 2×4 grid. Each cell: (kind, key, title, title_color). For image
# cells the model's prediction at the final checkpoint is appended in
# muted gray after a `·` separator at render time.
CELLS = (
    # row 1
    (("image", "clean",                "Uncorrupted",                LABEL),
     ("line",  None,                   "Accuracy & attack success",  LABEL),
     ("main",  "act_clean_full",       "Behavioural (test)",         LABEL),
     ("main",  "tn_sim",               "Tensor",                     LABEL)),
    # row 2
    (("image", "stamped",              "Corrupted",                  TARGET_COLOR),
     ("main",  "act_poisoned_full",    "Behavioural (poison)",       TARGET_COLOR),
     ("main",  "act_clean_9s",         "Behavioural (class 9)",      TARGET_COLOR),
     ("slice", TARGET_CLASS,           "Tensor (class 9)",           TARGET_COLOR)),
)

# Story headers along the top — one per column, 3 lines each. Parallel
# structure: (1) setup, (2) learned, (3) but blind, (4) caught + localized.
STORY_SPANS = (
    (0, "We inject a backdoor:<br>"
        "any stamped input<br>"
        "is flipped to class 9."),
    (1, "The backdoor is learned;<br>"
        "behaviour shifts<br>"
        "on corrupted samples."),
    (2, "Yet behavioural similarity<br>"
        "on clean data is blind<br>"
        "to the attack."),
    (3, "Tensor similarity is not;<br>"
        "it localizes the issue<br>"
        "to the attacked class."),
)

# Fig 1500 × 940, margin 4 → plot 1492 × 932. Each column 0.22 →
# 328.2 px wide. For square panels, y_dom range = 328/932 = 0.352.
# 3-line story header at 23pt (~0.12 paper) sits in the top strip;
# bottom edge of row-2 is ~10px from the figure edge.
COL_X  = ((0.000, 0.220), (0.260, 0.480),
          (0.520, 0.740), (0.780, 1.000))
ROW1_Y = (0.443, 0.795)
ROW2_Y = (0.011, 0.363)


def _bounds(values):
    zmin = min(float(values.quantile(0.01)), -0.01)
    zmax = max(float(values.quantile(0.99)),  0.01)
    return zmin, zmax


def main():
    meta = json.loads((CACHE / "metadata.json").read_text(encoding="utf-8"))
    steps = meta["steps"]
    n = len(steps)
    step_to_idx = {s: i for i, s in enumerate(steps)}
    phases = meta["phases"]

    history = pl.read_ipc(CACHE / "history.feather")
    sim_long = pl.read_ipc(CACHE / "similarity.feather")
    slice_long = pl.read_ipc(CACHE / "slice_similarity.feather")
    samples_long = pl.read_ipc(CACHE / "samples.feather")
    sample_preds = meta["sample_predictions"]

    def _matrix(metric):
        return (sim_long.filter(pl.col("metric") == metric)
                        .sort(["step_i", "step_j"])["value"]
                        .to_numpy().reshape(n, n).tolist())

    def _slice(class_idx):
        return (slice_long.filter(pl.col("class_idx") == class_idx)
                          .sort(["step_i", "step_j"])["value"]
                          .to_numpy().reshape(n, n).tolist())

    def _image(key):
        # 1024-vector reshaped to 32x32 grayscale for display.
        return (samples_long.filter(pl.col("image") == key)
                            .sort(["row", "col"])["value"]
                            .to_numpy().reshape(32, 32).tolist())

    clean_last = step_to_idx[phases[0]["last_step"]]
    poison_first = step_to_idx[phases[1]["first_step"]]
    boundary = (clean_last + poison_first) / 2

    a_idx = list(range(clean_last + 1))
    b_idx = list(range(poison_first, n))

    def _gap(df, value_col="value"):
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

    indices = list(range(n))

    fig = go.Figure()
    layout_axes = {}
    axis_num = 0

    for row_idx, row_cells in enumerate(CELLS):
        y_dom = ROW1_Y if row_idx == 0 else ROW2_Y
        for col_idx, (kind, key, title, color) in enumerate(row_cells):
            if kind == "empty":
                continue
            axis_num += 1
            x_dom = COL_X[col_idx]
            suffix = "" if axis_num == 1 else str(axis_num)
            x_id, y_id = f"x{suffix}", f"y{suffix}"
            x_paper = (x_dom[0] + x_dom[1]) / 2
            title_text = f"<b>{title}</b>"

            if kind == "image":
                pred = int(sample_preds[key])
                title_text = (f"<b>{title}</b>"
                              f"<span style='color:{MUTED}'>  ·  predicts {pred}</span>")
                # Sample images are already min/max-normalized to [0, 1] in
                # prepare.py, so the heatmap can render with fixed bounds.
                fig.add_trace(go.Heatmap(z=_image(key),
                                         zmin=0.0, zmax=1.0,
                                         colorscale=GRAY_COLORSCALE, showscale=False,
                                         xaxis=x_id, yaxis=y_id))
                layout_axes[f"xaxis{suffix}"] = dict(
                    domain=list(x_dom), anchor=y_id, range=[-0.5, 31.5],
                    showticklabels=False, ticks="",
                    showline=False, zeroline=False, mirror=False, showgrid=False,
                )
                layout_axes[f"yaxis{suffix}"] = dict(
                    domain=list(y_dom), anchor=x_id, range=[31.5, -0.5],  # flip y
                    showticklabels=False, ticks="",
                    showline=False, zeroline=False, mirror=False, showgrid=False,
                )

            elif kind == "line":
                for metric, name, lcolor in (("train_acc",           "Train acc",            TRAIN_COLOR),
                                             ("test_acc",            "Test acc",             TEST_COLOR),
                                             ("attack_success_rate", "Attack success",       ASR_COLOR)):
                    s = history.filter(pl.col("metric") == metric).sort("step")
                    x = [step_to_idx[step] for step in s["step"].to_list()]
                    y = s["value"].to_list()
                    fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
                                             name=name, legendgroup=metric,
                                             line=dict(color=lcolor, width=3.2,
                                                       shape="spline", smoothing=0.6),
                                             xaxis=x_id, yaxis=y_id,
                                             showlegend=True))
                    fig.add_annotation(x=x[-1], y=y[-1], xref=x_id, yref=y_id,
                                       text=f"<b>{int(round(y[-1] * 100))}%</b>",
                                       xanchor="left", yanchor="middle", xshift=4,
                                       showarrow=False,
                                       font=dict(size=12, color=lcolor))
                layout_axes[f"xaxis{suffix}"] = dict(
                    domain=list(x_dom), anchor=y_id, range=[-0.5, n - 0.5],
                    showticklabels=False, ticks="",
                    showline=False, zeroline=False, mirror=False, showgrid=False,
                )
                # Y-range padded to 1.32 so the legend at the top of the
                # panel doesn't clip into the ASR curve (peak ≈ 0.96).
                layout_axes[f"yaxis{suffix}"] = dict(
                    domain=list(y_dom), anchor=x_id, range=[-0.04, 1.32],
                    showticklabels=False, ticks="",
                    zeroline=True, zerolinecolor=MUTED, zerolinewidth=1,
                    showline=False, mirror=False, showgrid=False,
                )

            else:  # main / slice — similarity heatmap
                if kind == "main":
                    df = sim_long.filter(pl.col("metric") == key)
                    z = _matrix(key)
                else:
                    df = slice_long.filter(pl.col("class_idx") == key)
                    z = _slice(key)
                zmin, zmax = _bounds(df.filter(pl.col("step_i") != pl.col("step_j"))["value"])
                gap = _gap(df)
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
                title_text = (f"<b>{title}</b>"
                              f"<span style='color:{MUTED}'>  ·  Δ = {gap:.2f}</span>")

            fig.add_annotation(x=x_paper, y=y_dom[1] + 0.005, xref="paper", yref="paper",
                               text=title_text, showarrow=False,
                               xanchor="center", yanchor="bottom",
                               font=dict(size=19, color=color))

    # Story headers above each column.
    for col_idx, text in STORY_SPANS:
        x_paper = (COL_X[col_idx][0] + COL_X[col_idx][1]) / 2
        fig.add_annotation(x=x_paper, y=0.99, xref="paper", yref="paper",
                           text=text, showarrow=False,
                           xanchor="center", yanchor="top",
                           font=dict(size=27, color=LABEL),
                           align="center")

    apply_style(fig, title=None, width=1500, height=940, legend=True)
    fig.update_layout(
        layout_axes,
        margin=dict(l=4, r=4, t=4, b=4),
        paper_bgcolor=BG, plot_bgcolor=BG,
        legend=dict(orientation="h", yanchor="top", y=ROW1_Y[1] - 0.008,
                    xanchor="center", x=(COL_X[1][0] + COL_X[1][1]) / 2,
                    bgcolor="rgba(255,255,255,0.7)",
                    font=dict(size=14, color=LABEL),
                    itemsizing="constant"),
    )
    save_figure(fig, "svhn_backdoor")
