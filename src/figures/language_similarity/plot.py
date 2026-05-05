"""Pairwise similarity across language-model training checkpoints.

Same blueprint as `svhn-backdoor`: gentle terracotta↔slate diverging
palette anchored at the paper bg with `zmid=0`, asymmetric data-range
bounds so the heatmap structure isn't compressed into 7% of the
palette, manual axis-domain placement, no figure title, tight image
margins. Two panels: a slim n-gram score trace above, a square
checkpoint-similarity heatmap below.

Phase boundaries are hand-picked from the visual block structure of
the N=101 similarity matrix (steps 27 → 20000); they're tied to
checkpoint-index, not step value, so rerunning with a different N
requires re-picking. Kept neutral (no mechanistic-phase labels) — we
have no circuits-level evidence those bands mean any specific thing.
"""
import json

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from src.figures.language_similarity.prepare import CACHE
from src.figures.style import apply_style, save_figure

BG       = "#FAFAF7"
LABEL    = "#0f172a"
MUTED    = "#64748b"
BOUNDARY = "#0f172a"

# Diverging palette: rich brick ↔ deep slate-navy through the paper bg.
# Same palette as svhn-backdoor — these three figures share aesthetics.
HEAT_COLORSCALE = [
    [0.000, "#7c2d2d"],
    [0.150, "#a8453a"],
    [0.300, "#cf7f70"],
    [0.500, BG],
    [0.700, "#6b8ec0"],
    [0.850, "#33588f"],
    [1.000, "#1c3a72"],
]
BOUNDARY_KW = dict(line_color=BOUNDARY, line_width=2.5)

NGRAM_TRACES = (
    ("2gram_score", "2-gram", "#a5b4fc"),
    ("3gram_score", "3-gram", "#6366f1"),
    ("4gram_score", "4-gram", "#1c3a72"),  # match heatmap deep-blue endpoint
)

PHASE_BOUNDARIES_INDEX = (7.5, 27.5, 51.5, 91.5)

TICK_TARGETS = (100, 1_000, 10_000)
TICK_LABELS  = ("100", "1k", "10k")

# Spectrum of similarity definitions (companion figure). All in TN form
# (sums-then-cosine), all at matched input scale (Gaussian σ=0.02 to match
# embedding init). The only thing that changes per panel is WHERE the
# inner product is taken.
METRIC_SPEC = (
    ("empirical_pile_tnform.feather",          "Pile · logits"),
    ("empirical_pile_residual_tnform.feather", "Pile · residual"),
    ("empirical_gaussian_tnform.feather",      "Gaussian · logits"),
    ("matrix.feather",                         "Gaussian · Tensor"),
)


def _bounds(values):
    # Asymmetric data-range bounds with zmid=0 below: keeps white at 0
    # in the diverging palette, but the visible color range stretches
    # over where the data actually lives. Symmetric [-1, 1] would pack
    # the [0.85, 1.0] cluster into ~7% of the palette → uniform blue.
    zmin = min(float(values.quantile(0.01)), -0.01)
    zmax = max(float(values.quantile(0.99)),  0.01)
    return zmin, zmax


def main():
    meta = json.loads((CACHE / "metadata.json").read_text(encoding="utf-8"))
    steps = [s for s in meta["steps"] if s > 0]
    n = len(steps)
    step_to_idx = {s: i for i, s in enumerate(steps)}
    indices = list(range(n))

    matrix_long = (pl.read_ipc(CACHE / "matrix.feather")
                     .filter((pl.col("step_i") > 0) & (pl.col("step_j") > 0)))
    matrix = matrix_long.sort(["step_i", "step_j"])["similarity"].to_numpy().reshape(n, n).tolist()
    matrix_lo, matrix_hi = _bounds(
        matrix_long.filter(pl.col("step_i") != pl.col("step_j"))["similarity"]
    )

    behavior = (pl.read_ipc(CACHE / "behavior.feather")
                  .filter(pl.col("step").is_in(set(steps))))

    seen, tick_indices, tick_labels = set(), [], []
    for target, label in zip(TICK_TARGETS, TICK_LABELS):
        idx = min(range(n), key=lambda i: abs(steps[i] - target))
        if idx not in seen:
            seen.add(idx)
            tick_indices.append(idx)
            tick_labels.append(label)

    # ── main figure: n-gram trace + similarity heatmap ─────────────────
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.022)

    end_values = []
    for metric, label, color in NGRAM_TRACES:
        s = behavior.filter(pl.col("metric") == metric).sort("step")
        x = [step_to_idx[step] for step in s["step"].to_list()]
        y = s["value"].to_list()
        end_values.append((color, x[-1], y[-1]))
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
                                 name=label, legendgroup=metric,
                                 line=dict(color=color, width=2.4)),
                      row=1, col=1)

    fig.add_trace(go.Heatmap(z=matrix, x=indices, y=indices,
                             zmin=matrix_lo, zmax=matrix_hi, zmid=0,
                             colorscale=HEAT_COLORSCALE, showscale=False),
                  row=2, col=1)

    for x in PHASE_BOUNDARIES_INDEX:
        for row in (1, 2):
            fig.add_vline(x=x, row=row, col=1, **BOUNDARY_KW)

    for color, last_x, last_y in end_values:
        fig.add_annotation(text=f"<b>{last_y:.2f}</b>",
                           x=last_x, y=last_y, row=1, col=1,
                           xanchor="left", yanchor="middle", xshift=5,
                           showarrow=False, font=dict(size=12, color=color))

    fig.update_xaxes(showline=False, zeroline=False, mirror=False,
                     range=[-0.5, n - 0.5], showgrid=False, ticks="")
    fig.update_xaxes(tickmode="array", tickvals=tick_indices, ticktext=tick_labels,
                     row=2, col=1)
    fig.update_yaxes(automargin=False, title=None,
                     showline=False, zeroline=False, mirror=False, showgrid=False,
                     showticklabels=False, ticks="")
    fig.update_yaxes(domain=[0.83, 0.97],
                     range=[0, 2.6],
                     showticklabels=False, ticks="",
                     zeroline=True, zerolinecolor=MUTED, zerolinewidth=1,
                     row=1, col=1)
    fig.update_yaxes(range=[-0.5, n - 0.5], domain=[0.030, 0.785], row=2, col=1)

    for label, mid_y in (("n-gram score", 0.900),
                         ("Similarity",   0.40)):
        fig.add_annotation(text=f"<b>{label}</b>", xref="paper", yref="paper",
                           x=-0.060, y=mid_y, xanchor="center", yanchor="middle",
                           textangle=-90, showarrow=False,
                           font=dict(size=14, color=LABEL))

    apply_style(fig, title=None, width=900, height=920, legend=True)
    fig.update_layout(
        margin=dict(l=80, r=50, t=22, b=10),
        paper_bgcolor=BG, plot_bgcolor=BG,
        legend=dict(orientation="h", yanchor="bottom", y=0.985,
                    xanchor="center", x=0.5,
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12, color=LABEL)),
    )
    save_figure(fig, "language_similarity")

    # ── companion figure: 4 similarity definitions side-by-side ────────
    metrics = []
    for path, title in METRIC_SPEC:
        df_long = (pl.read_ipc(CACHE / path)
                     .filter((pl.col("step_i") > 0) & (pl.col("step_j") > 0)))
        z = df_long.sort(["step_i", "step_j"])["similarity"].to_numpy().reshape(n, n).tolist()
        lo, hi = _bounds(df_long.filter(pl.col("step_i") != pl.col("step_j"))["similarity"])
        metrics.append((z, lo, hi, title))

    # Manual axis-domain placement: 4 equal-width square panels.
    # Fig 1500 x 460, margins 4 → plot 1492 x 452. Each panel x_dom = 0.243
    # → 362.6px. For square: y_dom = 362.6/452 = 0.802. Stack with title
    # above (~0.025 paper height).
    PANEL_X = ((0.0000, 0.2425), (0.2525, 0.4950),
               (0.5050, 0.7475), (0.7575, 1.0000))
    PANEL_Y = (0.020, 0.890)

    fig2 = go.Figure()
    layout_axes = {}
    for k, (z, lo, hi, _) in enumerate(metrics, start=1):
        suffix = "" if k == 1 else str(k)
        x_id, y_id = f"x{suffix}", f"y{suffix}"
        fig2.add_trace(go.Heatmap(z=z, x=indices, y=indices,
                                  zmin=lo, zmax=hi, zmid=0,
                                  colorscale=HEAT_COLORSCALE, showscale=False,
                                  xaxis=x_id, yaxis=y_id))
        layout_axes[f"xaxis{suffix}"] = dict(
            domain=list(PANEL_X[k - 1]), anchor=y_id, range=[-0.5, n - 0.5],
            tickmode="array", tickvals=tick_indices, ticktext=tick_labels,
            ticks="", showline=False, zeroline=False, mirror=False, showgrid=False,
        )
        layout_axes[f"yaxis{suffix}"] = dict(
            domain=list(PANEL_Y), anchor=x_id, range=[-0.5, n - 0.5],
            showticklabels=(k == 1),
            tickmode="array", tickvals=tick_indices, ticktext=tick_labels,
            ticks="", showline=False, zeroline=False, mirror=False, showgrid=False,
        )

    for k, (_, _, _, title) in enumerate(metrics, start=1):
        x_paper = (PANEL_X[k - 1][0] + PANEL_X[k - 1][1]) / 2
        fig2.add_annotation(x=x_paper, y=PANEL_Y[1] + 0.005, xref="paper", yref="paper",
                            text=f"<b>{title}</b>", showarrow=False,
                            xanchor="center", yanchor="bottom",
                            font=dict(size=14, color=LABEL))

    apply_style(fig2, title=None, width=1500, height=460, legend=False)
    fig2.update_layout(
        layout_axes,
        margin=dict(l=4, r=4, t=4, b=4),
        paper_bgcolor=BG, plot_bgcolor=BG,
    )
    save_figure(fig2, "language_similarity_metrics")
