"""Pairwise similarity across language-model training checkpoints.

Spectrum of similarity definitions in a 2×3 grid. All in TN form
(sums-then-cosine), all at matched input scale (Gaussian σ=0.02 to match
embedding init). The only thing that changes per panel is WHERE the inner
product is taken: behaviourally on Pile activations, on the residual stream,
on Gaussian logits, or in weight space (Tensor / Matrix cosine).

Story flows column-by-column: (1) something is changing — n-gram score
above, Pile · residual matches it; (2) but logits don't pick it up —
Pile · logits, Gaussian · logits; (3) on the weight side, Tensor catches
it, naive Matrix cosine doesn't.

Same blueprint as `svhn-backdoor`: brick↔slate diverging palette anchored at
the paper bg with `zmid=0`, asymmetric data-range bounds so the heatmap
structure isn't compressed into 7% of the palette, manual axis-domain
placement, no figure title, tight image margins.
"""
import json

import plotly.graph_objects as go
import polars as pl

from src.figures.language_similarity.prepare import CACHE
from src.figures.style import apply_style, save_figure

BG    = "#FFFFFF"
LABEL = "#0f172a"
MUTED = "#64748b"

HEAT_COLORSCALE = [
    [0.000, "#7c2d2d"],
    [0.150, "#a8453a"],
    [0.300, "#cf7f70"],
    [0.500, BG],
    [0.700, "#6b8ec0"],
    [0.850, "#33588f"],
    [1.000, "#1c3a72"],
]

NGRAM_TRACES = (
    ("2gram_score", "2-gram", "#a5b4fc"),
    ("3gram_score", "3-gram", "#6366f1"),
    ("4gram_score", "4-gram", "#1c3a72"),  # match heatmap deep-blue endpoint
)

TICK_TARGETS = (100, 1_000, 10_000)
TICK_LABELS  = ("100", "1k", "10k")


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

    behavior = (pl.read_ipc(CACHE / "behavior.feather")
                  .filter(pl.col("step").is_in(set(steps))))

    seen, tick_indices, tick_labels = set(), [], []
    for target, label in zip(TICK_TARGETS, TICK_LABELS):
        idx = min(range(n), key=lambda i: abs(steps[i] - target))
        if idx not in seen:
            seen.add(idx)
            tick_indices.append(idx)
            tick_labels.append(label)

    # 2×3 metrics comparison. Story flows column-by-column: (1) something
    # is changing — n-gram score above, Pile · residual matches it; (2) but
    # logits don't pick it up — Pile · logits, Gaussian · logits; (3) on
    # the weight side, Tensor closed-form catches it, naive Matrix cosine
    # doesn't.
    METRIC_BOUNDARIES = (2.5, 92.5)  # 3 cells from start, 8 from end
    CELLS = (
        # row 1
        (("ngram", None,                                     "n-gram score"),
         ("heat",  "empirical_pile_tnform.feather",          "Pile · logits"),
         ("heat",  "matrix.feather",                         "Tensor")),
        # row 2
        (("heat",  "empirical_pile_residual_tnform.feather", "Pile · residual"),
         ("heat",  "empirical_gaussian_tnform.feather",      "Gaussian · logits"),
         ("heat",  "weight_cosine_matrix.feather",           "Matrix cosine")),
    )
    # Fig 1500 × 1100, margins 4 → plot 1492 × 1092. Each col 0.32 →
    # 477.4 px wide. For square panels, y_dom range = 477/1092 = 0.437.
    # Inter-row gap (~0.060 paper) gives row-2 panel titles breathing room.
    COL_X  = ((0.000, 0.320), (0.340, 0.660), (0.680, 1.000))
    M_ROW1_Y = (0.518, 0.955)
    M_ROW2_Y = (0.021, 0.458)

    fig = go.Figure()
    layout_axes = {}
    axis_num = 0
    for row_idx, row_cells in enumerate(CELLS):
        y_dom = M_ROW1_Y if row_idx == 0 else M_ROW2_Y
        for col_idx, (kind, key, title) in enumerate(row_cells):
            axis_num += 1
            x_dom = COL_X[col_idx]
            suffix = "" if axis_num == 1 else str(axis_num)
            x_id, y_id = f"x{suffix}", f"y{suffix}"
            x_paper = (x_dom[0] + x_dom[1]) / 2

            if kind == "heat":
                df_long = (pl.read_ipc(CACHE / key)
                             .filter((pl.col("step_i") > 0) & (pl.col("step_j") > 0)))
                z = df_long.sort(["step_i", "step_j"])["similarity"].to_numpy().reshape(n, n).tolist()
                lo, hi = _bounds(df_long.filter(pl.col("step_i") != pl.col("step_j"))["similarity"])
                fig.add_trace(go.Heatmap(z=z, x=indices, y=indices,
                                          zmin=lo, zmax=hi, zmid=0,
                                          colorscale=HEAT_COLORSCALE, showscale=False,
                                          xaxis=x_id, yaxis=y_id))
                layout_axes[f"xaxis{suffix}"] = dict(
                    domain=list(x_dom), anchor=y_id, range=[-0.5, n - 0.5],
                    tickmode="array", tickvals=tick_indices, ticktext=tick_labels,
                    ticks="", showline=False, zeroline=False, mirror=False, showgrid=False,
                )
                layout_axes[f"yaxis{suffix}"] = dict(
                    domain=list(y_dom), anchor=x_id, range=[-0.5, n - 0.5],
                    showticklabels=False, ticks="",
                    showline=False, zeroline=False, mirror=False, showgrid=False,
                )

            else:  # n-gram line plot — same content as the main figure
                for metric, label, color in NGRAM_TRACES:
                    s = behavior.filter(pl.col("metric") == metric).sort("step")
                    x = [step_to_idx[step] for step in s["step"].to_list()]
                    y = s["value"].to_list()
                    fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
                                              name=label, legendgroup=metric,
                                              line=dict(color=color, width=2.6),
                                              xaxis=x_id, yaxis=y_id,
                                              showlegend=True))
                    fig.add_annotation(x=x[-1], y=y[-1], xref=x_id, yref=y_id,
                                        text=f"<b>{y[-1]:.2f}</b>",
                                        xanchor="left", yanchor="middle", xshift=4,
                                        showarrow=False,
                                        font=dict(size=11, color=color))
                layout_axes[f"xaxis{suffix}"] = dict(
                    domain=list(x_dom), anchor=y_id, range=[-0.5, n - 0.5],
                    tickmode="array", tickvals=tick_indices, ticktext=tick_labels,
                    ticks="", showline=False, zeroline=False, mirror=False, showgrid=False,
                )
                # n-gram score is bounded below by 1 (uniform-distribution
                # baseline), so start the axis there and show ticks at the
                # half-points for context.
                layout_axes[f"yaxis{suffix}"] = dict(
                    domain=list(y_dom), anchor=x_id, range=[1, 2.6],
                    tickmode="array",
                    tickvals=[1, 1.5, 2, 2.5],
                    ticktext=["1", "1.5", "2", "2.5"],
                    showticklabels=True, ticks="outside",
                    tickfont=dict(size=10, color=MUTED),
                    showgrid=True, gridcolor="#e2e8f0", gridwidth=1,
                    showline=False, zeroline=False, mirror=False,
                )

            fig.add_annotation(x=x_paper, y=y_dom[1] + 0.005, xref="paper", yref="paper",
                                text=f"<b>{title}</b>", showarrow=False,
                                xanchor="center", yanchor="bottom",
                                font=dict(size=17, color=LABEL))

    apply_style(fig, title=None, width=1500, height=1100, legend=True)
    fig.update_layout(
        layout_axes,
        margin=dict(l=4, r=4, t=4, b=4),
        paper_bgcolor=BG, plot_bgcolor=BG,
        legend=dict(orientation="h", yanchor="top",
                    y=M_ROW1_Y[1] - 0.005,
                    xanchor="center", x=(COL_X[0][0] + COL_X[0][1]) / 2,
                    bgcolor="rgba(255,255,255,0.8)",
                    font=dict(size=12, color=LABEL),
                    itemsizing="constant"),
    )
    save_figure(fig, "language_similarity")
