"""Pairwise similarity across language-model training checkpoints.

Same visual blueprint as `grokking-similarity`: index-based x axis,
decade tick labels, off-white paper bg, diverging indigo↔rose heatmap
centered on the bg color, no axis chrome, endpoint labels on the line
traces. Phase boundaries are hand-picked from the visual block structure
of the TN similarity matrix — DP-optimal K-segmentation (used in the
grokking figure) double-penalised the off-diagonal anti-correlated
bands and shifted cuts away from where the blocks actually change here.
"""
import json

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from src.figures.language_similarity.prepare import CACHE
from src.figures.style import apply_style, save_figure

BG       = "#FAFAF7"
LABEL    = "#0f172a"
BOUNDARY = "#0f172a"

NGRAM_TRACES = (
    ("2gram_score", "2-gram", "#a5b4fc"),  # indigo-300
    ("3gram_score", "3-gram", "#6366f1"),  # indigo-500
    ("4gram_score", "4-gram", "#3730a3"),  # indigo-800
)

HEAT_COLORSCALE = [
    [0.0,  "#b2182b"],   # ColorBrewer RdBu (softened endpoints, easy on eyes)
    [0.25, "#f4a582"],
    [0.5,  BG],
    [0.75, "#92c5de"],
    [1.0,  "#2166ac"],
]

# Hand-picked from the visual block structure of the N=101 similarity
# matrix (steps 27 → 20000). Tied to checkpoint-index, not step value:
# rerunning with a different N requires re-picking. Phases get neutral
# `wash_a` / `wash_b` keys (no mechanistic claims like "memorize" /
# "induction" — we have no circuits-level evidence those phases mean
# anything specific in this language model).
PHASES = (
    (  0,   6, ""),
    (  7,  23, "wash_a"),
    ( 24,  51, ""),
    ( 52,  91, "wash_b"),
    ( 92, 100, ""),
)
PHASE_WASH = {"wash_a": "#EDEAE2", "wash_b": "#EDEAE2"}

TICK_TARGETS = (100, 1_000, 10_000)
TICK_LABELS  = ("100", "1k", "10k")


def main():
    meta = json.loads((CACHE / "metadata.json").read_text(encoding="utf-8"))
    steps = [s for s in meta["steps"] if s > 0]
    n = len(steps)
    step_to_idx = {s: i for i, s in enumerate(steps)}
    indices = list(range(n))

    matrix = (pl.read_ipc(CACHE / "matrix.feather")
                .filter((pl.col("step_i") > 0) & (pl.col("step_j") > 0))
                .sort(["step_i", "step_j"])["similarity"]
                .to_numpy().reshape(n, n).tolist())
    behavior = (pl.read_ipc(CACHE / "behavior.feather")
                  .filter(pl.col("step").is_in(set(steps))))

    bounds = []
    left = -0.5
    for i, (_, last, _) in enumerate(PHASES):
        right = (last + PHASES[i + 1][0]) / 2 if i < len(PHASES) - 1 else n - 0.5
        bounds.append((left, right))
        left = right
    boundary_x = [b[0] for b in bounds[1:]]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.018)

    for (x0, x1), (_, _, name) in zip(bounds, PHASES):
        if name not in PHASE_WASH:
            continue
        fig.add_shape(type="rect", xref="x", yref="y domain",
                      x0=x0, x1=x1, y0=0, y1=1,
                      fillcolor=PHASE_WASH[name], line_width=0, layer="below")

    end_values = []
    for metric, label, color in NGRAM_TRACES:
        s = behavior.filter(pl.col("metric") == metric).sort("step")
        x = [step_to_idx[step] for step in s["step"].to_list()]
        y = s["value"].to_list()
        end_values.append((color, x[-1], y[-1]))
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
                                 name=label, legendgroup=metric,
                                 line=dict(color=color, width=2.2)),
                      row=1, col=1)

    fig.add_trace(go.Heatmap(z=matrix, x=indices, y=indices,
                             zmin=-1, zmax=1, zmid=0,
                             colorscale=HEAT_COLORSCALE, showscale=False),
                  row=2, col=1)

    for x in boundary_x:
        for row in (1, 2):
            fig.add_vline(x=x, row=row, col=1, line_color=BOUNDARY, line_width=2.5)

    for color, last_x, last_y in end_values:
        fig.add_annotation(text=f"<b>{last_y:.2f}</b>",
                           x=last_x, y=last_y, row=1, col=1,
                           xanchor="left", yanchor="middle", xshift=5,
                           showarrow=False, font=dict(size=12, color=color))

    seen, tick_indices, tick_labels = set(), [], []
    for target, label in zip(TICK_TARGETS, TICK_LABELS):
        idx = min(range(n), key=lambda i: abs(steps[i] - target))
        if idx not in seen:
            seen.add(idx)
            tick_indices.append(idx)
            tick_labels.append(label)

    fig.update_xaxes(showline=False, zeroline=False, mirror=False,
                     range=[-0.5, n - 0.5], showgrid=False)
    fig.update_xaxes(tickmode="array", tickvals=tick_indices, ticktext=tick_labels,
                     title=dict(text="<b>Training step</b>", font=dict(size=15)),
                     row=2, col=1)
    fig.update_xaxes(ticks="", row=2, col=1)

    fig.update_yaxes(automargin=False, title=None,
                     showline=False, zeroline=False, mirror=False, showgrid=False,
                     showticklabels=False, ticks="")
    fig.update_yaxes(domain=[0.78, 0.92], row=1, col=1)
    fig.update_yaxes(range=[-0.5, n - 0.5], domain=[0.04, 0.76], row=2, col=1)

    for label, mid_y in (("n-gram score", 0.85),
                         ("Similarity",   0.40)):
        fig.add_annotation(text=f"<b>{label}</b>", xref="paper", yref="paper",
                           x=-0.038, y=mid_y, xanchor="center", yanchor="middle",
                           textangle=-90, showarrow=False,
                           font=dict(size=16, color=LABEL))

    apply_style(fig, title="Pairwise checkpoint similarity",
                width=900, height=900)
    fig.update_layout(
        margin=dict(l=85, r=70, t=38, b=10),
        title=dict(y=0.985, font=dict(size=18, color=LABEL)),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        legend=dict(orientation="h", yanchor="middle", y=0.965,
                    xanchor="center", x=0.5,
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(size=13, color=LABEL)),
    )
    save_figure(fig, "language_similarity")

    # ── companion: apples-to-apples spectrum of similarity definitions ─────
    # Four ways to compute pairwise checkpoint similarity, ALL in TN form
    # (cos = E[<f_A, f_B>] / sqrt(E[||f_A||²] · E[||f_B||²]) — sums-then-cosine,
    # never per-sample-cosine-then-mean), all at matched input scale (Gaussian
    # σ=0.02 to match embedding init, so the model stays in fp range and the
    # cosine's scale-invariance for bilinear nets makes σ unbiased). The only
    # thing that changes across panels is WHERE the inner product is taken:
    #   (1) Pile-DSIR text → final logits — the standard empirical baseline.
    #   (2) Pile-DSIR text → penultimate residual — skip the unembed projection.
    #   (3) Gaussian inputs at the embedding output → final logits — MC sim
    #       under TN's matching prior; would equal TN exactly if Wick were
    #       exact for this network.
    #   (4) TN — closed-form Gaussian function-space inner product from weights.
    def _matrix(path):
        return (pl.read_ipc(CACHE / path)
                  .filter((pl.col("step_i") > 0) & (pl.col("step_j") > 0))
                  .sort(["step_i", "step_j"])["similarity"]
                  .to_numpy().reshape(n, n).tolist())

    METRICS = (
        (_matrix("empirical_pile_tnform.feather"),          "Pile · logits"),
        (_matrix("empirical_pile_residual_tnform.feather"), "Pile · residual"),
        (_matrix("empirical_gaussian_tnform.feather"),      "Gaussian · logits"),
        (matrix,                                            "Gaussian · Tensor"),
    )

    fig2 = make_subplots(rows=1, cols=4, shared_yaxes=True, horizontal_spacing=0.018)
    for col, (z, _) in enumerate(METRICS, start=1):
        fig2.add_trace(go.Heatmap(z=z, x=indices, y=indices,
                                   zmin=-1, zmax=1, zmid=0,
                                   colorscale=HEAT_COLORSCALE, showscale=False),
                       row=1, col=col)
    for col, (_, title) in enumerate(METRICS, start=1):
        fig2.add_annotation(x=(col - 0.5) / 4, y=1.02, xref="paper", yref="paper",
                            text=f"<b>{title}</b>", showarrow=False,
                            xanchor="center", yanchor="bottom",
                            font=dict(size=14, color=LABEL))
    fig2.update_xaxes(showline=False, zeroline=False, mirror=False, showgrid=False,
                      range=[-0.5, n - 0.5],
                      tickmode="array", tickvals=tick_indices, ticktext=tick_labels,
                      ticks="")
    fig2.update_yaxes(showline=False, zeroline=False, mirror=False, showgrid=False,
                      range=[-0.5, n - 0.5], showticklabels=False, ticks="")
    fig2.update_yaxes(showticklabels=True, tickmode="array",
                      tickvals=tick_indices, ticktext=tick_labels,
                      row=1, col=1)
    apply_style(fig2, title=None, width=1500, height=440)
    fig2.update_layout(margin=dict(l=46, r=18, t=46, b=18),
                       paper_bgcolor=BG, plot_bgcolor=BG)
    save_figure(fig2, "language_similarity_metrics")
