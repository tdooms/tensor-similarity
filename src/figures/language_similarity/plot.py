"""Pairwise checkpoint-similarity heatmap with n-gram score overlay."""
import json
import math

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from src.figures.language_similarity.prepare import CACHE
from src.figures.style import COLORWAY, apply_style, save_figure

NGRAM_METRICS = (("2gram_score", "2-gram"), ("3gram_score", "3-gram"), ("4gram_score", "4-gram"))


def main():
    steps = [s for s in json.loads((CACHE / "metadata.json").read_text(encoding="utf-8"))["steps"] if s > 0]
    matrix = (pl.read_ipc(CACHE / "matrix.feather")
                .filter((pl.col("step_i") > 0) & (pl.col("step_j") > 0))
                .sort(["step_i", "step_j"])["similarity"]
                .to_numpy().reshape(len(steps), len(steps)).tolist())
    behavior = pl.read_ipc(CACHE / "behavior.feather")

    log_x = [math.log10(s) for s in steps]
    log_range = [log_x[0] - (log_x[1] - log_x[0]) / 2,
                 log_x[-1] + (log_x[-1] - log_x[-2]) / 2]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.25, 0.75], vertical_spacing=0.04)

    for i, (metric, label) in enumerate(NGRAM_METRICS):
        series = behavior.filter(pl.col("metric") == metric).sort("step")
        fig.add_trace(go.Scatter(x=series["step"], y=series["value"], mode="lines",
                                 name=label, line=dict(color=COLORWAY[i], width=2.5)),
                      row=1, col=1)

    fig.add_trace(go.Heatmap(z=matrix, x=steps, y=steps, zmin=-1, zmax=1,
                             colorscale="RdBu", reversescale=True,
                             colorbar=dict(title="<b>Cosine</b>", y=0.32, len=0.65)),
                  row=2, col=1)

    fig.update_xaxes(type="log", range=log_range, row=1, col=1)
    fig.update_xaxes(type="log", range=log_range, title="<b>Checkpoint step</b>", row=2, col=1)
    fig.update_yaxes(title="<b>n-gram score</b>", row=1, col=1)
    fig.update_yaxes(type="log", range=log_range, row=2, col=1)
    apply_style(fig, title="Pairwise checkpoint similarity", width=900, height=1000)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=0.78,
                                  xanchor="right", x=0.98, bgcolor="rgba(255,255,255,0.85)"))
    save_figure(fig, "language_similarity")
