"""Grokking summary: train/val accuracy + loss + pairwise TN-similarity heatmap."""
import json
import math

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from src.figures.grokking.prepare import CACHE
from src.figures.style import COLORWAY, apply_style, save_figure

ACC_TRACES = (("train_acc", "train", COLORWAY[1]), ("val_acc", "val", COLORWAY[5]))
LOSS_TRACES = (("train_loss", "train", COLORWAY[1]), ("val_loss", "val", COLORWAY[5]))


def main():
    meta = json.loads((CACHE / "metadata.json").read_text(encoding="utf-8"))
    steps = [s for s in meta["steps"] if s > 0]
    history = pl.read_ipc(CACHE / "history.feather").filter(pl.col("step") > 0)
    matrix = (pl.read_ipc(CACHE / "similarity.feather")
                .filter((pl.col("metric") == "tn_similarity")
                        & (pl.col("step_i") > 0) & (pl.col("step_j") > 0))
                .sort(["step_i", "step_j"])["value"]
                .to_numpy().reshape(len(steps), len(steps)).tolist())

    log_x = [math.log10(s) for s in steps]
    log_range = [log_x[0] - (log_x[1] - log_x[0]) / 2,
                 log_x[-1] + (log_x[-1] - log_x[-2]) / 2]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.18, 0.18, 0.64], vertical_spacing=0.04)

    for metric, label, color in ACC_TRACES:
        s = history.filter(pl.col("metric") == metric).sort("step")
        fig.add_trace(go.Scatter(x=s["step"], y=s["value"], mode="lines+markers",
                                 name=f"{label} acc", legendgroup="acc",
                                 line=dict(color=color, width=2.5), marker=dict(size=5)),
                      row=1, col=1)

    for metric, label, color in LOSS_TRACES:
        s = history.filter(pl.col("metric") == metric).sort("step")
        fig.add_trace(go.Scatter(x=s["step"], y=s["value"], mode="lines+markers",
                                 name=f"{label} loss", legendgroup="loss",
                                 line=dict(color=color, width=2.5, dash="dot"), marker=dict(size=5)),
                      row=2, col=1)

    fig.add_trace(go.Heatmap(z=matrix, x=steps, y=steps, zmin=0, zmax=1,
                             colorscale="Viridis",
                             colorbar=dict(title="<b>TN cosine</b>", y=0.32, len=0.55)),
                  row=3, col=1)

    fig.update_xaxes(type="log", range=log_range, row=1, col=1)
    fig.update_xaxes(type="log", range=log_range, row=2, col=1)
    fig.update_xaxes(type="log", range=log_range, title="<b>Training step</b>", row=3, col=1)
    fig.update_yaxes(title="<b>Accuracy</b>", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(type="log", title="<b>Loss</b>", row=2, col=1)
    fig.update_yaxes(type="log", range=log_range, title="<b>Training step</b>", row=3, col=1)
    apply_style(fig, title=f"Grokking on modular addition (P={meta['config']['P']})",
                width=900, height=1100)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.0,
                                  xanchor="right", x=1.0, bgcolor="rgba(255,255,255,0.85)"))
    save_figure(fig, "grokking")
