"""Pairwise checkpoint-similarity heatmap from prepared data."""
import json
import math

import plotly.graph_objects as go
import polars as pl

from src.figures.language_similarity.prepare import CACHE
from src.figures.style import apply_style, save_figure


def main():
    steps = [s for s in json.loads((CACHE / "metadata.json").read_text(encoding="utf-8"))["steps"] if s > 0]
    z = (pl.read_ipc(CACHE / "matrix.feather")
           .filter((pl.col("step_i") > 0) & (pl.col("step_j") > 0))
           .sort(["step_i", "step_j"])["similarity"]
           .to_numpy().reshape(len(steps), len(steps)).tolist())

    log_range = [math.log10(steps[0]) - 0.05, math.log10(steps[-1]) + 0.05]
    fig = go.Figure(go.Heatmap(z=z, x=steps, y=steps, zmin=-1, zmax=1,
                               colorscale="RdBu", reversescale=True,
                               colorbar=dict(title="<b>Cosine</b>")))
    fig.update_xaxes(type="log", range=log_range, title="<b>Checkpoint step</b>")
    fig.update_yaxes(type="log", range=log_range, title="<b>Checkpoint step</b>")
    apply_style(fig, title="Pairwise checkpoint similarity", width=900, height=820)
    save_figure(fig, "language_similarity")
