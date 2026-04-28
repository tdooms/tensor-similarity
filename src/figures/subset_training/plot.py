import json

import polars as pl
import plotly.express as px
import plotly.graph_objects as go

from src.figures import CACHE_DIR
from src.figures.style import SUBSET_COLORS, apply_style, save_figure, style_xy_axes

CACHE = CACHE_DIR / "subset_training"
COLORS = {k.replace("_", " "): v for k, v in SUBSET_COLORS.items()}


def main():
    meta = json.loads((CACHE / "metadata.json").read_text(encoding="utf-8"))
    evo = pl.read_ipc(CACHE / "evolution.feather")
    ref = meta["reference_seed"]
    non_ref = evo.filter(pl.col("seed") != ref).sort(["config", "seed", "batch"])
    ref_only = evo.filter(pl.col("seed") == ref).sort(["config", "batch"])

    fig = go.Figure()
    for (config, seed), sub in non_ref.partition_by(["config", "seed"], as_dict=True, maintain_order=True).items():
        fig.add_trace(go.Scatter(
            x=sub["batch"], y=sub["similarity"], mode="lines",
            name=f"{config.replace('_', ' ')} - seed {seed}",
            legendgroup=config, showlegend=False,
            line=dict(color=SUBSET_COLORS[config], width=1.8, dash="dash"),
            opacity=0.45,
        ))
    for (config,), sub in ref_only.partition_by("config", as_dict=True, maintain_order=True).items():
        fig.add_trace(go.Scatter(
            x=sub["batch"], y=sub["similarity"], mode="lines",
            name=config.replace("_", " "),
            legendgroup=config,
            line=dict(color=SUBSET_COLORS[config], width=3, dash="solid"),
        ))
    apply_style(fig, title="MNIST subset training across random seeds")
    style_xy_axes(fig, x_title="Batch steps", y_title="Tensor similarity")
    fig.update_yaxes(range=[0, 1.05])
    save_figure(fig, "subset_training_cross_seed_similarity")

    plot = pl.concat([
        ref_only.select("batch", pl.col("config").str.replace_all("_", " ").alias("config"),
                        pl.lit("similarity gap").alias("metric"),
                        (1 - pl.col("similarity")).clip(1e-6, 1.0).alias("value")),
        ref_only.select("batch", pl.col("config").str.replace_all("_", " ").alias("config"),
                        pl.lit("test error").alias("metric"),
                        (1 - pl.col("test_acc")).clip(1e-6, 1.0).alias("value")),
    ])
    fig = px.line(plot, x="batch", y="value", color="config", line_dash="metric", color_discrete_map=COLORS)
    apply_style(fig, title="MNIST subset training evolution")
    style_xy_axes(fig, x_title="Batch steps", y_title="Residual to 1")
    fig.update_yaxes(type="log")
    save_figure(fig, "subset_training_evolution")
