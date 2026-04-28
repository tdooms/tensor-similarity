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

    fig = go.Figure()
    for config in meta["configs"]:
        for seed in meta["seeds"]:
            frame = evo.filter((pl.col("config") == config) & (pl.col("seed") == seed)).sort("batch")
            fig.add_trace(
                go.Scatter(
                    x=frame["batch"],
                    y=frame["similarity"],
                    mode="lines",
                    name=f"{config.replace('_', ' ')} - seed {seed}",
                    legendgroup=config,
                    showlegend=seed == ref,
                    line=dict(color=SUBSET_COLORS.get(config, "#64748B"), width=3 if seed == ref else 1.8, dash="solid" if seed == ref else "dash"),
                    opacity=1.0 if seed == ref else 0.45,
                )
            )
    apply_style(fig, title="MNIST subset training across random seeds")
    style_xy_axes(fig, x_title="Batch steps", y_title="Tensor similarity")
    fig.update_yaxes(range=[0, 1.05])
    save_figure(fig, "subset_training_cross_seed_similarity")

    plot = pl.concat([
        evo.filter(pl.col("seed") == ref).select("batch", pl.col("config").str.replace_all("_", " ").alias("config"), pl.lit("similarity gap").alias("metric"), (1 - pl.col("similarity")).clip(1e-6, 1.0).alias("value")),
        evo.filter(pl.col("seed") == ref).select("batch", pl.col("config").str.replace_all("_", " ").alias("config"), pl.lit("test error").alias("metric"), (1 - pl.col("test_acc")).clip(1e-6, 1.0).alias("value")),
    ])
    fig = px.line(plot, x="batch", y="value", color="config", line_dash="metric", color_discrete_map=COLORS)
    apply_style(fig, title="MNIST subset training evolution")
    style_xy_axes(fig, x_title="Batch steps", y_title="Residual to 1")
    fig.update_yaxes(type="log")
    save_figure(fig, "subset_training_evolution")
