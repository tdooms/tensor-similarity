import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from src.figures import CACHE_DIR
from src.figures.style import CURRICULUM_COLORS, apply_style, save_figure, style_xy_axes

CACHE = CACHE_DIR / "curriculum_shift"


def _pretty(stage):
    return stage.replace("_", " ")


def main():
    traj = pl.read_ipc(CACHE / "trajectory.feather")
    acc = pl.read_ipc(CACHE / "accuracy.feather")
    heat = pl.read_ipc(CACHE / "heatmap.feather")

    fig = px.line(
        traj.sort(["stage", "batch"]).with_columns(pl.col("stage").str.replace_all("_", " ").alias("stage")),
        x="batch",
        y="similarity",
        color="stage",
        color_discrete_map={_pretty(k): v for k, v in CURRICULUM_COLORS.items()},
    )
    apply_style(fig, title="Curriculum shift to the final model")
    style_xy_axes(fig, x_title="Cumulative batch steps", y_title="Tensor similarity")
    fig.update_yaxes(range=[0, 1.05])
    save_figure(fig, "curriculum_shift_trajectory")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for stage, frame in traj.partition_by("stage", as_dict=True).items():
        fig.add_trace(
            go.Scatter(
                x=frame["batch"],
                y=frame["similarity"],
                mode="lines",
                name=_pretty(stage[0]),
                line=dict(color=CURRICULUM_COLORS.get(stage[0], "#64748B"), width=3),
            ),
            secondary_y=False,
        )
    for stage, frame in acc.partition_by("stage", as_dict=True).items():
        fig.add_trace(
            go.Scatter(
                x=frame["batch"],
                y=frame["val_acc"],
                mode="lines",
                name=f"{_pretty(stage[0])} accuracy",
                showlegend=False,
                line=dict(color=CURRICULUM_COLORS.get(stage[0], "#64748B"), width=2, dash="dash"),
                opacity=0.55,
            ),
            secondary_y=True,
        )
    apply_style(fig, title="Curriculum shift similarity and test accuracy")
    style_xy_axes(fig, x_title="Cumulative batch steps", y_title="Tensor similarity")
    fig.update_yaxes(range=[0, 1.05], secondary_y=False)
    fig.update_yaxes(title_text="<b>Test accuracy</b>", range=[0.8, 1.0], secondary_y=True)
    save_figure(fig, "curriculum_shift_similarity_accuracy")

    batches = sorted(set(heat["batch_i"].to_list()))
    index = {batch: i for i, batch in enumerate(batches)}
    matrix = torch.zeros(len(batches), len(batches), dtype=torch.float64)
    stages = {}
    for row in heat.iter_rows(named=True):
        matrix[index[int(row["batch_i"])], index[int(row["batch_j"])]] = float(row["similarity"])
        stages[int(row["batch_i"])] = row["stage_i"]

    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=batches,
            y=batches,
            zmin=-1,
            zmax=1,
            coloraxis="coloraxis",
            hovertemplate="x=%{x}<br>y=%{y}<br>similarity=%{z:.3f}<extra></extra>",
        )
    )
    last = stages[batches[0]]
    for batch in batches:
        if stages[batch] != last:
            color = CURRICULUM_COLORS.get(stages[batch], "#64748B")
            fig.add_vline(x=batch, line_color=color, line_width=2, opacity=0.8)
            fig.add_hline(y=batch, line_color=color, line_width=2, opacity=0.8)
            last = stages[batch]
    apply_style(fig, title="Curriculum shift checkpoint heatmap", legend=False, width=980, height=820)
    fig.update_layout(coloraxis=dict(colorscale="RdBu", reversescale=True, colorbar=dict(title="<b>Tensor similarity</b>")))
    style_xy_axes(fig, x_title="Cumulative batch step", y_title="Cumulative batch step", x_grid=False, y_grid=False)
    save_figure(fig, "curriculum_shift_heatmap")
