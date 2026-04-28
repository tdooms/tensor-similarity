import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.figures import CACHE_DIR
from src.figures.style import CURRICULUM_COLORS, apply_style, save_figure, style_xy_axes

CACHE = CACHE_DIR / "curriculum_shift"
PRETTY_COLORS = {k.replace("_", " "): v for k, v in CURRICULUM_COLORS.items()}


def main():
    traj = pl.read_ipc(CACHE / "trajectory.feather")
    acc = pl.read_ipc(CACHE / "accuracy.feather")
    heat = pl.read_ipc(CACHE / "heatmap.feather")

    fig = px.line(
        traj.sort(["stage", "batch"]).with_columns(pl.col("stage").str.replace_all("_", " ").alias("stage")),
        x="batch", y="similarity", color="stage", color_discrete_map=PRETTY_COLORS,
    )
    apply_style(fig, title="Curriculum shift to the final model")
    style_xy_axes(fig, x_title="Cumulative batch steps", y_title="Tensor similarity")
    fig.update_yaxes(range=[0, 1.05])
    save_figure(fig, "curriculum_shift_trajectory")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for (stage,), frame in traj.partition_by("stage", as_dict=True, maintain_order=True).items():
        fig.add_trace(
            go.Scatter(x=frame["batch"], y=frame["similarity"], mode="lines",
                       name=stage.replace("_", " "),
                       line=dict(color=CURRICULUM_COLORS[stage], width=3)),
            secondary_y=False,
        )
    for (stage,), frame in acc.partition_by("stage", as_dict=True, maintain_order=True).items():
        fig.add_trace(
            go.Scatter(x=frame["batch"], y=frame["val_acc"], mode="lines",
                       name=f"{stage.replace('_', ' ')} accuracy", showlegend=False,
                       line=dict(color=CURRICULUM_COLORS[stage], width=2, dash="dash"),
                       opacity=0.55),
            secondary_y=True,
        )
    apply_style(fig, title="Curriculum shift similarity and test accuracy")
    style_xy_axes(fig, x_title="Cumulative batch steps", y_title="Tensor similarity")
    fig.update_yaxes(range=[0, 1.05], secondary_y=False)
    fig.update_yaxes(title_text="<b>Test accuracy</b>", range=[0.8, 1.0], secondary_y=True)
    save_figure(fig, "curriculum_shift_similarity_accuracy")

    matrix = (heat.sort(["batch_i", "batch_j"])
                  .pivot(values="similarity", index="batch_i", on="batch_j", sort_columns=True)
                  .drop("batch_i").to_numpy())
    batches = sorted(set(heat["batch_i"].to_list()))
    boundaries = (heat.select("batch_i", "stage_i").unique()
                      .sort("batch_i")
                      .with_columns(prev_stage=pl.col("stage_i").shift(1))
                      .filter(pl.col("stage_i") != pl.col("prev_stage"))
                      .filter(pl.col("prev_stage").is_not_null()))

    fig = go.Figure(go.Heatmap(z=matrix, x=batches, y=batches, zmin=-1, zmax=1,
                               coloraxis="coloraxis",
                               hovertemplate="x=%{x}<br>y=%{y}<br>similarity=%{z:.3f}<extra></extra>"))
    for batch, stage in zip(boundaries["batch_i"], boundaries["stage_i"]):
        color = CURRICULUM_COLORS[stage]
        fig.add_vline(x=batch, line_color=color, line_width=2, opacity=0.8)
        fig.add_hline(y=batch, line_color=color, line_width=2, opacity=0.8)
    apply_style(fig, title="Curriculum shift checkpoint heatmap", legend=False, width=980, height=820)
    fig.update_layout(coloraxis=dict(colorscale="RdBu", reversescale=True,
                                     colorbar=dict(title="<b>Tensor similarity</b>")))
    style_xy_axes(fig, x_title="Cumulative batch step", y_title="Cumulative batch step",
                  x_grid=False, y_grid=False)
    save_figure(fig, "curriculum_shift_heatmap")
