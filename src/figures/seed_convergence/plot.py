import json

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.figures import CACHE_DIR
from src.figures.style import COLORWAY, apply_style, save_figure, style_xy_axes

CACHE = CACHE_DIR / "seed_convergence"


def main():
    meta = json.loads((CACHE / "metadata.json").read_text(encoding="utf-8"))
    sim = pl.read_ipc(CACHE / "similarity.feather")
    hist = pl.read_ipc(CACHE / "history.feather")
    ref = meta["reference_seed"]
    ref_sim = sim.filter(pl.col("seed") == ref).sort("batch")
    ref_hist = hist.filter(pl.col("seed") == ref).sort("batch")
    non_ref = sim.filter(pl.col("seed") != ref).sort(["seed", "batch"])

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=ref_sim["batch"], y=ref_sim["similarity"], mode="lines",
                             name="Tensor similarity", line=dict(color=COLORWAY[0], width=3)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=ref_hist["batch"], y=ref_hist["train_acc"], mode="lines",
                             name="Train accuracy", line=dict(color=COLORWAY[1], width=2.5, dash="dot")),
                  secondary_y=True)
    fig.add_trace(go.Scatter(x=ref_hist["batch"], y=ref_hist["val_acc"], mode="lines",
                             name="Test accuracy", line=dict(color=COLORWAY[2], width=2.5, dash="dash")),
                  secondary_y=True)
    apply_style(fig, title="Seed convergence: tensor similarity and accuracy")
    style_xy_axes(fig, x_title="Batch steps", y_title="Tensor similarity")
    fig.update_yaxes(title_text="<b>Accuracy</b>", range=[0, 1.05], secondary_y=True)
    fig.update_yaxes(range=[0, 1.05], secondary_y=False)
    save_figure(fig, "seed_convergence_similarity_accuracy")

    fig = go.Figure()
    for i, ((seed,), sub) in enumerate(non_ref.partition_by("seed", as_dict=True, maintain_order=True).items()):
        fig.add_trace(go.Scatter(
            x=sub["batch"], y=sub["similarity"], mode="lines",
            name=f"Seed {seed}",
            line=dict(width=1.8, dash="dash", color=COLORWAY[(i + 1) % len(COLORWAY)]),
            opacity=0.62,
        ))
    fig.add_trace(go.Scatter(
        x=ref_sim["batch"], y=ref_sim["similarity"], mode="lines",
        name=f"Seed {ref}",
        line=dict(width=3, dash="solid", color=COLORWAY[0]),
    ))
    apply_style(fig, title="Seed convergence across random seeds")
    style_xy_axes(fig, x_title="Batch steps", y_title="Tensor similarity")
    fig.update_yaxes(range=[0, 1.05])
    save_figure(fig, "seed_convergence_cross_seed")
