import json
import math

import polars as pl
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from src.figures import CACHE_DIR
from src.figures.style import COLORWAY, apply_style, save_figure, style_xy_axes

CACHE = CACHE_DIR / "checkpoint_similarity"


def _load():
    steps = json.loads((CACHE / "metadata.json").read_text(encoding="utf-8"))["steps"]
    df = pl.read_ipc(CACHE / "matrix.feather")
    behavior = pl.read_ipc(CACHE / "behavior.feather")
    n = len(steps)
    matrix = torch.zeros(n, n, dtype=torch.float64)
    index = {s: i for i, s in enumerate(steps)}
    for row in df.iter_rows(named=True):
        matrix[index[int(row["step_i"])], index[int(row["step_j"])]] = float(row["similarity"])
    return steps, matrix, behavior


def _series(behavior, metric):
    s = behavior.filter(pl.col("metric") == metric).sort("step")
    return s["step"].to_list(), s["value"].to_list()


def _plot_heatmap(steps, matrix):
    behavior = pl.read_ipc(CACHE / "behavior.feather")
    emp_df = pl.read_ipc(CACHE / "empirical_matrix.feather")
    n = len(steps)
    emp = torch.zeros(n, n, dtype=torch.float64)
    idx = {s: i for i, s in enumerate(steps)}
    for r in emp_df.iter_rows(named=True):
        emp[idx[int(r["step_i"])], idx[int(r["step_j"])]] = float(r["similarity"])

    # Drop step 0 — log axis can't render it (log(0) = -∞), would leave a
    # phantom blank slot for the first row/column.
    steps_v = steps[1:]
    gauss_v = matrix[1:, 1:]
    emp_v = emp[1:, 1:]

    fig = make_subplots(
        rows=2, cols=2, shared_xaxes="columns",
        row_heights=[0.22, 0.78], vertical_spacing=0.04,
        column_widths=[0.5, 0.5], horizontal_spacing=0.06,
        subplot_titles=("<b>Gaussian (TN)</b>", "<b>empirical (tokens)</b>", None, None),
    )
    overlay_colors = ("#0F172A", "#0EA5E9", "#14B8A6")
    for col in (1, 2):
        for i, metric in enumerate(("2gram_score", "3gram_score", "4gram_score")):
            x, y = _series(behavior, metric)
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=metric,
                                     line=dict(color=overlay_colors[i], width=2.5),
                                     legendgroup=metric, showlegend=(col == 1)),
                          row=1, col=col)
    fig.add_trace(go.Heatmap(z=gauss_v.tolist(), x=steps_v, y=steps_v,
                             zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
                             showscale=False),
                  row=2, col=1)
    fig.add_trace(go.Heatmap(z=emp_v.tolist(), x=steps_v, y=steps_v,
                             zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
                             colorbar=dict(title="<b>Cosine</b>", y=0.39, len=0.78)),
                  row=2, col=2)
    for col in (1, 2):
        for step in (298, 1761):
            for r in (1, 2):
                fig.add_vline(x=step, line_color="#475569", line_width=1.5,
                              line_dash="dot", row=r, col=col)
    log_range = [math.log10(steps_v[0]) - 0.05, math.log10(steps_v[-1]) + 0.05]
    for col in (1, 2):
        fig.update_xaxes(type="log", range=log_range, row=1, col=col)
        fig.update_xaxes(type="log", range=log_range,
                         title="<b>Checkpoint step</b>", row=2, col=col)
        fig.update_yaxes(type="log", range=log_range,
                         showticklabels=False, showgrid=False, row=2, col=col)
    fig.update_yaxes(title="<b>n-gram score</b>", row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    apply_style(fig, title="Pairwise checkpoint similarity",
                width=1400, height=900)
    save_figure(fig, "checkpoint_similarity")


def _plot_evolution(steps, matrix, behavior):
    sim_to_final = matrix[:, -1].tolist()

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[1, 1, 1])

    fig.add_trace(go.Scatter(
        x=steps, y=sim_to_final, mode="lines+markers", name="cosine to final",
        line=dict(color=COLORWAY[0], width=3), marker=dict(size=6),
        legendgroup="g1", legendgrouptitle_text="<b>Weight similarity</b>",
    ), row=1, col=1)

    for i, metric in enumerate(("val_loss", "loss_50", "loss_500")):
        x, y = _series(behavior, metric)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=metric,
                                 line=dict(color=COLORWAY[i + 1], width=2),
                                 legendgroup="g2", legendgrouptitle_text="<b>Validation loss</b>"),
                      row=2, col=1)

    for i, metric in enumerate(("2gram_score", "3gram_score", "4gram_score")):
        x, y = _series(behavior, metric)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=metric,
                                 line=dict(color=COLORWAY[i + 4], width=2),
                                 legendgroup="g3", legendgrouptitle_text="<b>n-gram score</b>"),
                      row=3, col=1)

    phases = ((298, "bigram peak"), (1761, "3-gram saturates"))
    for step, label in phases:
        for r in (1, 2, 3):
            fig.add_vline(x=step, line_color="#94A3B8", line_width=1, line_dash="dot",
                          row=r, col=1)
        fig.add_annotation(x=math.log10(step), y=1.0, xref="x1", yref="y1",
                           text=f"<b>{label}</b>", showarrow=False,
                           xanchor="left", yanchor="top", textangle=-25,
                           font=dict(color="#475569", size=11))

    log_range = [math.log10(max(steps[0], 1)) - 0.05, math.log10(steps[-1]) + 0.05]
    for r in (1, 2, 3):
        fig.update_xaxes(type="log", range=log_range, row=r, col=1)
    fig.update_xaxes(title="<b>Checkpoint step</b>", row=3, col=1)
    fig.update_yaxes(range=[-0.3, 1.05], title="<b>cos(step, final)</b>", row=1, col=1)
    fig.update_yaxes(type="log", title="<b>val loss</b>", row=2, col=1)
    fig.update_yaxes(title="<b>n-gram score</b>", row=3, col=1)

    apply_style(fig, title="Checkpoint evolution: weights vs behavior", width=1000, height=900)
    fig.update_layout(legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02,
                                  groupclick="toggleitem", tracegroupgap=14))
    save_figure(fig, "checkpoint_evolution")


def _plot_layer_decomposition(steps, matrix):
    sim_to_final = matrix[:, -1].tolist()
    wc = pl.read_ipc(CACHE / "weight_cosine.feather")
    groups = {
        "embed":   ["embed.weight"],
        "attn_0":  [f"layers.0.{p}.weight" for p in ("q1", "k1", "q2", "k2", "v", "o")],
        "attn_1":  [f"layers.1.{p}.weight" for p in ("q1", "k1", "q2", "k2", "v", "o")],
        "unembed": ["unembed.weight"],
    }
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[1, 1])
    fig.add_trace(go.Scatter(x=steps, y=sim_to_final, mode="lines+markers",
                             name="TN function cos", line=dict(color="#0F172A", width=3)),
                  row=1, col=1)
    for i, (name, params) in enumerate(groups.items()):
        for p in params:
            df = wc.filter(pl.col("param") == p).sort("step")
            fig.add_trace(go.Scatter(x=df["step"], y=df["cos_to_final"], mode="lines",
                                     name=name if p == params[0] else None,
                                     legendgroup=name, showlegend=(p == params[0]),
                                     line=dict(color=COLORWAY[i], width=2),
                                     opacity=1.0 if p == params[0] else 0.55),
                          row=2, col=1)
    log_range = [math.log10(max(steps[0], 1)) - 0.05, math.log10(steps[-1]) + 0.05]
    for r in (1, 2):
        fig.update_xaxes(type="log", range=log_range, row=r, col=1)
    fig.update_xaxes(title="<b>Checkpoint step</b>", row=2, col=1)
    fig.update_yaxes(range=[-0.3, 1.05], title="<b>function cos→final</b>", row=1, col=1)
    fig.update_yaxes(range=[-0.05, 1.05], title="<b>weight cos→final</b>", row=2, col=1)
    apply_style(fig, title="Layer-wise convergence: function-space vs weight-space",
                width=1000, height=720)
    save_figure(fig, "checkpoint_layer_decomposition", experimental=True)


def _plot_pca(steps):
    pca = pl.read_ipc(CACHE / "pca.feather").sort("step")
    summary = json.loads((CACHE / "pca_summary.json").read_text(encoding="utf-8"))
    e1, e2 = summary["explained_variance"][0], summary["explained_variance"][1]
    fig = go.Figure(go.Scatter(
        x=pca["pc1"], y=pca["pc2"], mode="lines+markers",
        text=[f"step {s}" for s in pca["step"]], hoverinfo="text",
        marker=dict(size=10, color=[math.log1p(s) for s in pca["step"]], colorscale="Viridis",
                    colorbar=dict(title="<b>log step</b>"),
                    line=dict(color="white", width=1)),
        line=dict(color="#94A3B8", width=1.5),
    ))
    apply_style(fig, title="Weight-trajectory PCA (top 2 components capture 89% of variance)",
                legend=False, width=820, height=720)
    style_xy_axes(fig, x_title=f"PC1 ({e1*100:.1f}%)",
                  y_title=f"PC2 ({e2*100:.1f}%)", x_grid=True, y_grid=True)
    save_figure(fig, "checkpoint_pca", experimental=True)


def _plot_tn_vs_empirical(steps, matrix):
    sim_to_final = matrix[:, -1].tolist()
    emp = pl.read_ipc(CACHE / "empirical_cosine.feather").sort("step")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=sim_to_final, mode="lines+markers",
                             name="TN cos (Gaussian inputs)",
                             line=dict(color=COLORWAY[0], width=3), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=emp["step"], y=emp["cos_to_final"], mode="lines+markers",
                             name="empirical cos (random tokens)",
                             line=dict(color=COLORWAY[5], width=3, dash="dash"),
                             marker=dict(size=6, symbol="diamond")))
    log_range = [math.log10(max(steps[0], 1)) - 0.05, math.log10(steps[-1]) + 0.05]
    fig.update_xaxes(type="log", range=log_range, title="<b>Checkpoint step</b>")
    fig.update_yaxes(range=[-0.3, 1.05], title="<b>cos(step, final)</b>")
    fig.add_hline(y=0, line_color="#94A3B8", line_width=1, line_dash="dot")
    apply_style(fig, title="TN (Gaussian) vs empirical (token) cosine to final",
                width=1000, height=620)
    save_figure(fig, "checkpoint_tn_vs_empirical", experimental=True)


def _plot_ablation(steps, matrix):
    path = CACHE / "ablation_cosine.feather"
    if not path.exists():
        return
    abl = pl.read_ipc(path)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=matrix[:, -1].tolist(), mode="lines+markers",
                             name="full (cached TN cos)",
                             line=dict(color="#0F172A", width=3, dash="solid"),
                             marker=dict(size=6)))
    style = {"linear": (COLORWAY[2], "dot"), "attn0_only": (COLORWAY[3], "dash"),
             "attn1_only": (COLORWAY[5], "dashdot"), "full": (COLORWAY[1], "solid")}
    for mode in ("linear", "attn0_only", "attn1_only", "full"):
        df = abl.filter(pl.col("mode") == mode).sort("step")
        color, dash = style[mode]
        fig.add_trace(go.Scatter(x=df["step"], y=df["cos_to_final"], mode="lines+markers",
                                 name=mode, line=dict(color=color, width=2, dash=dash),
                                 marker=dict(size=5)))
    fig.add_hline(y=0, line_color="#94A3B8", line_width=1, line_dash="dot")
    log_range = [math.log10(max(steps[0], 1)) - 0.05, math.log10(steps[-1]) + 0.05]
    fig.update_xaxes(type="log", range=log_range, title="<b>Checkpoint step</b>")
    fig.update_yaxes(range=[-0.3, 1.05], title="<b>cos(step, final)</b>")
    apply_style(fig, title="Layer ablation: localising the negative-cosine basin",
                width=1000, height=620)
    save_figure(fig, "checkpoint_ablation", experimental=True)


def _plot_all_metrics(behavior):
    metrics = sorted(set(behavior["metric"].to_list()))
    cols = 3
    rows = (len(metrics) + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, vertical_spacing=0.045, horizontal_spacing=0.06,
                        subplot_titles=metrics)
    for k, metric in enumerate(metrics):
        r, c = k // cols + 1, k % cols + 1
        x, y = _series(behavior, metric)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
                                 line=dict(color=COLORWAY[k % len(COLORWAY)], width=2),
                                 showlegend=False),
                      row=r, col=c)
        fig.update_xaxes(type="log", row=r, col=c)
    apply_style(fig, title="All behavior metrics over training", width=1200, height=180 * rows, legend=False)
    save_figure(fig, "checkpoint_all_metrics", experimental=True)


def main():
    steps, matrix, behavior = _load()
    _plot_heatmap(steps, matrix)
    _plot_evolution(steps, matrix, behavior)
    _plot_all_metrics(behavior)
    _plot_layer_decomposition(steps, matrix)
    _plot_pca(steps)
    _plot_tn_vs_empirical(steps, matrix)
    _plot_ablation(steps, matrix)
