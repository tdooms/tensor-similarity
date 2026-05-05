"""Grokking summary on a checkpoint-INDEX axis (not log-step).

Each checkpoint occupies one unit on the x axis, so the heatmap cells are
uniform 1×1 squares. The bottom axis labels show step values at hand-picked
sparse indices instead of plotly's automatic log ticks.

Five DP-optimal phases (`start → memorize → [plateau] → grok → consolidate`);
memorize and grok get a subtle warm-gray wash on the line panels as a visual
cue for the two canonical "events" of grokking. Train in muted slate-400,
validation in slate-800. Heatmap on the brick↔slate diverging palette
anchored to the paper bg (matches svhn-backdoor, svhn-forgetting, svhn-diffing).
"""
import json
import math

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from src.figures.grokking_similarity.prepare import CACHE
from src.figures.style import apply_style, save_figure

BG    = "#FFFFFF"
LABEL = "#0f172a"

# Neutral train/val (medium-gray vs near-black) — the line carries the
# information, the legend pulls them apart by name not by hue.
TRAIN = "#94a3b8"   # slate-400
VAL   = "#1e293b"   # slate-800

# Subtle warm-gray wash on memorize and grok line panels only.
PHASE_WASH = {
    "memorize": "#EDEAE2",
    "grok":     "#EDEAE2",
}

# Diverging palette: brick ↔ slate-navy through the paper bg, matching the
# svhn family. Saturated extremes for paper rendering, warmer/cooler hues
# than primary R/B.
HEAT_COLORSCALE = [
    [0.000, "#7c2d2d"],
    [0.150, "#a8453a"],
    [0.300, "#cf7f70"],
    [0.500, BG],
    [0.700, "#6b8ec0"],
    [0.850, "#33588f"],
    [1.000, "#1c3a72"],
]
FREQ_COLORSCALE = [[0.0, BG], [1.0, "#1c3a72"]]   # matches heatmap deep slate
BOUNDARY_LINE_KW = dict(line_color=LABEL, line_width=2.5)

TICK_TARGETS = (1, 10, 100, 1_000, 10_000, 100_000)
TICK_LABELS  = ("1", "10", "100", "1k", "10k", "100k")


def main():
    meta = json.loads((CACHE / "metadata.json").read_text(encoding="utf-8"))
    steps = [s for s in meta["steps"] if s > 0]
    n = len(steps)
    step_to_idx = {s: i for i, s in enumerate(steps)}
    phases = meta["phases"]

    history = pl.read_ipc(CACHE / "history.feather").filter(pl.col("step") > 0)
    sim_long = (pl.read_ipc(CACHE / "similarity.feather")
                  .filter((pl.col("metric") == "tn_similarity")
                          & (pl.col("step_i") > 0) & (pl.col("step_j") > 0)))
    matrix = (sim_long.sort(["step_i", "step_j"])["value"]
                      .to_numpy().reshape(n, n).tolist())
    # Asymmetric data-range bounds with zmid=0 — same trick the svhn heatmaps
    # use: keeps white at true 0 (sign is honest) but stretches the palette
    # over where the data actually lives, instead of leaving half the colormap
    # unused for [-1, 0] when grokking similarities sit mostly in [0, 1].
    sim_off_diag = sim_long.filter(pl.col("step_i") != pl.col("step_j"))["value"]
    sim_zmin = min(float(sim_off_diag.quantile(0.01)), -0.01)
    sim_zmax = max(float(sim_off_diag.quantile(0.99)),  0.01)
    freq = (pl.read_ipc(CACHE / "freq_marginals.feather")
              .filter(pl.col("step") > 0)
              .with_columns(value=pl.col("value") / pl.col("value").sum().over("step"))
              .sort(["freq_idx", "step"]))
    n_freqs = freq["freq_idx"].n_unique()
    freq_z = freq["value"].to_numpy().reshape(n_freqs, n).tolist()

    phase_idx = [(step_to_idx.get(p["first_step"], 0),
                  step_to_idx[p["last_step"]],
                  p["name"])
                 for p in phases]

    bounds = []
    left = -0.5
    for i, (_, last, _) in enumerate(phase_idx):
        right = (last + phase_idx[i + 1][0]) / 2 if i < len(phase_idx) - 1 else n - 0.5
        bounds.append((left, right))
        left = right

    indices = list(range(n))
    indices_freq = list(range(n_freqs))

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.018)

    marker = lambda color: dict(size=8, color=color, line=dict(color=BG, width=1.8))

    for metric, name, color in (("train_acc", "Train",      TRAIN),
                                ("val_acc",   "Validation", VAL)):
        s = history.filter(pl.col("metric") == metric).sort("step")
        x = [step_to_idx[step] for step in s["step"].to_list()]
        fig.add_trace(go.Scatter(x=x, y=s["value"].to_list(), mode="lines+markers",
                                 name=name, legendgroup=name.lower(),
                                 line=dict(color=color, width=2.2),
                                 marker=marker(color)),
                      row=1, col=1)

    for metric, name, color in (("train_loss", "Train",      TRAIN),
                                ("val_loss",   "Validation", VAL)):
        s = history.filter(pl.col("metric") == metric).sort("step")
        x = [step_to_idx[step] for step in s["step"].to_list()]
        fig.add_trace(go.Scatter(x=x, y=s["value"].to_list(), mode="lines+markers",
                                 name=name, legendgroup=name.lower(), showlegend=False,
                                 line=dict(color=color, width=2.2),
                                 marker=marker(color)),
                      row=2, col=1)

    fig.add_trace(go.Heatmap(z=matrix, x=indices, y=indices,
                             zmin=sim_zmin, zmax=sim_zmax, zmid=0,
                             colorscale=HEAT_COLORSCALE, showscale=False),
                  row=3, col=1)

    fig.add_trace(go.Heatmap(z=freq_z, x=indices, y=indices_freq,
                             colorscale=FREQ_COLORSCALE, showscale=False),
                  row=4, col=1)

    # Two boundary vlines: 4 cells from start, 8 cells from end.
    for x_b in (3.5, n - 8.5):
        for r in (1, 2, 3, 4):
            fig.add_vline(x=x_b, row=r, col=1, **BOUNDARY_LINE_KW)

    tick_indices = [min(range(n), key=lambda i: abs(steps[i] - t)) for t in TICK_TARGETS]

    fig.update_xaxes(showline=False, zeroline=False, mirror=False,
                     range=[-0.5, n - 0.5], showgrid=False)
    fig.update_xaxes(tickmode="array", tickvals=tick_indices, ticktext=list(TICK_LABELS),
                     title=dict(text="<b>Training step</b>", font=dict(size=15)),
                     row=4, col=1)
    fig.update_xaxes(ticks="", row=3, col=1)
    fig.update_xaxes(ticks="", row=4, col=1)

    fig.update_yaxes(automargin=False, title=None,
                     showline=False, zeroline=False, mirror=False, showgrid=False,
                     showticklabels=False, ticks="")
    fig.update_yaxes(range=[-0.04, 1.07], domain=[0.755, 0.905], row=1, col=1)
    fig.update_yaxes(type="log", range=[-6.7, 1.3], domain=[0.585, 0.735], row=2, col=1)
    fig.update_yaxes(range=[-0.5, n - 0.5], domain=[0.295, 0.565], row=3, col=1)
    fig.update_yaxes(domain=[0.015, 0.275], row=4, col=1)

    # Endpoint labels: color-matched annotations at the right edge of each
    # line panel, showing where the curve actually lands. Replaces the
    # left-side y-axis ticks with information that's anchored to the data.
    def _fmt_log(v):
        sup = "⁰¹²³⁴⁵⁶⁷⁸⁹"
        e = int(math.floor(math.log10(v)))
        mantissa = v / 10 ** e
        super_e = ("⁻" + sup[abs(e)]) if e < 0 else sup[e]
        return f"{mantissa:.1f}·10{super_e}"

    train_acc_end = float(history.filter(pl.col("metric") == "train_acc")
                                 .sort("step")["value"].to_list()[-1])
    val_acc_end   = float(history.filter(pl.col("metric") == "val_acc")
                                 .sort("step")["value"].to_list()[-1])
    train_loss_end = float(history.filter(pl.col("metric") == "train_loss")
                                  .sort("step")["value"].to_list()[-1])
    val_loss_end   = float(history.filter(pl.col("metric") == "val_loss")
                                  .sort("step")["value"].to_list()[-1])
    last_x = n - 1
    # Plotly annotations on a log y-axis want y in LOG10 space, not data space.
    # Linear axes take the raw data value as-is.
    for text, y, color, row, log_y in (
        (f"{train_acc_end:.2f}",   train_acc_end,  TRAIN, 1, False),
        (f"{val_acc_end:.2f}",     val_acc_end,    VAL,   1, False),
        (_fmt_log(train_loss_end), train_loss_end, TRAIN, 2, True),
        (_fmt_log(val_loss_end),   val_loss_end,   VAL,   2, True),
    ):
        y_pos = math.log10(y) if log_y else y
        fig.add_annotation(text=f"<b>{text}</b>", x=last_x, y=y_pos, row=row, col=1,
                           xanchor="left", yanchor="middle", xshift=5,
                           showarrow=False, font=dict(size=12, color=color))

    for label, mid_y in (("Accuracy",   0.830),
                          ("Loss (log)", 0.660),
                          ("Similarity", 0.430),
                          ("Frequency",  0.145)):
        fig.add_annotation(text=f"<b>{label}</b>", xref="paper", yref="paper",
                           x=-0.012, y=mid_y, xanchor="center", yanchor="middle",
                           textangle=-90, showarrow=False,
                           font=dict(size=14, color=LABEL))

    apply_style(fig, title=None, width=920, height=900, legend=True)
    fig.update_layout(
        margin=dict(l=46, r=64, t=8, b=12),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        legend=dict(orientation="h", yanchor="bottom", y=0.918,
                    xanchor="center", x=0.5,
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(size=13, color=LABEL)),
    )
    save_figure(fig, "grokking_similarity")
