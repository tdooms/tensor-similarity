"""SVHN backdoor: weight vs functional similarity through a mid-training trigger.

Two-phase training of a 1-layer bilinear model (Bilinear(1024 → 64) → Linear):
8000 clean steps, then 2000 steps with 10% per-batch poisoning (5×7 black
diamond top-right → label 9). The backdoor onset at step 8000 produces a
sharp phase block in functional similarity on poisoned eval, but barely
shifts weight-space TN sim or clean-eval cosine — a visual demonstration
that empirical similarity depends on the choice of probe distribution.

Same blueprint as grokking-similarity: 4 stacked rows on a checkpoint-INDEX
axis (uniform 1×1 cells), off-white bg, diverging RdBu heatmap centered on
bg, dark vline at the phase boundary, soft rose wash on the poisoned phase.
"""
import json

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from src.figures.style import apply_style, save_figure
from src.figures.svhn_backdoor.prepare import CACHE

BG       = "#FAFAF7"
TRAIN    = "#4f46e5"   # indigo-600
TEST     = "#0ea5e9"   # sky-500
DANGER   = "#e11d48"   # rose-600 — ASR / poison_loss
LABEL    = "#0f172a"
BOUNDARY = "#0f172a"

HEAT_COLORSCALE = [
    [0.0,  "#b2182b"],
    [0.25, "#f4a582"],
    [0.5,  BG],
    [0.75, "#92c5de"],
    [1.0,  "#2166ac"],
]
PHASE_WASH = {"poisoned": "#fce4ec"}
BOUNDARY_LINE_KW = dict(line_color=BOUNDARY, line_width=2.5)

TICK_TARGETS = (0, 2_000, 4_000, 6_000, 8_000, 10_000)
TICK_LABELS  = ("0", "2k", "4k", "6k", "8k", "10k")


def main():
    meta = json.loads((CACHE / "metadata.json").read_text(encoding="utf-8"))
    steps = meta["steps"]
    n = len(steps)
    step_to_idx = {s: i for i, s in enumerate(steps)}
    phases = meta["phases"]

    history = pl.read_ipc(CACHE / "history.feather")
    sim_long = pl.read_ipc(CACHE / "similarity.feather")

    def _matrix(metric):
        return (sim_long.filter(pl.col("metric") == metric)
                        .sort(["step_i", "step_j"])["value"]
                        .to_numpy().reshape(n, n).tolist())

    def _bounds(metric):
        # Off-diagonal percentile bounds give the heatmap real contrast in the
        # range where the data actually lives — fixed [-1, 1] would compress
        # all checkpoint pairs into the upper half of the colorscale and wash
        # out the phase-block structure.
        off = sim_long.filter((pl.col("metric") == metric)
                              & (pl.col("step_i") != pl.col("step_j")))["value"]
        return float(off.quantile(0.01)), float(off.quantile(0.99))

    tn_sim = _matrix("tn_sim")
    act_poisoned = _matrix("act_poisoned_full")

    phase_idx = [(step_to_idx[p["first_step"]],
                  step_to_idx[p["last_step"]],
                  p["name"]) for p in phases]

    bounds = []
    left = -0.5
    for i, (_, last, _) in enumerate(phase_idx):
        right = (last + phase_idx[i + 1][0]) / 2 if i < len(phase_idx) - 1 else n - 0.5
        bounds.append((left, right))
        left = right
    boundary_x = [b[0] for b in bounds[1:]]

    indices = list(range(n))

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.018)

    for (x0, x1), (_, _, name) in zip(bounds, phase_idx):
        if name not in PHASE_WASH:
            continue
        for ax in ("", "2"):
            fig.add_shape(type="rect", xref=f"x{ax}", yref=f"y{ax} domain",
                          x0=x0, x1=x1, y0=0, y1=1,
                          fillcolor=PHASE_WASH[name], line_width=0, layer="below")

    marker = lambda color: dict(size=7, color=color, line=dict(color=BG, width=1.5))

    for metric, name, color, group in (("train_acc",           "Train",          TRAIN,  "train"),
                                       ("test_acc",            "Test (clean)",   TEST,   "test"),
                                       ("attack_success_rate", "Attack success", DANGER, "attack")):
        s = history.filter(pl.col("metric") == metric).sort("step")
        x = [step_to_idx[step] for step in s["step"].to_list()]
        fig.add_trace(go.Scatter(x=x, y=s["value"].to_list(), mode="lines+markers",
                                 name=name, legendgroup=group,
                                 line=dict(color=color, width=2.2),
                                 marker=marker(color)),
                      row=1, col=1)

    for metric, color, group in (("train_loss",  TRAIN,  "train"),
                                 ("test_loss",   TEST,   "test"),
                                 ("poison_loss", DANGER, "attack")):
        s = history.filter(pl.col("metric") == metric).sort("step")
        x = [step_to_idx[step] for step in s["step"].to_list()]
        fig.add_trace(go.Scatter(x=x, y=s["value"].to_list(), mode="lines+markers",
                                 legendgroup=group, showlegend=False,
                                 line=dict(color=color, width=2.2),
                                 marker=marker(color)),
                      row=2, col=1)

    tn_lo, tn_hi = _bounds("tn_sim")
    ap_lo, ap_hi = _bounds("act_poisoned_full")
    fig.add_trace(go.Heatmap(z=tn_sim, x=indices, y=indices,
                             zmin=tn_lo, zmax=tn_hi, zmid=0,
                             colorscale=HEAT_COLORSCALE, showscale=False),
                  row=3, col=1)

    fig.add_trace(go.Heatmap(z=act_poisoned, x=indices, y=indices,
                             zmin=ap_lo, zmax=ap_hi, zmid=0,
                             colorscale=HEAT_COLORSCALE, showscale=False),
                  row=4, col=1)

    for x in boundary_x:
        for row in (1, 2, 3, 4):
            fig.add_vline(x=x, row=row, col=1, **BOUNDARY_LINE_KW)

    for first, last, name in phase_idx:
        if name != "poisoned":
            continue
        x_paper = ((first + last) / 2 + 0.5) / n
        fig.add_annotation(x=x_paper, y=0.96, xref="paper", yref="paper",
                           xanchor="center", yanchor="middle",
                           text="<b>backdoor on</b>", showarrow=False,
                           font=dict(size=14, color=LABEL))

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
    fig.update_yaxes(range=[-0.04, 1.07], domain=[0.795, 0.935], row=1, col=1)
    fig.update_yaxes(domain=[0.625, 0.765], row=2, col=1)
    fig.update_yaxes(range=[-0.5, n - 0.5], domain=[0.310, 0.595], row=3, col=1)
    fig.update_yaxes(range=[-0.5, n - 0.5], domain=[0.015, 0.300], row=4, col=1)

    def _last(metric):
        return float(history.filter(pl.col("metric") == metric)
                           .sort("step")["value"].to_list()[-1])

    last_x = n - 1
    for text, y, color, row in (
        (f"{_last('train_acc'):.2f}",           _last("train_acc"),           TRAIN,  1),
        (f"{_last('test_acc'):.2f}",            _last("test_acc"),            TEST,   1),
        (f"{_last('attack_success_rate'):.2f}", _last("attack_success_rate"), DANGER, 1),
        (f"{_last('train_loss'):.2f}",          _last("train_loss"),          TRAIN,  2),
        (f"{_last('test_loss'):.2f}",           _last("test_loss"),           TEST,   2),
        (f"{_last('poison_loss'):.2f}",         _last("poison_loss"),         DANGER, 2),
    ):
        fig.add_annotation(text=f"<b>{text}</b>", x=last_x, y=y, row=row, col=1,
                           xanchor="left", yanchor="middle", xshift=5,
                           showarrow=False, font=dict(size=12, color=color))

    for label, mid_y in (("Accuracy / ASR",        0.865),
                         ("Loss",                  0.695),
                         ("Weight (TN)",           0.453),
                         ("Functional (poison)",   0.158)):
        fig.add_annotation(text=f"<b>{label}</b>", xref="paper", yref="paper",
                           x=-0.038, y=mid_y, xanchor="center", yanchor="middle",
                           textangle=-90, showarrow=False,
                           font=dict(size=16, color=LABEL))

    cfg = meta["config"]
    apply_style(fig,
                title=f"SVHN backdoor — bilinear (d_hidden={cfg['d_hidden']}, "
                      f"5×7 diamond → 9, poison_rate={cfg['poison_rate']})",
                width=900, height=900)
    fig.update_layout(
        margin=dict(l=85, r=70, t=38, b=10),
        title=dict(y=0.985, font=dict(size=18, color=LABEL)),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        legend=dict(orientation="h", yanchor="middle", y=0.965,
                    xanchor="center", x=0.5,
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(size=13, color=LABEL)),
    )
    save_figure(fig, "svhn_backdoor")
