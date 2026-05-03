"""Grokking summary on a checkpoint-INDEX axis (not log-step).

Each checkpoint occupies one unit on the x axis, so the heatmap cells are
uniform 1×1 squares. The bottom axis labels show step values at hand-picked
sparse indices instead of plotly's automatic log ticks.

Five DP-optimal phases (`start → memorize → [stuck plateau] → grok →
consolidate`); the unnamed middle segment gets a subtle slate wash on the
line plots (no label) — visual marker for "stuck" without competing with
the rest of the figure. Train traces in deep blue, validation in warm
amber, both solid; the dotted-vs-solid distinction was redundant once the
two curves got distinct colors. Heatmap on a diverging palette
(red ↔ paper-bg ↔ navy) so cosine sign is visible without contaminating
the diagonal at value 1.
"""
import json
import math

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from src.figures.grokking_similarity.prepare import CACHE
from src.figures.style import apply_style, save_figure

# Modern palette — indigo + rose, off-white centered, dark slate accents.
BG          = "#FAFAF7"   # warm-white
TRAIN       = "#4f46e5"   # indigo-600
VAL         = "#e11d48"   # rose-600
LABEL       = "#0f172a"   # slate-900 (annotation text)
BOUNDARY    = "#0f172a"   # slate-900 vlines

# Subtle warm-gray wash on memorize and grok line panels only. The
# differentiation between phases comes from the dark vlines, not from
# colored fills.
PHASE_WASH = {
    "memorize": "#EDEAE2",
    "grok":     "#EDEAE2",
}

# Diverging colorscale anchored to the paper bg. Heatmap goes deeper
# than the line traces (indigo-900 / rose-800 vs indigo-600 / rose-600)
# — the relationship is "darker cousin in the same hue family", not
# pixel-match. Reads richer at full saturation.
HEAT_COLORSCALE = [
    [0.0,  "#b2182b"],   # softened ColorBrewer RdBu, easy on eyes
    [0.25, "#f4a582"],
    [0.5,  BG],
    [0.75, "#92c5de"],
    [1.0,  "#2166ac"],
]
FREQ_COLORSCALE = [[0.0, BG], [1.0, "#2166ac"]]   # matches heatmap max
BOUNDARY_LINE_KW = dict(line_color=BOUNDARY, line_width=2.5)

TICK_TARGETS = (1, 10, 100, 1_000, 10_000, 100_000)
TICK_LABELS  = ("1", "10", "100", "1k", "10k", "100k")


def main():
    meta = json.loads((CACHE / "metadata.json").read_text(encoding="utf-8"))
    steps = [s for s in meta["steps"] if s > 0]
    n = len(steps)
    step_to_idx = {s: i for i, s in enumerate(steps)}
    phases = meta["phases"]

    history = pl.read_ipc(CACHE / "history.feather").filter(pl.col("step") > 0)
    matrix = (pl.read_ipc(CACHE / "similarity.feather")
                .filter((pl.col("metric") == "tn_similarity")
                        & (pl.col("step_i") > 0) & (pl.col("step_j") > 0))
                .sort(["step_i", "step_j"])["value"]
                .to_numpy().reshape(n, n).tolist())
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
    boundary_x = [b[0] for b in bounds[1:]]

    indices = list(range(n))
    indices_freq = list(range(n_freqs))

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.018)

    # Subtle washes on the line panels: memorize gets a soft navy tint
    # (matches the train color), grok gets a soft amber tint (matches
    # validation), and the unnamed plateau gets a neutral warm gray.
    # Start and consolidate stay untinted.
    for (x0, x1), (_, _, name) in zip(bounds, phase_idx):
        if name not in PHASE_WASH:
            continue
        for ax in ("", "2"):
            fig.add_shape(type="rect", xref=f"x{ax}", yref=f"y{ax} domain",
                          x0=x0, x1=x1, y0=0, y1=1,
                          fillcolor=PHASE_WASH[name], line_width=0, layer="below")

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
                             zmin=-1, zmax=1, zmid=0,
                             colorscale=HEAT_COLORSCALE, showscale=False),
                  row=3, col=1)

    fig.add_trace(go.Heatmap(z=freq_z, x=indices, y=indices_freq,
                             colorscale=FREQ_COLORSCALE, showscale=False),
                  row=4, col=1)

    for x in boundary_x:
        for row in (1, 2, 3, 4):
            fig.add_vline(x=x, row=row, col=1, **BOUNDARY_LINE_KW)

    for first, last, name in phase_idx:
        if name not in ("memorize", "grok"):
            continue
        x_paper = ((first + last) / 2 + 0.5) / n
        fig.add_annotation(x=x_paper, y=0.922, xref="paper", yref="paper",
                           xanchor="center", yanchor="middle",
                           text=f"<b>{name}</b>", showarrow=False,
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
    fig.update_yaxes(range=[-0.04, 1.07], domain=[0.755, 0.905], row=1, col=1)
    fig.update_yaxes(type="log", range=[-6.7, 1.3], domain=[0.585, 0.735], row=2, col=1)
    fig.update_yaxes(range=[-0.5, n - 0.5], domain=[0.235, 0.565], row=3, col=1)
    fig.update_yaxes(domain=[0.015, 0.215], row=4, col=1)

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

    for label, mid_y in (("Accuracy",     0.83),
                         ("Loss (log)",   0.66),
                         ("Similarity",   0.40),
                         ("Frequency",    0.115)):
        fig.add_annotation(text=f"<b>{label}</b>", xref="paper", yref="paper",
                           x=-0.038, y=mid_y, xanchor="center", yanchor="middle",
                           textangle=-90, showarrow=False,
                           font=dict(size=16, color=LABEL))

    apply_style(fig, title=f"Grokking on modular addition (P={meta['config']['P']})",
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
    save_figure(fig, "grokking_similarity")

    # ── companion: similarity-to-final progress curves ───────────────────────
    # Same dataset, different metrics. TN is computed from weights and is
    # gauge-invariant by construction; act_similarity is per-sample logit
    # cosine on Logan's full P² grid; JS is divergence between per-checkpoint
    # frequency-output distributions. The first two are smooth monotonic
    # progress markers. JS is non-monotonic — it PEAKS during grok (step
    # 6922, val_acc=30%) — exactly when generalization is appearing.
    # The point: empirical similarity has free choices (which metric,
    # which inputs, which aggregation); different choices give qualitatively
    # different progress curves. TN is one canonical answer.
    final_step = meta["steps"][-1]
    sim_long = (pl.read_ipc(CACHE / "similarity.feather")
                  .filter((pl.col("step_j") == final_step) & (pl.col("step_i") > 0))
                  .sort("step_i"))

    def _series(metric, df=sim_long, value_col="value"):
        s = df.filter(pl.col("metric") == metric).sort("step_i" if "step_i" in df.columns else "step")
        x = [step_to_idx[step] for step in s["step_i" if "step_i" in df.columns else "step"].to_list()]
        return x, s[value_col].to_list()

    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                         row_heights=[0.5, 0.5])

    PROGRESS_TRACES = (
        ("train_acc",      "Train acc",           "#94a3b8", history,  "value"),
        ("val_acc",        "Val acc",             "#475569", history,  "value"),
        ("act_similarity", "Empirical (logits)",  "#a5b4fc", sim_long, "value"),
        ("tn_similarity",  "Tensor (TN)",         "#312e81", sim_long, "value"),
    )
    end_top = []
    for metric, name, color, df, vc in PROGRESS_TRACES:
        x, y = _series(metric, df, vc)
        fig3.add_trace(go.Scatter(x=x, y=y, mode="lines",
                                  name=name, line=dict(color=color, width=2.4)),
                       row=1, col=1)
        end_top.append((name, color, x[-1], y[-1]))

    x_js, y_js = _series("js_divergence")
    fig3.add_trace(go.Scatter(x=x_js, y=y_js, mode="lines",
                              name="JS divergence", line=dict(color=VAL, width=2.4)),
                   row=2, col=1)
    js_peak_idx = max(range(len(y_js)), key=lambda i: y_js[i])
    js_peak_step = steps[x_js[js_peak_idx]]

    for x in boundary_x:
        for row in (1, 2):
            fig3.add_vline(x=x, row=row, col=1, **BOUNDARY_LINE_KW)

    fig3.add_annotation(x=x_js[js_peak_idx], y=y_js[js_peak_idx], row=2, col=1,
                        text=f"<b>peak at step {js_peak_step:,}</b><br>(val_acc≈30%, mid-grok)",
                        xanchor="left", yanchor="bottom", xshift=8, yshift=4,
                        showarrow=True, arrowhead=2, arrowcolor=VAL, arrowsize=1, arrowwidth=1.4,
                        ax=40, ay=-30,
                        font=dict(size=11, color=LABEL))

    yshifts = (-32, -10, 10, 32)  # stagger labels (4 lines all end near y≈1)
    for (name, color, x_end, y_end), yshift in zip(end_top, yshifts):
        fig3.add_annotation(text=f"<b>{name}</b>", x=x_end, y=y_end, row=1, col=1,
                            xanchor="left", yanchor="middle", xshift=6, yshift=yshift,
                            showarrow=False, font=dict(size=11, color=color))

    fig3.update_xaxes(showline=False, zeroline=False, mirror=False,
                      range=[-0.5, n - 0.5], showgrid=False, ticks="")
    fig3.update_xaxes(tickmode="array", tickvals=tick_indices, ticktext=list(TICK_LABELS),
                      title=dict(text="<b>Training step</b>", font=dict(size=14)),
                      row=2, col=1)
    fig3.update_yaxes(showline=False, zeroline=False, mirror=False, showgrid=False,
                      ticks="", showticklabels=False)
    fig3.update_yaxes(range=[-0.05, 1.08], row=1, col=1)
    fig3.update_yaxes(range=[0, max(y_js) * 1.15], row=2, col=1)

    for label, mid_y in (("Cosine / accuracy", 0.78),
                         ("JS divergence",     0.32)):
        fig3.add_annotation(text=f"<b>{label}</b>", xref="paper", yref="paper",
                            x=-0.025, y=mid_y, xanchor="center", yanchor="middle",
                            textangle=-90, showarrow=False,
                            font=dict(size=14, color=LABEL))

    apply_style(fig3,
                title="Different similarity metrics, different progress curves (P=113 grokking)",
                width=900, height=560)
    fig3.update_layout(margin=dict(l=70, r=120, t=60, b=50),
                       title=dict(y=0.965, font=dict(size=17, color=LABEL)),
                       paper_bgcolor=BG, plot_bgcolor=BG, showlegend=False)
    save_figure(fig3, "grokking_similarity_progress")

    # ── companion: apples-to-apples spectrum on grokking ────────────────────
    # Same four panels as language_similarity_metrics but on Logan's grokking
    # task with our retrained bilinear model (purely polynomial → Wick is
    # exact). Expected (and observed): all four panels agree closely (Pearson
    # >0.99 with TN). This is the methodological control showing our pipeline
    # is correct; the language-case discrepancies are real Wick approximation
    # error from non-polynomial transformer pieces, not pipeline bugs.
    METRICS_DIR = CACHE.parent / "grokking_similarity_metrics"
    if (METRICS_DIR / "tensor.feather").exists():
        steps_g = pl.read_ipc(METRICS_DIR / "steps.feather")["step"].to_list()
        n_g = len(steps_g)
        idx_g = list(range(n_g))

        def _gmat(path):
            return (pl.read_ipc(METRICS_DIR / path)
                      .sort(["step_i", "step_j"])["similarity"]
                      .to_numpy().reshape(n_g, n_g).tolist())

        GROK_METRICS = (
            (_gmat("grid_train.feather"),    "Grid (train) · logits"),
            (_gmat("grid_test.feather"),     "Grid (test) · logits"),
            (_gmat("gauss_logits.feather"),  "Gaussian · logits"),
            (_gmat("tensor.feather"),        "Gaussian · Tensor"),
        )

        fig4 = make_subplots(rows=1, cols=4, shared_yaxes=True, horizontal_spacing=0.018)
        for col, (z, _) in enumerate(GROK_METRICS, start=1):
            fig4.add_trace(go.Heatmap(z=z, x=idx_g, y=idx_g,
                                       zmin=-1, zmax=1, zmid=0,
                                       colorscale=HEAT_COLORSCALE, showscale=False),
                           row=1, col=col)
        for col, (_, title) in enumerate(GROK_METRICS, start=1):
            fig4.add_annotation(x=(col - 0.5) / 4, y=1.02, xref="paper", yref="paper",
                                text=f"<b>{title}</b>", showarrow=False,
                                xanchor="center", yanchor="bottom",
                                font=dict(size=14, color=LABEL))
        TICK_GROK = (1, 100, 10_000)
        TICK_GROK_LABELS = ("1", "100", "10k")
        tick_idx_g = [min(range(n_g), key=lambda i: abs(steps_g[i] - t)) for t in TICK_GROK]
        fig4.update_xaxes(showline=False, zeroline=False, mirror=False, showgrid=False,
                          range=[-0.5, n_g - 0.5],
                          tickmode="array", tickvals=tick_idx_g, ticktext=list(TICK_GROK_LABELS),
                          ticks="")
        fig4.update_yaxes(showline=False, zeroline=False, mirror=False, showgrid=False,
                          range=[-0.5, n_g - 0.5], showticklabels=False, ticks="")
        fig4.update_yaxes(showticklabels=True, tickmode="array",
                          tickvals=tick_idx_g, ticktext=list(TICK_GROK_LABELS),
                          row=1, col=1)
        apply_style(fig4, title=None, width=1500, height=440)
        fig4.update_layout(margin=dict(l=46, r=18, t=46, b=18),
                           paper_bgcolor=BG, plot_bgcolor=BG)
        save_figure(fig4, "grokking_similarity_metrics")
