from src.figures import FIGURE_DIR

COLORWAY = ("#0F172A", "#0EA5E9", "#14B8A6", "#84CC16", "#F59E0B",
            "#F97316", "#E11D48", "#8B5CF6")

CURRICULUM_STAGES = ("base", "add_5", "add_6", "add_7", "add_8", "add_9", "remove_9", "readd_9")
CURRICULUM_COLORS = dict(zip(CURRICULUM_STAGES, COLORWAY))

SUBSET_CONFIGS = ("all", "drop_9_8_7_6_5_4_3_2")
SUBSET_COLORS = dict(zip(SUBSET_CONFIGS, COLORWAY))


def apply_style(fig, title=None, width=1100, height=620, legend=True):
    fig.update_layout(template="plotly_white", paper_bgcolor="#FCFCF9", plot_bgcolor="#FFFFFF", colorway=COLORWAY, width=width, height=height, showlegend=legend, font=dict(family="Aptos, Avenir Next, Segoe UI Semibold, Segoe UI, Helvetica Neue, Arial, sans-serif", color="#0F172A", size=15), title=dict(text=f"<b>{title}</b>", x=0.02, xanchor="left", y=0.97) if title else None, margin=dict(l=48, r=36, t=78, b=48), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0, bgcolor="rgba(255,255,255,0.8)", borderwidth=0, font=dict(size=13, color="#475569"), title=None))


def style_xy_axes(fig, x_title=None, y_title=None, x_grid=False, y_grid=True):
    fig.update_xaxes(title_text=f"<b>{x_title}</b>" if x_title else None, showgrid=x_grid, gridcolor="#E2E8F0", showline=True, linewidth=1, linecolor="#CBD5E1", ticks="outside", tickcolor="#CBD5E1", tickfont=dict(color="#475569"))
    fig.update_yaxes(title_text=f"<b>{y_title}</b>" if y_title else None, showgrid=y_grid, gridcolor="#E2E8F0", showline=False, ticks="outside", tickcolor="#CBD5E1", tickfont=dict(color="#475569"))


def save_figure(fig, stem, experimental=False):
    """Render to .html + .png. `experimental=True` routes to figures/experimental/."""
    out = FIGURE_DIR / "experimental" if experimental else FIGURE_DIR
    out.mkdir(parents=True, exist_ok=True)
    fig.write_html(out / f"{stem}.html", include_plotlyjs="cdn", full_html=True)
    fig.write_image(out / f"{stem}.png", scale=2)
