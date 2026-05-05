from src.figures import FIGURE_DIR

for _d in ("pdf", "png"):
    (FIGURE_DIR / _d).mkdir(parents=True, exist_ok=True)

COLORWAY = ("#0F172A", "#0EA5E9", "#14B8A6", "#84CC16", "#F59E0B",
            "#F97316", "#E11D48", "#8B5CF6")


def apply_style(fig, title=None, width=1100, height=620, legend=True):
    fig.update_layout(
        template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        colorway=COLORWAY, width=width, height=height, showlegend=legend,
        font=dict(family="Aptos, Avenir Next, Segoe UI Semibold, Segoe UI, Helvetica Neue, Arial, sans-serif",
                  color="#0F172A", size=15),
        title=dict(text=f"<b>{title}</b>", x=0.02, xanchor="left", y=0.97) if title else None,
        margin=dict(l=48, r=36, t=78, b=48),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0,
                    bgcolor="rgba(255,255,255,0.8)", borderwidth=0,
                    font=dict(size=13, color="#475569"), title=None),
    )


def style_xy_axes(fig, x_title=None, y_title=None, x_grid=False, y_grid=True):
    fig.update_xaxes(title_text=f"<b>{x_title}</b>" if x_title else None,
                     showgrid=x_grid, gridcolor="#E2E8F0",
                     showline=True, linewidth=1, linecolor="#CBD5E1",
                     ticks="outside", tickcolor="#CBD5E1", tickfont=dict(color="#475569"))
    fig.update_yaxes(title_text=f"<b>{y_title}</b>" if y_title else None,
                     showgrid=y_grid, gridcolor="#E2E8F0",
                     ticks="outside", tickcolor="#CBD5E1", tickfont=dict(color="#475569"))


def save_figure(fig, stem):
    fig.write_image(FIGURE_DIR / "pdf" / f"{stem}.pdf")
    fig.write_image(FIGURE_DIR / "png" / f"{stem}.png", scale=2)
