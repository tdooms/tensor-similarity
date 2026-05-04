from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_fig(fig, run_dir, name):
    out_png = Path(run_dir) / "figures" / "png"
    out_pdf = Path(run_dir) / "figures" / "pdf"
    out_png.mkdir(parents=True, exist_ok=True)
    out_pdf.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
