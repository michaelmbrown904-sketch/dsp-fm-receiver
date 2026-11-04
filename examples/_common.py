"""Shared setup for the example scripts: import path, plot style, figure saving."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files, never open windows
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
sys.path.insert(0, str(ROOT))  # lets the examples `import dsp` from the repo root

BLUE, ORANGE, GREEN, GRAY = "#2563eb", "#ea580c", "#16a34a", "#6b7280"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "legend.frameon": False,
})


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")
