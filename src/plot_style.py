from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any


COLORS = MappingProxyType(
    {
        "blue": "#2F5597",
        "orange": "#D97706",
        "green": "#2E7D32",
        "magenta": "#A23B72",
        "dark_gray": "#5B5B5B",
        "ink": "#222222",
        "grid": "#D9D9D9",
        "background": "#FFFFFF",
    }
)

MODEL_COLORS = MappingProxyType(
    {
        "Persistence": COLORS["dark_gray"],
        "Seasonal Naive": COLORS["green"],
        "XGBoost": COLORS["blue"],
        "LSTM": COLORS["orange"],
        "Actual": COLORS["ink"],
    }
)

MODEL_LINESTYLES = MappingProxyType(
    {
        "Persistence": "--",
        "Seasonal Naive": "-.",
        "XGBoost": "-",
        "LSTM": ":",
        "Actual": "-",
    }
)

FIGURE_SIZES = MappingProxyType(
    {
        "single": (7.2, 4.5),
        "wide": (10.0, 5.2),
        "double": (10.0, 7.2),
    }
)

SAVEFIG_KWARGS = MappingProxyType(
    {
        "dpi": 300,
        "bbox_inches": "tight",
        "facecolor": COLORS["background"],
        "edgecolor": "none",
    }
)

BAR_ALPHA = 0.9
HEATMAP_CMAP = "cividis"

RC_PARAMS = MappingProxyType(
    {
        "figure.facecolor": COLORS["background"],
        "axes.facecolor": COLORS["background"],
        "savefig.facecolor": COLORS["background"],
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.titleweight": "semibold",
        "axes.edgecolor": COLORS["dark_gray"],
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.2,
        "grid.linewidth": 0.7,
        "lines.linewidth": 1.8,
        "lines.markersize": 5.0,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    }
)


def apply_plot_style() -> None:
    import matplotlib as mpl
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update(dict(RC_PARAMS))


def save_figure(figure: Any, path: str | Path, **overrides: Any) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    options = dict(SAVEFIG_KWARGS)
    options.update(overrides)
    figure.savefig(output_path, **options)
    return output_path
