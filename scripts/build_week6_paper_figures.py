from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.plot_style import COLORS, MODEL_COLORS, MODEL_LINESTYLES, apply_plot_style, save_figure


SERIES = ROOT / "data" / "processed" / "series_5min.parquet"
TABLES = ROOT / "outputs" / "tables"
SHARED = TABLES / "w5_visualization"
OUTPUT = ROOT / "outputs" / "figures" / "week6_revision"
MODEL_ORDER = ["Persistence", "Seasonal Naive", "XGBoost", "LSTM"]


def token_formatter(value: float, _: int) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if absolute >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors=COLORS["ink"])
    axis.xaxis.label.set_color(COLORS["ink"])
    axis.yaxis.label.set_color(COLORS["ink"])


def plot_full_series(series: pd.DataFrame, threshold: float) -> Path:
    relative_days = (series.index - series.index.min()).total_seconds() / 86_400
    trailing = series["token_load"].rolling(288, min_periods=1).mean()
    fig, axis = plt.subplots(figsize=(12.2, 4.8))
    axis.plot(relative_days, series["token_load"], color="#5F8DB8", linewidth=0.55, alpha=0.48, label="5-minute load")
    axis.plot(relative_days, trailing, color="#17324D", linewidth=1.8, label="24-hour trailing mean")
    axis.axhline(threshold, color="#7A3E2B", linewidth=1.5, linestyle=(0, (2, 2)), label="Train P95")
    axis.set_xlabel("Relative day from trace start")
    axis.set_ylabel("Token load (Token/5-min)")
    axis.yaxis.set_major_formatter(mticker.FuncFormatter(token_formatter))
    axis.legend(frameon=False, ncol=3, loc="upper right")
    style_axis(axis)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.98, bottom=0.15)
    return save_figure(fig, OUTPUT / "fig_week6_01_full_series_trend.png")


def plot_distribution(series: pd.DataFrame) -> Path:
    values = series["token_load"].to_numpy(dtype=float)
    zero_count = int(np.count_nonzero(values == 0))
    positive = values[values > 0]
    shares = np.array([zero_count, len(positive)], dtype=float) / len(values)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), gridspec_kw={"width_ratios": [0.72, 2.28]})
    axes[0].bar([0, 1], shares * 100, color=["#9AA7B2", COLORS["blue"]], width=0.68, edgecolor="white")
    axes[0].set_xticks([0, 1], ["Zero", "Positive"])
    axes[0].set_ylabel("Share of 5-minute windows (%)")
    axes[0].set_ylim(0, 90)
    for index, share in enumerate(shares):
        axes[0].text(index, share * 100 + 2, f"{share:.2%}\n(n={int(share * len(values)):,})", ha="center", va="bottom", color=COLORS["ink"], fontsize=9)
    sns.histplot(np.log1p(positive), bins=70, stat="density", color=COLORS["blue"], edgecolor="white", linewidth=0.25, ax=axes[1])
    axes[1].set_xlabel("log(1 + positive Token load per 5-minute window)")
    axes[1].set_ylabel("Density among positive windows")
    axes[1].set_xlim(4, 16)
    axes[1].set_xticks(np.arange(4, 17, 2))
    for axis in axes:
        style_axis(axis)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.97, bottom=0.18, wspace=0.30)
    return save_figure(fig, OUTPUT / "fig_week6_02_token_load_distribution.png")


def plot_half_week() -> Path:
    week = pd.read_csv(SHARED / "w5_representative_week_h15.csv")
    week = week.loc[week["relative_hour"].ge(84)].copy()
    fig, axis = plt.subplots(figsize=(12.2, 4.8))
    actual = week[["relative_hour", "y_true"]].drop_duplicates("relative_hour")
    axis.plot(actual["relative_hour"], actual["y_true"], color=COLORS["ink"], linewidth=1.35, label="Actual", zorder=5)
    for model in MODEL_ORDER:
        part = week.loc[week["model"].eq(model)]
        axis.plot(
            part["relative_hour"],
            part["y_pred"],
            color=MODEL_COLORS[model],
            linestyle=MODEL_LINESTYLES[model],
            linewidth=0.9,
            label=model,
            alpha=0.95,
        )
    axis.set_xlim(84, 168)
    axis.set_xlabel("Hours from representative-week start")
    axis.set_ylabel("Token load (Token/5-min)")
    axis.yaxis.set_major_formatter(mticker.FuncFormatter(token_formatter))
    axis.legend(ncol=5, loc="upper left", frameon=False)
    style_axis(axis)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.98, bottom=0.15)
    return save_figure(fig, OUTPUT / "fig_week6_07_representative_half_week_h15.png")


def plot_confusion_matrices() -> Path:
    table = pd.read_csv(TABLES / "table_05_burst_results.csv").set_index("model").loc[MODEL_ORDER]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.5))
    for axis, model in zip(axes.flat, MODEL_ORDER):
        row = table.loc[model]
        counts = np.array([[row["true_negative"], row["false_positive"]], [row["false_negative"], row["true_positive"]]], dtype=int)
        normalized = counts / counts.sum(axis=1, keepdims=True)
        labels = np.empty_like(counts, dtype=object)
        for row_index in range(2):
            for column_index in range(2):
                labels[row_index, column_index] = f"{counts[row_index, column_index]:,}\n({normalized[row_index, column_index]:.1%})"
        sns.heatmap(
            normalized,
            annot=labels,
            fmt="",
            cmap="Blues",
            vmin=0,
            vmax=1,
            cbar=False,
            square=True,
            linewidths=0.8,
            linecolor="#5B5B5B",
            xticklabels=["Non-burst", "Burst"],
            yticklabels=["Non-burst", "Burst"],
            ax=axis,
        )
        axis.set_xlabel(f"Predicted label - {model}")
        axis.set_ylabel("Actual label")
        axis.tick_params(axis="x", rotation=20)
        axis.tick_params(axis="y", rotation=0)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.99, bottom=0.08, hspace=0.34, wspace=0.28)
    return save_figure(fig, OUTPUT / "fig_week6_09_burst_confusion_matrices_h15.png")


def main() -> None:
    apply_plot_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    series = pd.read_parquet(SERIES)
    threshold = float(pd.read_csv(TABLES / "table_01_burst_threshold.csv")["p95_threshold"].iloc[0])
    paths = [plot_full_series(series, threshold), plot_distribution(series), plot_half_week(), plot_confusion_matrices()]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
