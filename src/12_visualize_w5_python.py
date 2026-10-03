"""Create the complete Week-5 Python figure bundle and shared RStudio inputs.

This is a reporting-only script.  It never fits a model and only reads the
frozen Week-5 CSV outputs.  The exported ``outputs/tables/w5_visualization``
tables are the sole inputs to the companion RStudio script, so both toolchains
plot identical observations, thresholds, metrics, and model labels.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG = ROOT / "outputs" / ".matplotlib"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


TABLES = ROOT / "outputs" / "tables"
SHARED = TABLES / "w5_visualization"
FIGURES = ROOT / "outputs" / "figures" / "w5_python"

MODEL_ORDER = ("Persistence", "Seasonal Naive", "XGBoost", "LSTM")
MODEL_COLORS = {
    "Persistence": "#31688E",
    "Seasonal Naive": "#E69F00",
    "XGBoost": "#D55E00",
    "LSTM": "#7A8F2A",
}
MODEL_LINESTYLES = {
    "Persistence": (0, (4, 2)),
    "Seasonal Naive": (0, (6, 2, 1, 2)),
    "XGBoost": "-",
    "LSTM": (0, (2, 1)),
}
INK = "#232A31"
MUTED = "#5B6470"
GRID = "#D9DEE5"
OPEN = "#EFF3F7"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9.5,
            "axes.titlesize": 12,
            "axes.labelsize": 9.5,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.7,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.9,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def read(name: str) -> pd.DataFrame:
    path = TABLES / name
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen Week-5 input: {path}")
    return pd.read_csv(path)


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def write(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(SHARED / name, index=False, float_format="%.10g")


def style_axis(axis: plt.Axes, *, zero_line: bool = False) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", visible=True)
    axis.grid(axis="x", visible=False)
    axis.set_axisbelow(True)
    if zero_line:
        axis.axhline(0, color=INK, linewidth=0.7, zorder=1)


def token_formatter(value: float, _position: int) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if absolute >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def save(fig: plt.Figure, name: str) -> Path:
    path = FIGURES / name
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def select_representative_week(h15: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    unique = h15[["timestamp", "y_true"]].drop_duplicates("timestamp").sort_values("timestamp")
    week_id = ((unique["timestamp"] - EPOCH) // pd.Timedelta(days=7)).astype(int)
    summary = unique.assign(week_id=week_id).groupby("week_id").agg(
        start=("timestamp", "min"), end=("timestamp", "max"), rows=("timestamp", "size"), total=("y_true", "sum")
    )
    complete = summary.loc[summary["rows"].ge(7 * 24 * 12)].copy()
    if complete.empty:
        start = unique["timestamp"].iloc[0]
        return start, min(start + pd.Timedelta(days=7), unique["timestamp"].iloc[-1] + pd.Timedelta(minutes=5))
    complete["distance"] = (complete["total"] - complete["total"].median()).abs()
    chosen = complete.sort_values(["distance", "start"], kind="mergesort").iloc[0]
    return pd.Timestamp(chosen["start"]), pd.Timestamp(chosen["start"]) + pd.Timedelta(days=7)


def export_shared_data() -> dict[str, object]:
    SHARED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    main = read("table_04_main_results.csv")
    appendix = read("table_04a_appendix_results_h5_h60.csv")
    burst = read("table_05_burst_results.csv")
    burst_appendix = read("table_05a_appendix_burst_results_h5_h60.csv")
    regression = pd.concat([main, appendix], ignore_index=True)
    burst_all = pd.concat([burst, burst_appendix], ignore_index=True)
    metrics = regression.merge(
        burst_all,
        on=["horizon_minutes", "model", "n_observations"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_burst"),
    )
    metrics["model"] = pd.Categorical(metrics["model"], MODEL_ORDER, ordered=True)
    metrics.sort_values(["horizon_minutes", "model"], inplace=True)
    write(metrics, "w5_metrics_all.csv")

    predictions = read("test_predictions_all_models.csv")
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], utc=True)
    predictions["actual_burst"] = as_bool(predictions["actual_burst"])
    predictions["predicted_burst"] = as_bool(predictions["predicted_burst"])
    h15 = predictions.loc[predictions["horizon"].eq(15)].copy()
    threshold = float(burst.loc[burst["horizon_minutes"].eq(15), "train_p95_threshold"].iloc[0])

    week_start, week_end = select_representative_week(h15)
    week = h15.loc[h15["timestamp"].ge(week_start) & h15["timestamp"].lt(week_end)].copy()
    week["relative_hour"] = (week["timestamp"] - week_start).dt.total_seconds() / 3600.0
    write(week, "w5_representative_week_h15.csv")

    actual = h15[["timestamp", "y_true"]].drop_duplicates("timestamp")
    center = pd.Timestamp(actual.loc[actual["y_true"].idxmax(), "timestamp"])
    zoom = h15.loc[h15["timestamp"].between(center - pd.Timedelta(hours=6), center + pd.Timedelta(hours=6))].copy()
    zoom["relative_hour"] = (zoom["timestamp"] - center).dt.total_seconds() / 3600.0
    zoom["train_p95_threshold"] = threshold
    write(zoom, "w5_burst_zoom_h15.csv")

    curve_parts: list[pd.DataFrame] = []
    for model in MODEL_ORDER:
        part = h15.loc[h15["model"].eq(model)]
        precision, recall, thresholds = precision_recall_curve(part["actual_burst"], part["y_pred"])
        scores = np.r_[thresholds, np.nan]
        curve = pd.DataFrame({"model": model, "recall": recall, "precision": precision, "score_threshold": scores})
        if len(curve) > 500:
            take = np.unique(np.linspace(0, len(curve) - 1, 500).round().astype(int))
            curve = curve.iloc[take]
        curve_parts.append(curve)
    pr_curve = pd.concat(curve_parts, ignore_index=True)
    write(pr_curve, "w5_pr_curve_h15.csv")

    segment = read("table_06_segment_errors.csv")
    write(segment, "w5_segment_errors_h15.csv")
    ablation = read("table_06a_xgb_h15_ablation.csv")
    write(ablation, "w5_ablation_h15.csv")
    robustness = read("table_07_robustness_burstgpt3.csv")
    write(robustness, "w5_robustness_all.csv")

    meta = pd.DataFrame(
        [
            {"name": "representative_week_start", "value": week_start.isoformat()},
            {"name": "representative_week_end", "value": week_end.isoformat()},
            {"name": "burst_zoom_center", "value": center.isoformat()},
            {"name": "train_p95_threshold", "value": f"{threshold:.10g}"},
        ]
    )
    write(meta, "w5_visualization_metadata.csv")
    return {
        "metrics": metrics,
        "h15": h15,
        "week": week,
        "zoom": zoom,
        "pr_curve": pr_curve,
        "segment": segment,
        "ablation": ablation,
        "robustness": robustness,
        "threshold": threshold,
        "week_start": week_start,
        "week_end": week_end,
        "center": center,
    }


def plot_h15_regression(data: dict[str, object]) -> Path:
    metrics = data["metrics"]
    main = metrics.loc[metrics["horizon_minutes"].eq(15)].set_index("model").loc[list(MODEL_ORDER)]
    positions = np.arange(len(MODEL_ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4))
    for axis, metric, label in zip(axes, ("mae", "rmse"), ("MAE", "RMSE")):
        values = main[metric].to_numpy(float)
        low = main[f"{metric}_ci_low"].to_numpy(float)
        high = main[f"{metric}_ci_high"].to_numpy(float)
        axis.bar(positions, values, color=[MODEL_COLORS[m] for m in MODEL_ORDER], width=0.68, edgecolor="white")
        axis.errorbar(positions, values, yerr=np.vstack([values - low, high - values]), fmt="none", ecolor=INK, capsize=3, linewidth=0.9)
        axis.set_xticks(positions, MODEL_ORDER, rotation=17, ha="right")
        axis.set_ylabel(f"{label} (Token/5-min)")
        axis.yaxis.set_major_formatter(mticker.FuncFormatter(token_formatter))
        axis.set_ylim(bottom=0)
        style_axis(axis)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.97, bottom=0.22, wspace=0.28)
    return save(fig, "fig_w5_py_01_h15_regression_accuracy.png")


def plot_horizon_mae(data: dict[str, object]) -> Path:
    metrics = data["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))
    horizons = [5, 15, 60]
    positions = np.arange(len(horizons))
    width = 0.19
    for offset, model in enumerate(MODEL_ORDER):
        part = metrics.loc[metrics["model"].eq(model)].set_index("horizon_minutes").loc[horizons]
        x = positions + (offset - 1.5) * width
        for axis, metric in zip(axes, ("mae", "f1")):
            axis.bar(x, part[metric], width=width, color=MODEL_COLORS[model], label=model, hatch=("//" if model == "Persistence" else None), edgecolor="white")
    for axis in axes:
        axis.set_xticks(positions, ["5 min", "15 min", "60 min"])
        axis.set_ylim(bottom=0)
        style_axis(axis)
    axes[0].set_ylabel("MAE (Token/5-min)")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(token_formatter))
    axes[1].set_ylabel("Burst F1 at frozen train P95")
    axes[1].set_ylim(0, 1.02)
    if float(metrics["f1"].max()) <= 1e-12:
        axes[1].text(0.5, 0.52, "All models and horizons: F1 = 0", transform=axes[1].transAxes, ha="center", va="center", color=MUTED, fontsize=10, weight="bold")
    axes[0].legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.02))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.14, wspace=0.28)
    return save(fig, "fig_w5_py_02_mae_by_horizon.png")


def plot_week(data: dict[str, object]) -> Path:
    week = data["week"]
    fig, axis = plt.subplots(figsize=(13.0, 5.5))
    actual = week[["relative_hour", "y_true"]].drop_duplicates("relative_hour")
    axis.plot(actual["relative_hour"], actual["y_true"], color=INK, linewidth=1.25, label="Actual", zorder=4)
    for model in MODEL_ORDER:
        part = week.loc[week["model"].eq(model)]
        axis.plot(part["relative_hour"], part["y_pred"], color=MODEL_COLORS[model], linestyle=MODEL_LINESTYLES[model], linewidth=0.8, label=model, alpha=0.92)
    axis.set_xlim(0, max(1, week["relative_hour"].max()))
    axis.set_xlabel("Hours from representative-week start")
    axis.set_ylabel("Token load (Token/5-min)")
    axis.yaxis.set_major_formatter(mticker.FuncFormatter(token_formatter))
    axis.legend(ncol=5, loc="upper left", bbox_to_anchor=(0, 1.03))
    style_axis(axis)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.97, bottom=0.14)
    return save(fig, "fig_w5_py_03_representative_week_h15.png")


def plot_burst_zoom(data: dict[str, object]) -> Path:
    zoom = data["zoom"]
    fig, axis = plt.subplots(figsize=(12.2, 5.6))
    actual = zoom[["relative_hour", "y_true"]].drop_duplicates("relative_hour")
    axis.plot(actual["relative_hour"], actual["y_true"], color=INK, linewidth=1.5, label="Actual", zorder=5)
    for model in MODEL_ORDER:
        part = zoom.loc[zoom["model"].eq(model)]
        axis.plot(part["relative_hour"], part["y_pred"], color=MODEL_COLORS[model], linestyle=MODEL_LINESTYLES[model], linewidth=1.0, label=model)
    axis.axhline(data["threshold"], color=MUTED, linestyle=(0, (2, 2)), linewidth=1.0, label="Train P95")
    axis.set_xlabel("Hours from highest observed test-load window")
    axis.set_ylabel("Token load (Token/5-min)")
    axis.yaxis.set_major_formatter(mticker.FuncFormatter(token_formatter))
    axis.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.03))
    style_axis(axis)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.97, bottom=0.15)
    return save(fig, "fig_w5_py_04_burst_zoom_h15.png")


def plot_pr(data: dict[str, object]) -> Path:
    curve = data["pr_curve"]
    metrics = data["metrics"]
    ap = metrics.loc[metrics["horizon_minutes"].eq(15)].set_index("model")["pr_auc"]
    actual_rate = float(data["h15"][["timestamp", "actual_burst"]].drop_duplicates("timestamp")["actual_burst"].mean())
    fig, axis = plt.subplots(figsize=(8.5, 6.1))
    for model in MODEL_ORDER:
        part = curve.loc[curve["model"].eq(model)]
        axis.plot(part["recall"], part["precision"], color=MODEL_COLORS[model], linestyle=MODEL_LINESTYLES[model], linewidth=1.4, label=f"{model} (AP={ap.loc[model]:.3f})")
    axis.axhline(actual_rate, color=MUTED, linestyle="dotted", linewidth=1.0, label=f"Prevalence={actual_rate:.3f}")
    axis.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision")
    axis.legend(loc="upper right")
    style_axis(axis)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.97, bottom=0.12)
    return save(fig, "fig_w5_py_05_pr_curves_h15.png")


def plot_confusion(data: dict[str, object]) -> Path:
    metrics = data["metrics"]
    main = metrics.loc[metrics["horizon_minutes"].eq(15)].set_index("model")
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.9), sharex=True, sharey=True)
    for axis, model in zip(axes, MODEL_ORDER):
        row = main.loc[model]
        matrix = np.array([[row["true_negative"], row["false_positive"]], [row["false_negative"], row["true_positive"]]], dtype=float)
        row_sum = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum.ne(0) if hasattr(row_sum, "ne") else row_sum != 0)
        axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues", aspect="equal")
        for i in range(2):
            for j in range(2):
                axis.text(j, i, f"{int(matrix[i, j])}\n({normalized[i, j]:.1%})", ha="center", va="center", color=("white" if normalized[i, j] > 0.55 else INK), fontsize=9)
        axis.set_xticks([0, 1], ["Non-burst", "Burst"], rotation=25, ha="right")
        axis.set_yticks([0, 1], ["Non-burst", "Burst"])
        axis.grid(False)
    axes[0].set_ylabel("Actual class")
    for axis, model in zip(axes, MODEL_ORDER, strict=True):
        axis.set_xlabel(f"Predicted class - {model}")
    fig.subplots_adjust(left=0.065, right=0.99, top=0.96, bottom=0.24, wspace=0.32)
    return save(fig, "fig_w5_py_06_confusion_matrices_h15.png")


def plot_burst_segment(data: dict[str, object]) -> Path:
    segment = data["segment"]
    part = segment.loc[segment["dimension"].eq("actual_burst")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.3))
    positions = np.arange(len(MODEL_ORDER))
    width = 0.36
    for offset, level in enumerate(("Non-burst", "Burst")):
        values = part.loc[part["segment"].eq(level)].set_index("model").reindex(MODEL_ORDER)
        x = positions + (offset - 0.5) * width
        axes[0].bar(x, values["mae"], width=width, label=level, color=("#9CC2D8" if level == "Non-burst" else "#C44E52"), edgecolor="white")
        axes[1].bar(x, values["mean_bias_y_pred_minus_y_true"], width=width, label=level, color=("#9CC2D8" if level == "Non-burst" else "#C44E52"), edgecolor="white")
    for axis in axes:
        axis.set_xticks(positions, MODEL_ORDER, rotation=17, ha="right")
        axis.yaxis.set_major_formatter(mticker.FuncFormatter(token_formatter))
        style_axis(axis, zero_line=True)
    axes[0].set_ylabel("MAE (Token/5-min)")
    axes[1].set_ylabel("Mean bias: y_pred - y_true (Token/5-min)")
    axes[0].legend(loc="upper left")
    fig.subplots_adjust(left=0.08, right=0.985, top=0.97, bottom=0.22, wspace=0.30)
    return save(fig, "fig_w5_py_07_burst_segment_mae_bias_h15.png")


def plot_other_segments(data: dict[str, object]) -> Path:
    segment = data["segment"]
    dimensions = ["relative_day_phase", "relative_week_phase", "gpt4_share", "api_share"]
    labels = ["Relative day phase", "Relative week phase", "GPT-4 share", "API share"]
    fig, axes = plt.subplots(4, 2, figsize=(13.2, 13.0))
    for row, (dimension, label) in enumerate(zip(dimensions, labels)):
        part = segment.loc[segment["dimension"].eq(dimension)].copy()
        levels = part["segment"].drop_duplicates().tolist()
        x = np.arange(len(levels))
        for model in MODEL_ORDER:
            values = part.loc[part["model"].eq(model)].set_index("segment").reindex(levels)
            axes[row, 0].plot(x, values["mae"], marker="o", markersize=3.5, linewidth=1.0, color=MODEL_COLORS[model], linestyle=MODEL_LINESTYLES[model], label=model)
            axes[row, 1].plot(x, values["mean_bias_y_pred_minus_y_true"], marker="o", markersize=3.5, linewidth=1.0, color=MODEL_COLORS[model], linestyle=MODEL_LINESTYLES[model], label=model)
        for col in range(2):
            axes[row, col].set_xticks(x, levels, rotation=(25 if len(levels) > 4 else 0), ha=("right" if len(levels) > 4 else "center"))
            axes[row, col].yaxis.set_major_formatter(mticker.FuncFormatter(token_formatter))
            style_axis(axes[row, col], zero_line=(col == 1))
        axes[row, 0].set_ylabel(f"{label}\nMAE")
        axes[row, 1].set_ylabel("Mean bias")
    axes[0, 0].legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.08))
    fig.subplots_adjust(left=0.10, right=0.99, top=0.97, bottom=0.07, hspace=0.72, wspace=0.28)
    return save(fig, "fig_w5_py_08_segment_profiles_h15.png")


def plot_ablation(data: dict[str, object]) -> Path:
    frame = data["ablation"].copy()
    labels = {"full_features": "Full features", "without_service_structure": "No service structure", "lags_plus_relative_calendar": "Lags + relative calendar"}
    order = list(labels)
    positions = np.arange(len(order))
    width = 0.36
    fig, axis = plt.subplots(figsize=(10.6, 5.6))
    for offset, split in enumerate(("valid", "test")):
        values = frame.loc[frame["split"].eq(split)].set_index("feature_set").reindex(order)
        x = positions + (offset - 0.5) * width
        bars = axis.bar(x, values["mae"], width=width, label=split.title(), color=("#8FBBD5" if split == "valid" else "#D97706"), edgecolor="white", hatch=(None if split == "valid" else "//"))
        for bar, value in zip(bars, values["mae"]):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value / 1000:.1f}k", ha="center", va="bottom", fontsize=8)
    axis.set_xticks(positions, [labels[key] for key in order])
    axis.set_ylabel("MAE (Token/5-min)")
    axis.yaxis.set_major_formatter(mticker.FuncFormatter(token_formatter))
    axis.set_ylim(bottom=0)
    axis.legend(loc="upper left")
    style_axis(axis)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.97, bottom=0.15)
    return save(fig, "fig_w5_py_09_xgb_ablation_h15.png")


def plot_robustness(data: dict[str, object]) -> Path:
    frame = data["robustness"].copy()
    horizons = [5, 15, 60]
    positions = np.arange(len(horizons))
    width = 0.19
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
    for offset, model in enumerate(MODEL_ORDER):
        values = frame.loc[frame["model"].eq(model)].set_index("horizon_minutes").reindex(horizons)
        x = positions + (offset - 1.5) * width
        axes[0].bar(x, values["mae"], width=width, color=MODEL_COLORS[model], edgecolor="white", label=model)
        axes[1].bar(x, values["f1"], width=width, color=MODEL_COLORS[model], edgecolor="white", label=model)
    for axis in axes:
        axis.set_xticks(positions, ["5 min", "15 min", "60 min"])
        axis.set_ylim(bottom=0)
        style_axis(axis)
    axes[0].set_ylabel("MAE (Token/5-min)")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(token_formatter))
    axes[1].set_ylabel("Burst F1 at train-1/2 P95")
    axes[1].set_ylim(0, 1.02)
    axes[0].legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.02))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.14, wspace=0.28)
    return save(fig, "fig_w5_py_10_burstgpt3_robustness.png")


def build_overview(paths: list[Path]) -> Path:
    fig, axes = plt.subplots(5, 2, figsize=(14, 22))
    for axis, path in zip(axes.ravel(), paths):
        axis.imshow(plt.imread(path))
        axis.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.005, hspace=0.04, wspace=0.02)
    return save(fig, "w5_python_overview.png")


def main() -> None:
    configure_style()
    inputs = [
        "table_04_main_results.csv", "table_04a_appendix_results_h5_h60.csv",
        "table_05_burst_results.csv", "table_05a_appendix_burst_results_h5_h60.csv",
        "test_predictions_all_models.csv", "table_06_segment_errors.csv",
        "table_06a_xgb_h15_ablation.csv", "table_07_robustness_burstgpt3.csv",
    ]
    missing = [name for name in inputs if not (TABLES / name).exists()]
    if missing:
        raise SystemExit("Frozen Week-5 outputs are incomplete: " + ", ".join(missing))
    data = export_shared_data()
    paths = [
        plot_h15_regression(data), plot_horizon_mae(data), plot_week(data),
        plot_burst_zoom(data), plot_pr(data), plot_confusion(data),
        plot_burst_segment(data), plot_other_segments(data),
        plot_ablation(data), plot_robustness(data),
    ]
    overview = build_overview(paths)
    manifest = pd.DataFrame(
        {
            "figure_number": [f"W5-PY-{index:02d}" for index in range(1, 11)],
            "python_file": [path.name for path in paths],
            "exists": [path.exists() for path in paths],
            "bytes": [path.stat().st_size for path in paths],
        }
    )
    write(manifest, "w5_python_figure_manifest.csv")
    print(f"Saved {len(paths)} Python Week-5 figures to {FIGURES}")
    print(f"Saved overview: {overview}")


if __name__ == "__main__":
    main()
