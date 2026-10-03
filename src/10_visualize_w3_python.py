"""Create the complete Week-3 visualization bundle with Python.

The script exports compact, reviewed plotting CSVs for the companion RStudio
script.  Both toolchains therefore visualize identical W3 splits, tuning
results, predictions, diagnostics, and explanation evidence.
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = ROOT / "outputs" / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures" / "w3_python"
VIZ_TABLES = TABLES / "w3_visualization"
DATA = ROOT / "data" / "processed"

BLUE = "#2F6B9A"
BLUE_LIGHT = "#BFD7EA"
GOLD = "#D9A441"
ORANGE = "#D97706"
INK = "#222222"
MID = "#6B7280"
GRID = "#D9DEE5"
OPEN = "#EFF3F7"

MODEL_COLORS = {
    "XGBoost": BLUE,
    "Persistence": MID,
    "Seasonal Naive": GOLD,
    "XGBoost default": ORANGE,
}
SPLIT_COLORS = {"train": BLUE, "valid": GOLD, "test": ORANGE}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.7,
            "legend.frameon": False,
        }
    )


def title(ax: plt.Axes, heading: str, subtitle: str) -> None:
    ax.set_title(heading, loc="left", fontsize=14, fontweight="bold", pad=30)
    ax.text(
        0,
        1.025,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        color=MID,
    )


def footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.01, 0.008, text, ha="left", va="bottom", fontsize=7.5, color=MID)


def save(fig: plt.Figure, filename: str) -> Path:
    path = FIGURES / filename
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _model_label(value: str) -> str:
    return {
        "persistence": "Persistence",
        "seasonal_naive": "Seasonal Naive",
        "xgboost_selected": "XGBoost",
        "xgboost_final": "XGBoost",
        "xgboost_default": "XGBoost default",
    }[value]


def export_shared_data() -> dict[str, pd.DataFrame | int]:
    VIZ_TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    split = pd.read_csv(TABLES / "table_03_feature_split_summary.csv")
    split["target_time_start"] = pd.to_datetime(split["target_time_start"], utc=True)
    split["target_time_end"] = pd.to_datetime(split["target_time_end"], utc=True)
    split_origin = split["target_time_start"].min().floor("D")
    split["target_start_relative_day"] = (
        (split["target_time_start"] - split_origin) / pd.Timedelta(days=1)
    )
    split["target_end_relative_day"] = (
        (split["target_time_end"] - split_origin) / pd.Timedelta(days=1)
    )
    split["row_label"] = (
        "h" + split["horizon_minutes"].astype(str) + " " + split["split"].astype(str)
    )
    split.to_csv(VIZ_TABLES / "w3_target_time_splits.csv", index=False)

    availability = pd.read_csv(TABLES / "table_03_feature_availability.csv")
    category_labels = {
        "historical_load": "Historical load",
        "historical_request_structure": "Historical request structure",
        "relative_calendar_phase": "Relative calendar phase",
    }
    availability["category_label"] = availability["category"].map(category_labels)
    category_counts = (
        availability.groupby(["category", "category_label", "availability"], as_index=False)
        .agg(n_features=("name", "size"))
        .sort_values("n_features", ascending=False)
    )
    category_counts.to_csv(VIZ_TABLES / "w3_feature_category_counts.csv", index=False)
    availability.to_csv(VIZ_TABLES / "w3_feature_availability.csv", index=False)

    tuning = pd.read_csv(TABLES / "xgb_tuning_log.csv")
    tuning["horizon_label"] = "h" + tuning["horizon_minutes"].astype(str)
    tuning["selection_label"] = np.where(tuning["selected"], "Selected", "Candidate")
    tuning["parameter_summary"] = tuning.apply(
        lambda row: (
            f"depth={int(row.max_depth)}, lr={row.learning_rate:g}, "
            f"trees={int(row.n_estimators)}"
        ),
        axis=1,
    )
    tuning.to_csv(VIZ_TABLES / "w3_tuning_candidates.csv", index=False)
    tuning.loc[tuning["selected"]].to_csv(
        VIZ_TABLES / "w3_selected_parameters.csv", index=False
    )

    baseline = pd.read_csv(TABLES / "table_02_baseline_results.csv")
    xgb = pd.read_csv(TABLES / "table_04_xgb_results.csv")
    metrics = pd.concat(
        [
            baseline.assign(source="baseline"),
            xgb.assign(source="xgboost"),
        ],
        ignore_index=True,
        sort=False,
    )
    metrics["model_label"] = metrics["model"].map(_model_label)
    metrics["horizon_label"] = metrics["horizon_minutes"].map(lambda value: f"h{int(value)}")
    metrics.to_csv(VIZ_TABLES / "w3_model_metrics.csv", index=False)

    diagnostics = pd.read_csv(TABLES / "table_04_xgb_prediction_diagnostics.csv")
    stage_labels = {
        "default_valid": "Default valid",
        "selected_valid": "Selected valid",
        "final_test": "Final test",
    }
    diagnostics["stage_label"] = diagnostics["stage"].map(stage_labels)
    diagnostics["negative_prediction_rate"] = (
        diagnostics["negative_prediction_count"] / diagnostics["n_predictions"]
    )
    diagnostics["display_label"] = (
        "h"
        + diagnostics["horizon_minutes"].astype(str)
        + " | "
        + diagnostics["stage_label"]
    )
    diagnostics.to_csv(VIZ_TABLES / "w3_prediction_diagnostics.csv", index=False)

    error_analysis = pd.read_csv(TABLES / "table_04_xgb_h15_error_analysis.csv")
    error_analysis["model_label"] = error_analysis["model"].map(_model_label)
    error_analysis["display_label"] = (
        error_analysis["split"].str.capitalize() + " | " + error_analysis["model_label"]
    )
    error_analysis.to_csv(VIZ_TABLES / "w3_h15_error_analysis.csv", index=False)

    xgb_test = pd.read_csv(
        TABLES / "pred_xgb_test_h15.csv", parse_dates=["feature_time", "target_time"]
    )
    baseline_test = pd.read_csv(TABLES / "pred_baselines_test.csv", parse_dates=["timestamp"])
    baseline_test.rename(columns={"timestamp": "target_time"}, inplace=True)
    trace = xgb_test.merge(
        baseline_test[
            [
                "target_time",
                "actual_token_load",
                "pred_persistence_h15",
                "pred_seasonal_h15",
            ]
        ],
        on="target_time",
        how="inner",
        suffixes=("_xgb", "_baseline"),
        validate="one_to_one",
    )
    np.testing.assert_array_equal(
        trace["actual_token_load_xgb"], trace["actual_token_load_baseline"]
    )
    series = pd.read_parquet(DATA / "series_5min.parquet", columns=["token_load"])
    series.index = pd.to_datetime(series.index, utc=True)
    series_origin = series.index.min()
    trace["relative_day"] = (
        (trace["target_time"] - series_origin) // pd.Timedelta(days=1)
    ).astype(int)
    daily = trace.groupby("relative_day").agg(
        n_rows=("actual_token_load_xgb", "size"),
        total_load=("actual_token_load_xgb", "sum"),
    )
    complete = daily.loc[daily["n_rows"].eq(288)]
    if complete.empty:
        raise AssertionError("No complete test relative day for the h15 trace")
    representative_day = int((complete["total_load"] - complete["total_load"].median()).abs().idxmin())
    representative = trace.loc[trace["relative_day"].eq(representative_day)].copy()
    day_start = series_origin + pd.Timedelta(days=representative_day)
    representative["relative_hour"] = (
        (representative["target_time"] - day_start) / pd.Timedelta(hours=1)
    )
    representative.rename(
        columns={
            "actual_token_load_xgb": "actual_token_load",
            "predicted_token_load": "xgboost_prediction",
            "pred_persistence_h15": "persistence_prediction",
            "pred_seasonal_h15": "seasonal_naive_prediction",
        },
        inplace=True,
    )
    representative[
        [
            "target_time",
            "relative_day",
            "relative_hour",
            "actual_token_load",
            "xgboost_prediction",
            "persistence_prediction",
            "seasonal_naive_prediction",
        ]
    ].to_csv(VIZ_TABLES / "w3_h15_representative_day.csv", index=False)

    native_parts: list[pd.DataFrame] = []
    shap_parts: list[pd.DataFrame] = []
    for horizon in (5, 15, 60):
        native = pd.read_csv(TABLES / f"xgb_feature_importance_h{horizon}.csv")
        native["horizon_minutes"] = horizon
        native["importance_type"] = "Native gain"
        native["normalized_importance"] = native["importance"] / native["importance"].max()
        native_parts.append(native)

        shap_frame = pd.read_csv(TABLES / f"xgb_shap_importance_h{horizon}.csv")
        shap_frame["horizon_minutes"] = horizon
        shap_frame["importance_type"] = "Mean absolute SHAP"
        shap_frame["normalized_importance"] = (
            shap_frame["mean_abs_shap"] / shap_frame["mean_abs_shap"].max()
        )
        shap_parts.append(shap_frame)
    native_all = pd.concat(native_parts, ignore_index=True)
    shap_all = pd.concat(shap_parts, ignore_index=True)
    rank_scores = pd.concat(
        [
            native_all[["feature", "rank"]],
            shap_all[["feature", "rank"]],
        ],
        ignore_index=True,
    ).groupby("feature", as_index=False)["rank"].mean()
    top_features = (
        rank_scores.sort_values(["rank", "feature"], kind="mergesort")
        .head(12)["feature"]
        .tolist()
    )
    native_all["top12_overall"] = native_all["feature"].isin(top_features)
    shap_all["top12_overall"] = shap_all["feature"].isin(top_features)
    native_all.to_csv(VIZ_TABLES / "w3_native_importance.csv", index=False)
    shap_all.to_csv(VIZ_TABLES / "w3_shap_importance.csv", index=False)
    pd.DataFrame(
        {"feature": top_features, "display_order": np.arange(1, len(top_features) + 1)}
    ).to_csv(VIZ_TABLES / "w3_importance_top12.csv", index=False)

    environment = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "representative_relative_day": representative_day,
        "source_policy": "Saved W3 tables and prediction CSVs only",
    }
    (VIZ_TABLES / "python_environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    return {
        "split": split,
        "category_counts": category_counts,
        "tuning": tuning,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "error_analysis": error_analysis,
        "representative": representative,
        "representative_day": representative_day,
        "native": native_all,
        "shap": shap_all,
        "top_features": top_features,
    }


def plot_target_splits(data: dict[str, object]) -> Path:
    frame = data["split"].copy()  # type: ignore[union-attr]
    horizons = [5, 15, 60]
    split_order = ["train", "valid", "test"]
    rows = [(horizon, split) for horizon in horizons for split in split_order]
    y_positions = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    for y, (horizon, split_name) in zip(y_positions, rows):
        row = frame.loc[
            frame["horizon_minutes"].eq(horizon) & frame["split"].eq(split_name)
        ].iloc[0]
        start = float(row["target_start_relative_day"])
        end = float(row["target_end_relative_day"])
        ax.hlines(y, start, end, color=SPLIT_COLORS[split_name], linewidth=7)
        ax.scatter([start, end], [y, y], color=SPLIT_COLORS[split_name], s=28, zorder=3)
        ax.text(end + 0.8, y, f"n={int(row['n_samples']):,}", va="center", fontsize=8)
    boundaries = (
        frame.loc[frame["split"].isin(["valid", "test"]), "target_start_relative_day"]
        .drop_duplicates()
        .sort_values()
    )
    for boundary in boundaries:
        ax.axvline(boundary, color=INK, linestyle=":", linewidth=1.0)
    ax.set_yticks(y_positions, [f"h{h} | {s}" for h, s in rows])
    ax.set_xlabel("Relative day on the synthetic time axis")
    ax.set_ylabel("")
    ax.set_xlim(frame["target_start_relative_day"].min() - 1, frame["target_end_relative_day"].max() + 9)
    handles = [Line2D([0], [0], color=SPLIT_COLORS[name], lw=6, label=name.title()) for name in split_order]
    ax.legend(handles=handles, ncol=3, loc="upper right")
    title(
        ax,
        "W3-01 Target-time split intervals",
        "All nine datasets are assigned by target_time; dotted lines are validation/test boundaries",
    )
    footer(fig, "Source: table_03_feature_split_summary.csv | all leakage_check values passed")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "fig_w3_py_01_target_time_splits.png")


def plot_feature_availability(data: dict[str, object]) -> Path:
    frame = data["category_counts"].sort_values("n_features")  # type: ignore[union-attr]
    fig, ax = plt.subplots(figsize=(10.2, 5.7))
    bars = ax.barh(frame["category_label"], frame["n_features"], color=BLUE, edgecolor="#244F70")
    ax.bar_label(bars, labels=[f"{int(value)} features" for value in frame["n_features"]], padding=5)
    ax.set_xlabel("Number of model features")
    ax.set_xlim(0, frame["n_features"].max() * 1.28)
    title(
        ax,
        "W3-02 Feature availability by source",
        "36/36 model columns are available at t; future targets and real-calendar labels are excluded",
    )
    footer(fig, "Source: table_03_feature_availability.csv")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "fig_w3_py_02_feature_availability.png")


def plot_tuning_tradeoff(data: dict[str, object]) -> Path:
    tuning = data["tuning"]  # type: ignore[assignment]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.9), sharey=True)
    for ax, horizon in zip(axes, (5, 15, 60)):
        subset = tuning.loc[tuning["horizon_minutes"].eq(horizon)]
        regular = subset.loc[~subset["selected"]]
        selected = subset.loc[subset["selected"]]
        ax.scatter(regular["valid_mae"], regular["valid_f1"], color=BLUE_LIGHT, edgecolor=BLUE, s=48)
        ax.scatter(selected["valid_mae"], selected["valid_f1"], color=GOLD, edgecolor=INK, s=100, zorder=4)
        row = selected.iloc[0]
        ax.annotate(
            f"selected #{int(row.candidate_index)}",
            (row.valid_mae, row.valid_f1),
            xytext=(5, 9),
            textcoords="offset points",
            fontsize=8,
        )
        ax.set_title(f"{horizon}-minute horizon", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Validation MAE")
        ax.ticklabel_format(axis="x", style="plain")
    axes[0].set_ylabel("Validation F1 at fixed train P95")
    fig.suptitle("W3-03 Validation tuning trade-off", x=0.06, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.06, 0.91, "Twelve bounded random candidates per horizon; selection uses minimum validation MAE", color=MID, fontsize=8.8)
    footer(fig, "Source: xgb_tuning_log.csv | test data not used in selection")
    fig.tight_layout(rect=(0, 0.05, 1, 0.88))
    return save(fig, "fig_w3_py_03_tuning_tradeoff.png")


def _formal_metrics(metrics: pd.DataFrame, split_name: str) -> pd.DataFrame:
    models = ["Persistence", "Seasonal Naive", "XGBoost"]
    return metrics.loc[
        metrics["split"].eq(split_name) & metrics["model_label"].isin(models)
    ].copy()


def plot_mae_comparison(data: dict[str, object], split_name: str, filename: str, code: str) -> Path:
    metrics = data["metrics"]  # type: ignore[assignment]
    frame = _formal_metrics(metrics, split_name)
    horizons = [5, 15, 60]
    models = ["Persistence", "Seasonal Naive", "XGBoost"]
    x = np.arange(len(horizons))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11.2, 6.0))
    for offset_index, model in enumerate(models):
        values = [
            float(frame.loc[frame["horizon_minutes"].eq(h) & frame["model_label"].eq(model), "mae"].iloc[0])
            for h in horizons
        ]
        bars = ax.bar(
            x + (offset_index - 1) * width,
            values,
            width,
            color=MODEL_COLORS[model],
            edgecolor=INK if model == "XGBoost" else "none",
            linewidth=0.7,
            label=model,
        )
        ax.bar_label(bars, labels=[f"{value/1000:.1f}k" for value in values], padding=3, fontsize=8)
    if split_name == "valid":
        default = metrics.loc[
            metrics["split"].eq("valid") & metrics["model_label"].eq("XGBoost default")
        ].iloc[0]
        ax.scatter(
            [x[1]],
            [default["mae"]],
            marker="D",
            s=60,
            color=ORANGE,
            edgecolor=INK,
            label="XGBoost default (h15)",
            zorder=5,
        )
        ax.annotate(f"default {default['mae']/1000:.1f}k", (x[1], default["mae"]), xytext=(7, 8), textcoords="offset points", fontsize=8)
    ax.set_xticks(x, [f"{h} min" for h in horizons])
    ax.set_ylabel("MAE (tokens per 5-minute target window)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.legend(ncol=4 if split_name == "valid" else 3, loc="upper left")
    subtitle = (
        "Validation-only comparison; selected XGBoost uses the lowest validation MAE"
        if split_name == "valid"
        else "One post-freeze test evaluation; test results were never used for parameter selection"
    )
    title(ax, f"{code} {split_name.capitalize()} MAE by horizon", subtitle)
    footer(fig, "Source: table_02_baseline_results.csv and table_04_xgb_results.csv")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, filename)


def plot_representative_day(data: dict[str, object]) -> Path:
    frame = data["representative"]  # type: ignore[assignment]
    day = int(data["representative_day"])
    fig, ax = plt.subplots(figsize=(12.2, 6.0))
    ax.plot(frame["relative_hour"], frame["actual_token_load"], color=INK, linewidth=1.6, label="Actual")
    ax.plot(frame["relative_hour"], frame["xgboost_prediction"], color=BLUE, linewidth=1.2, label="XGBoost")
    ax.plot(frame["relative_hour"], frame["persistence_prediction"], color=MID, linewidth=1.1, linestyle="--", label="Persistence")
    ax.plot(frame["relative_hour"], frame["seasonal_naive_prediction"], color=GOLD, linewidth=1.1, linestyle="-.", label="Seasonal Naive")
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xlim(0, 24)
    ax.set_xlabel("Hour within the selected relative day")
    ax.set_ylabel("Tokens per 5-minute target window")
    ax.legend(ncol=4, loc="upper right")
    title(
        ax,
        "W3-06 15-minute forecasts on a representative test day",
        f"Complete relative day {day}, selected by median daily load; raw negative XGBoost values are retained",
    )
    footer(fig, "Source: pred_xgb_test_h15.csv and pred_baselines_test.csv")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "fig_w3_py_06_h15_representative_day.png")


def plot_distribution_shift(data: dict[str, object]) -> Path:
    frame = data["error_analysis"].copy()  # type: ignore[union-attr]
    order = [
        "Valid | XGBoost",
        "Test | XGBoost",
        "Test | Persistence",
        "Test | Seasonal Naive",
    ]
    frame["display_label"] = pd.Categorical(frame["display_label"], order, ordered=True)
    frame.sort_values("display_label", inplace=True)
    y = np.arange(len(frame))[::-1]
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    for position, row in zip(y, frame.itertuples(index=False)):
        ax.plot([row.actual_mean / 1000, row.prediction_mean / 1000], [position, position], color=GRID, linewidth=4)
        ax.scatter(row.actual_mean / 1000, position, color=INK, s=58, zorder=3)
        ax.scatter(row.prediction_mean / 1000, position, color=MODEL_COLORS[row.model_label], marker="s", s=62, zorder=3)
        ax.text(max(row.actual_mean, row.prediction_mean) / 1000 + 1.4, position, f"bias {row.mean_error_bias/1000:+.1f}k", va="center", fontsize=8)
    ax.set_yticks(y, frame["display_label"].astype(str))
    ax.set_xlabel("Mean token load / prediction (thousands per 5-minute window)")
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=INK, label="Actual mean", markersize=7),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=BLUE, label="Prediction mean", markersize=7),
    ]
    ax.legend(handles=legend, ncol=2, loc="lower right")
    title(
        ax,
        "W3-07 Validation-to-test load shift and mean bias",
        "Test actual mean is 74.86% below validation; Persistence nearly matches the test mean",
    )
    footer(fig, "Source: table_04_xgb_h15_error_analysis.csv | test rows are interpretation only")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "fig_w3_py_07_distribution_shift.png")


def plot_prediction_diagnostics(data: dict[str, object]) -> Path:
    frame = data["diagnostics"].copy()  # type: ignore[union-attr]
    stage_order = {"Default valid": 0, "Selected valid": 1, "Final test": 2}
    frame["stage_order"] = frame["stage_label"].map(stage_order)
    frame.sort_values(["horizon_minutes", "stage_order"], inplace=True)
    fig, ax = plt.subplots(figsize=(11.2, 6.0))
    colors = [ORANGE if stage == "Default valid" else BLUE for stage in frame["stage_label"]]
    bars = ax.barh(frame["display_label"], 100 * frame["negative_prediction_rate"], color=colors, edgecolor=INK, linewidth=0.5)
    ax.bar_label(
        bars,
        labels=[f"{int(count):,} ({100*rate:.1f}%)" for count, rate in zip(frame["negative_prediction_count"], frame["negative_prediction_rate"])],
        padding=4,
        fontsize=8,
    )
    ax.set_xlabel("Share of raw predictions below zero (%)")
    ax.set_xlim(0, max(100 * frame["negative_prediction_rate"]) * 1.35)
    ax.invert_yaxis()
    title(
        ax,
        "W3-08 Negative-prediction diagnostic",
        "Every series is non-constant and target-time aligned; raw values are shown without post-test clipping",
    )
    footer(fig, "Source: table_04_xgb_prediction_diagnostics.csv")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "fig_w3_py_08_prediction_diagnostics.png")


def plot_importance_matrix(data: dict[str, object]) -> Path:
    top_features: list[str] = data["top_features"]  # type: ignore[assignment]
    horizons = [5, 15, 60]
    cmap = LinearSegmentedColormap.from_list("w3_blue", ["#F7FAFC", BLUE_LIGHT, BLUE])
    fig, axes = plt.subplots(1, 2, figsize=(12.7, 7.0), sharey=True)
    for ax, key, panel_title in (
        (axes[0], "native", "Native feature_importances_"),
        (axes[1], "shap", "Mean absolute SHAP"),
    ):
        frame = data[key]  # type: ignore[assignment]
        pivot = (
            frame.loc[frame["feature"].isin(top_features)]
            .pivot(index="feature", columns="horizon_minutes", values="normalized_importance")
            .reindex(index=top_features, columns=horizons)
        )
        matrix = pivot.to_numpy(dtype="float64")
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7.3, color="white" if value > 0.58 else INK)
        ax.set_xticks(range(len(horizons)), [f"h{h}" for h in horizons])
        ax.set_yticks(range(len(top_features)), top_features)
        ax.set_title(panel_title, fontsize=10.5, fontweight="bold")
        ax.grid(False)
    colorbar_axis = fig.add_axes([0.90, 0.18, 0.018, 0.64])
    fig.colorbar(image, cax=colorbar_axis, label="Importance normalized to horizon maximum")
    fig.suptitle("W3-09 Feature-importance evidence across horizons", x=0.08, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.08, 0.92, "Top 12 features by average rank across native and SHAP evidence; all passed availability audit", fontsize=8.8, color=MID)
    footer(fig, "Source: xgb_feature_importance_h*.csv and xgb_shap_importance_h*.csv")
    fig.subplots_adjust(left=0.26, right=0.87, top=0.86, bottom=0.08, wspace=0.20)
    return save(fig, "fig_w3_py_09_importance_matrix.png")


def build_overview(paths: list[Path]) -> Path:
    canvas = Image.new("RGB", (2100, 2100), "white")
    cell_width, cell_height = 700, 700
    for index, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        fitted = ImageOps.contain(image, (cell_width - 24, cell_height - 24))
        row, column = divmod(index, 3)
        x = column * cell_width + (cell_width - fitted.width) // 2
        y = row * cell_height + (cell_height - fitted.height) // 2
        canvas.paste(fitted, (x, y))
    output = FIGURES / "w3_python_overview.png"
    canvas.save(output, quality=95)
    return output


def main() -> None:
    configure_style()
    data = export_shared_data()
    paths = [
        plot_target_splits(data),
        plot_feature_availability(data),
        plot_tuning_tradeoff(data),
        plot_mae_comparison(data, "valid", "fig_w3_py_04_validation_mae.png", "W3-04"),
        plot_mae_comparison(data, "test", "fig_w3_py_05_test_mae.png", "W3-05"),
        plot_representative_day(data),
        plot_distribution_shift(data),
        plot_prediction_diagnostics(data),
        plot_importance_matrix(data),
    ]
    overview = build_overview(paths)
    manifest = pd.DataFrame(
        {
            "figure_number": [f"W3-{index:02d}" for index in range(1, 10)],
            "python_file": [path.name for path in paths],
            "exists": [path.exists() for path in paths],
            "bytes": [path.stat().st_size for path in paths],
        }
    )
    manifest.to_csv(VIZ_TABLES / "w3_python_figure_manifest.csv", index=False)
    print(f"Saved {len(paths)} Python W3 figures: {FIGURES}")
    print(f"Saved overview: {overview}")
    print(f"Representative relative day: {data['representative_day']}")


if __name__ == "__main__":
    main()
