"""Freeze and evaluate all final test forecasts under one common protocol.

This module is reporting-only.  It never fits or selects a model.  It reads the
already-frozen baseline, XGBoost, and LSTM prediction files, verifies target-time
alignment, computes common regression and burst metrics, adds calendar-day block
bootstrap confidence intervals, and creates the Week-5 paper tables and figures.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / "outputs" / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)

from utils import ensure_output_directories, project_path


SEED = 42
EPSILON = 1e-9
HORIZONS = (5, 15, 60)
HORIZON_ORDER = {15: 0, 5: 1, 60: 2}
MODEL_ORDER = ("Persistence", "Seasonal Naive", "XGBoost", "LSTM")
MODEL_RANK = {model: rank for rank, model in enumerate(MODEL_ORDER)}
MAIN_FIGURE_MODELS = ("Seasonal Naive", "XGBoost", "LSTM")
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")

INK = "#232A31"
MUTED = "#5B6470"
GRID = "#D9DEE5"
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


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return the three frozen regression metrics in original Token units."""
    actual = np.asarray(y_true, dtype="float64")
    predicted = np.asarray(y_pred, dtype="float64")
    error = predicted - actual
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "smape_percent": float(
            100.0
            * np.mean(
                2.0 * np.abs(error)
                / (np.abs(actual) + np.abs(predicted) + EPSILON)
            )
        ),
    }


def burst_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float
) -> dict[str, float | int | bool]:
    """Threshold actual and predicted load using the single train-only P95."""
    actual = np.asarray(y_true, dtype="float64") >= threshold
    predicted = np.asarray(y_pred, dtype="float64") >= threshold
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[False, True]).ravel()
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    ap_defined = bool(actual.any())
    pr_auc = (
        float(average_precision_score(actual, np.asarray(y_pred, dtype="float64")))
        if ap_defined
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        # In this project PR-AUC is the non-interpolated average precision.
        "pr_auc": pr_auc,
        "pr_auc_method": "average_precision",
        "average_precision_defined": ap_defined,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "actual_burst_windows": int(actual.sum()),
        "predicted_burst_windows": int(predicted.sum()),
        "actual_burst_rate": float(actual.mean()),
        "predicted_burst_rate": float(predicted.mean()),
    }


def load_threshold() -> float:
    path = project_path("outputs", "tables", "table_01_burst_threshold.csv")
    if not path.exists():
        raise FileNotFoundError(f"Missing fixed training threshold: {path}")
    table = pd.read_csv(path)
    if len(table) != 1 or "p95_threshold" not in table:
        raise ValueError("Burst threshold table must contain exactly one p95_threshold")
    return float(table.loc[0, "p95_threshold"])


def _as_utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="raise")


def _as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    converted = normalized.map(mapping)
    if converted.isna().any():
        bad = sorted(normalized.loc[converted.isna()].dropna().unique().tolist())
        raise ValueError(f"Cannot parse boolean values: {bad}")
    return converted.astype(bool)


def _long_part(
    timestamp: pd.Series,
    horizon: int,
    model: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    actual_burst: pd.Series,
    source_predicted_burst: pd.Series | None,
    threshold: float,
    source_file: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    part = pd.DataFrame(
        {
            "timestamp": _as_utc(timestamp),
            "horizon": int(horizon),
            "model": model,
            "y_true": pd.to_numeric(y_true, errors="raise").astype("float64"),
            "y_pred": pd.to_numeric(y_pred, errors="raise").astype("float64"),
            "actual_burst": _as_bool(actual_burst),
        }
    )
    part["predicted_burst"] = part["y_pred"].ge(threshold)
    if source_predicted_burst is not None:
        np.testing.assert_array_equal(
            part["predicted_burst"].to_numpy(),
            _as_bool(source_predicted_burst).to_numpy(),
            err_msg=f"Frozen predicted_burst does not match threshold in {source_file}",
        )
    np.testing.assert_array_equal(
        part["actual_burst"].to_numpy(),
        part["y_true"].ge(threshold).to_numpy(),
        err_msg=f"Frozen actual_burst does not match threshold in {source_file}",
    )
    if not np.isfinite(part[["y_true", "y_pred"]].to_numpy()).all():
        raise ValueError(f"Non-finite actual or predicted values in {source_file}")
    if part["timestamp"].duplicated().any():
        raise AssertionError(f"Duplicate target timestamps in {source_file} h{horizon}")
    audit = {
        "source_file": source_file,
        "horizon_minutes": horizon,
        "model": model,
        "n_rows": len(part),
        "timestamp_start": part["timestamp"].min().isoformat(),
        "timestamp_end": part["timestamp"].max().isoformat(),
        "unique_timestamps": int(part["timestamp"].nunique()),
        "null_cells": int(part.isna().sum().sum()),
        "negative_predictions": int(part["y_pred"].lt(0).sum()),
        "actual_label_matches_fixed_threshold": True,
        "predicted_label_matches_fixed_threshold": True,
    }
    return part, audit


def build_unified_test_predictions(threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read every frozen test prediction and enforce identical target labels."""
    table_dir = project_path("outputs", "tables")
    baseline_path = table_dir / "pred_baselines_test.csv"
    baseline = pd.read_csv(baseline_path)
    parts: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []

    for horizon in HORIZONS:
        for model, prediction_column in (
            ("Persistence", f"pred_persistence_h{horizon}"),
            ("Seasonal Naive", f"pred_seasonal_h{horizon}"),
        ):
            part, audit = _long_part(
                baseline["timestamp"],
                horizon,
                model,
                baseline["actual_token_load"],
                baseline[prediction_column],
                baseline["actual_burst"],
                None,
                threshold,
                baseline_path.relative_to(PROJECT_ROOT).as_posix(),
            )
            parts.append(part)
            audits.append(audit)

        for model, stem in (("XGBoost", "xgb"), ("LSTM", "lstm")):
            source_path = table_dir / f"pred_{stem}_test_h{horizon}.csv"
            source = pd.read_csv(source_path)
            if not source["horizon_minutes"].eq(horizon).all():
                raise AssertionError(f"Wrong horizon values in {source_path}")
            part, audit = _long_part(
                source["target_time"],
                horizon,
                model,
                source["actual_token_load"],
                source["predicted_token_load"],
                source["actual_burst"],
                source["predicted_burst"],
                threshold,
                source_path.relative_to(PROJECT_ROOT).as_posix(),
            )
            parts.append(part)
            audits.append(audit)

    combined = pd.concat(parts, ignore_index=True)
    combined["_model_rank"] = combined["model"].map(MODEL_RANK)
    combined.sort_values(
        ["timestamp", "horizon", "_model_rank"], kind="mergesort", inplace=True
    )
    combined.drop(columns="_model_rank", inplace=True)

    key = ["timestamp", "horizon"]
    coverage = combined.groupby(key, sort=False).agg(
        rows=("model", "size"),
        models=("model", "nunique"),
        y_true_values=("y_true", "nunique"),
        actual_label_values=("actual_burst", "nunique"),
    )
    if not coverage["rows"].eq(len(MODEL_ORDER)).all():
        raise AssertionError("Not every timestamp+horizon has all four model rows")
    if not coverage["models"].eq(len(MODEL_ORDER)).all():
        raise AssertionError("Duplicate or missing model at timestamp+horizon grain")
    if not coverage["y_true_values"].eq(1).all():
        raise AssertionError("Models disagree on y_true at timestamp+horizon grain")
    if not coverage["actual_label_values"].eq(1).all():
        raise AssertionError("Models disagree on actual_burst at timestamp+horizon grain")
    if combined.duplicated(["timestamp", "horizon", "model"]).any():
        raise AssertionError("Unified prediction key is not unique")

    # All three horizons intentionally share the same target-time rows and labels.
    across_horizons = combined.groupby("timestamp", sort=False).agg(
        y_true_values=("y_true", "nunique"),
        actual_label_values=("actual_burst", "nunique"),
    )
    if not across_horizons.eq(1).all().all():
        raise AssertionError("Target labels differ across horizons at the same target time")

    output_path = table_dir / "test_predictions_all_models.csv"
    combined.to_csv(output_path, index=False, float_format="%.10g")
    audit_table = pd.DataFrame(audits)
    audit_table["all_models_share_target_labels"] = True
    audit_table["expected_models_per_timestamp_horizon"] = len(MODEL_ORDER)
    audit_table.to_csv(table_dir / "table_04_prediction_alignment_audit.csv", index=False)
    return combined, audit_table


def block_bootstrap_intervals(
    frame: pd.DataFrame,
    threshold: float,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    """Resample whole synthetic-UTC calendar days and return percentile CIs."""
    if replicates <= 0:
        return {}
    working = frame.reset_index(drop=True)
    day = working["timestamp"].dt.floor("D")
    blocks = [indices.to_numpy() for _, indices in working.groupby(day).groups.items()]
    if len(blocks) < 2:
        raise ValueError("At least two day blocks are required for block bootstrap")
    rng = np.random.default_rng(seed)
    metric_names = ("mae", "rmse", "smape_percent", "precision", "recall", "f1", "pr_auc")
    samples = {name: np.empty(replicates, dtype="float64") for name in metric_names}
    actual_all = working["y_true"].to_numpy(dtype="float64")
    predicted_all = working["y_pred"].to_numpy(dtype="float64")
    for replicate in range(replicates):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        indices = np.concatenate([blocks[int(position)] for position in chosen])
        actual = actual_all[indices]
        predicted = predicted_all[indices]
        values = {**regression_metrics(actual, predicted), **burst_metrics(actual, predicted, threshold)}
        for name in metric_names:
            samples[name][replicate] = float(values[name])
    intervals: dict[str, float] = {}
    for name, values in samples.items():
        low, high = np.quantile(values, [0.025, 0.975])
        intervals[f"{name}_ci_low"] = float(low)
        intervals[f"{name}_ci_high"] = float(high)
    intervals["bootstrap_blocks"] = len(blocks)
    intervals["bootstrap_replicates"] = replicates
    return intervals


def compute_final_metrics(
    predictions: pd.DataFrame, threshold: float, bootstrap_replicates: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        for model in MODEL_ORDER:
            part = predictions.loc[
                predictions["horizon"].eq(horizon) & predictions["model"].eq(model)
            ].copy()
            if part.empty:
                raise AssertionError(f"Missing predictions for {model} h{horizon}")
            regression = regression_metrics(part["y_true"], part["y_pred"])
            burst = burst_metrics(part["y_true"], part["y_pred"], threshold)
            intervals = block_bootstrap_intervals(
                part,
                threshold,
                bootstrap_replicates,
                seed=SEED + horizon * 100 + MODEL_RANK[model],
            )
            rows.append(
                {
                    "horizon_minutes": horizon,
                    "model": model,
                    "n_observations": len(part),
                    "train_p95_threshold": threshold,
                    **regression,
                    **burst,
                    **intervals,
                    "confidence_level": 0.95,
                    "ci_method": "synthetic-UTC calendar-day block bootstrap",
                    "test_role": "post-freeze reporting only",
                }
            )
    metrics = pd.DataFrame(rows)
    seasonal_mae = (
        metrics.loc[metrics["model"].eq("Seasonal Naive")]
        .set_index("horizon_minutes")["mae"]
        .to_dict()
    )
    metrics["seasonal_naive_mae"] = metrics["horizon_minutes"].map(seasonal_mae)
    metrics["mae_improvement_vs_seasonal_percent"] = 100.0 * (
        metrics["seasonal_naive_mae"] - metrics["mae"]
    ) / metrics["seasonal_naive_mae"]
    metrics["_horizon_rank"] = metrics["horizon_minutes"].map(HORIZON_ORDER)
    metrics["_model_rank"] = metrics["model"].map(MODEL_RANK)
    metrics.sort_values(["_horizon_rank", "_model_rank"], kind="mergesort", inplace=True)
    metrics.drop(columns=["_horizon_rank", "_model_rank"], inplace=True)
    return metrics.reset_index(drop=True)


def _write_three_significant(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.3g")


def save_result_tables(metrics: pd.DataFrame) -> None:
    table_dir = project_path("outputs", "tables")
    regression_columns = [
        "horizon_minutes",
        "model",
        "n_observations",
        "mae",
        "mae_ci_low",
        "mae_ci_high",
        "rmse",
        "rmse_ci_low",
        "rmse_ci_high",
        "smape_percent",
        "smape_percent_ci_low",
        "smape_percent_ci_high",
        "seasonal_naive_mae",
        "mae_improvement_vs_seasonal_percent",
        "confidence_level",
        "ci_method",
        "bootstrap_blocks",
        "bootstrap_replicates",
        "test_role",
    ]
    burst_columns = [
        "horizon_minutes",
        "model",
        "n_observations",
        "train_p95_threshold",
        "precision",
        "precision_ci_low",
        "precision_ci_high",
        "recall",
        "recall_ci_low",
        "recall_ci_high",
        "f1",
        "f1_ci_low",
        "f1_ci_high",
        "pr_auc",
        "pr_auc_ci_low",
        "pr_auc_ci_high",
        "pr_auc_method",
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
        "actual_burst_windows",
        "predicted_burst_windows",
        "actual_burst_rate",
        "predicted_burst_rate",
        "confidence_level",
        "ci_method",
        "bootstrap_blocks",
        "bootstrap_replicates",
        "test_role",
    ]
    main = metrics.loc[metrics["horizon_minutes"].eq(15)]
    appendix = metrics.loc[metrics["horizon_minutes"].isin([5, 60])]
    _write_three_significant(
        main[regression_columns], table_dir / "table_04_main_results.csv"
    )
    _write_three_significant(
        appendix[regression_columns], table_dir / "table_04a_appendix_results_h5_h60.csv"
    )
    _write_three_significant(
        main[burst_columns], table_dir / "table_05_burst_results.csv"
    )
    _write_three_significant(
        appendix[burst_columns], table_dir / "table_05a_appendix_burst_results_h5_h60.csv"
    )


def build_segment_errors(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compute the five pre-specified diagnostic segment comparisons for h15."""
    series_path = project_path("data", "processed", "series_5min.parquet")
    series = pd.read_parquet(series_path)
    series.index = pd.DatetimeIndex(pd.to_datetime(series.index, utc=True), name="timestamp")
    train = series.loc[series["split"].eq("train") & series["request_count"].gt(0)]
    thresholds = {
        "gpt4_share_train_median": float(
            (train["gpt4_request_count"] / train["request_count"]).median()
        ),
        "api_share_train_median": float(
            (train["api_request_count"] / train["request_count"]).median()
        ),
    }

    target = series.loc[
        :, ["request_count", "gpt4_request_count", "api_request_count"]
    ].reset_index()
    working = predictions.loc[predictions["horizon"].eq(15)].merge(
        target, on="timestamp", how="left", validate="many_to_one"
    )
    if working[["request_count", "gpt4_request_count", "api_request_count"]].isna().any().any():
        raise AssertionError("Segment composition join lost target-time rows")
    elapsed_seconds = (working["timestamp"] - EPOCH).dt.total_seconds()
    working["relative_day_hour"] = (elapsed_seconds % 86_400.0) / 3600.0
    working["relative_day_phase"] = pd.cut(
        working["relative_day_hour"],
        bins=[0, 6, 12, 18, 24],
        labels=["00–06h", "06–12h", "12–18h", "18–24h"],
        include_lowest=True,
        right=False,
    ).astype("string")
    week_day = np.floor((elapsed_seconds % 604_800.0) / 86_400.0).astype(int) + 1
    working["relative_week_phase"] = "Relative phase " + week_day.astype(str)
    nonzero = working["request_count"].gt(0)
    working["gpt4_share"] = np.where(
        nonzero, working["gpt4_request_count"] / working["request_count"], np.nan
    )
    working["api_share"] = np.where(
        nonzero, working["api_request_count"] / working["request_count"], np.nan
    )
    working["gpt4_share_level"] = np.where(
        working["gpt4_share"].gt(thresholds["gpt4_share_train_median"]), "High", "Low"
    )
    working["api_share_level"] = np.where(
        working["api_share"].gt(thresholds["api_share_train_median"]), "High", "Low"
    )
    working.loc[~nonzero, ["gpt4_share_level", "api_share_level"]] = pd.NA
    working["actual_burst_level"] = np.where(
        working["actual_burst"], "Burst", "Non-burst"
    )
    working["error"] = working["y_pred"] - working["y_true"]

    specifications = [
        ("actual_burst", "actual_burst_level", "fixed train-P95 label", working.index),
        ("relative_day_phase", "relative_day_phase", "6-hour relative bins", working.index),
        (
            "relative_week_phase",
            "relative_week_phase",
            "seven equal 24-hour relative phases; not real weekdays",
            working.index,
        ),
        (
            "gpt4_share",
            "gpt4_share_level",
            f"train nonzero-window median={thresholds['gpt4_share_train_median']:.10g}",
            working.index[nonzero],
        ),
        (
            "api_share",
            "api_share_level",
            f"train nonzero-window median={thresholds['api_share_train_median']:.10g}",
            working.index[nonzero],
        ),
    ]
    rows: list[dict[str, object]] = []
    for dimension, column, rule, eligible_index in specifications:
        eligible = working.loc[eligible_index]
        for (model, segment), part in eligible.groupby(["model", column], observed=True):
            rows.append(
                {
                    "horizon_minutes": 15,
                    "dimension": dimension,
                    "segment": str(segment),
                    "model": model,
                    "n_observations": len(part),
                    "mae": float(part["error"].abs().mean()),
                    "mean_bias_y_pred_minus_y_true": float(part["error"].mean()),
                    "mean_y_true": float(part["y_true"].mean()),
                    "mean_y_pred": float(part["y_pred"].mean()),
                    "group_definition": rule,
                    "excluded_zero_request_windows": (
                        int((~nonzero).sum() / len(MODEL_ORDER))
                        if dimension in {"gpt4_share", "api_share"}
                        else 0
                    ),
                    "target_time_composition_role": (
                        "diagnostic grouping only; never used as a future predictor"
                        if dimension in {"gpt4_share", "api_share"}
                        else "not applicable"
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result["_model_rank"] = result["model"].map(MODEL_RANK)
    result.sort_values(["dimension", "segment", "_model_rank"], kind="mergesort", inplace=True)
    result.drop(columns="_model_rank", inplace=True)
    _write_three_significant(
        result, project_path("outputs", "tables", "table_06_segment_errors.csv")
    )
    return result.reset_index(drop=True), thresholds


def _style_axis(axis: plt.Axes, *, grid: bool = True) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors=INK, labelsize=8.5)
    if grid:
        axis.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)


def _token_formatter(value: float, _position: int) -> str:
    magnitude = abs(value)
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if magnitude >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def _h15_wide(predictions: pd.DataFrame) -> pd.DataFrame:
    part = predictions.loc[predictions["horizon"].eq(15)].copy()
    forecasts = part.pivot(index="timestamp", columns="model", values="y_pred")
    actual = part.groupby("timestamp", sort=True)["y_true"].first()
    wide = forecasts.join(actual.rename("Actual"), how="inner").sort_index()
    if list(wide.index) != sorted(wide.index):
        raise AssertionError("h15 plotting rows are not chronological")
    return wide


def select_representative_week(wide: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    first = wide.index.min().ceil("D")
    last_start = (wide.index.max() - pd.Timedelta(days=7) + pd.Timedelta(minutes=5)).floor("D")
    candidates: list[tuple[pd.Timestamp, float]] = []
    for start in pd.date_range(first, last_start, freq="D"):
        end = start + pd.Timedelta(days=7)
        part = wide.loc[(wide.index >= start) & (wide.index < end)]
        if len(part) == 7 * 288:
            candidates.append((start, float(part["Actual"].sum())))
    if not candidates:
        raise AssertionError("Test interval contains no complete seven-day window")
    median_total = float(np.median([total for _, total in candidates]))
    chosen_start, _ = min(candidates, key=lambda item: (abs(item[1] - median_total), item[0]))
    return chosen_start, chosen_start + pd.Timedelta(days=7)


def plot_test_week(predictions: pd.DataFrame, output_path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    wide = _h15_wide(predictions)
    start, end = select_representative_week(wide)
    selected = wide.loc[(wide.index >= start) & (wide.index < end)]
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.plot(selected.index, selected["Actual"], color=INK, linewidth=1.5, label="Actual")
    for model in MAIN_FIGURE_MODELS:
        ax.plot(
            selected.index,
            selected[model],
            color=MODEL_COLORS[model],
            linestyle=MODEL_LINESTYLES[model],
            linewidth=1.05,
            label=model,
        )
    _style_axis(ax)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Tokens per 5-minute target window")
    ax.set_xlabel("Synthetic UTC target time (relative trace axis)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_token_formatter))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d", tz=start.tz))
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.97, bottom=0.20)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return start, end


def plot_burst_zoom(
    predictions: pd.DataFrame, threshold: float, output_path: Path
) -> pd.Timestamp:
    wide = _h15_wide(predictions)
    actual_burst = wide["Actual"].ge(threshold)
    center = (
        wide.loc[actual_burst, "Actual"].idxmax()
        if actual_burst.any()
        else wide["Actual"].idxmax()
    )
    selected = wide.loc[
        (wide.index >= center - pd.Timedelta(hours=3))
        & (wide.index <= center + pd.Timedelta(hours=3))
    ]
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.plot(
        selected.index,
        selected["Actual"],
        color=INK,
        linewidth=1.8,
        marker="o",
        markersize=2.8,
        label="Actual",
    )
    for model in MAIN_FIGURE_MODELS:
        ax.plot(
            selected.index,
            selected[model],
            color=MODEL_COLORS[model],
            linestyle=MODEL_LINESTYLES[model],
            linewidth=1.2,
            label=model,
        )
    ax.axhline(
        threshold,
        color="#6B7280",
        linestyle=":",
        linewidth=1.4,
        label=f"Train P95 = {threshold:,.0f}",
    )
    ax.axvline(center, color=INK, linewidth=0.8, alpha=0.45)
    _style_axis(ax)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Tokens per 5-minute target window")
    ax.set_xlabel("Synthetic UTC target time (relative trace axis)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_token_formatter))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M", tz=center.tz))
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.97, bottom=0.22)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return center


def plot_metric_bars(metrics: pd.DataFrame, output_path: Path) -> None:
    main = metrics.loc[metrics["horizon_minutes"].eq(15)].set_index("model").loc[list(MODEL_ORDER)]
    positions = np.arange(len(MODEL_ORDER))
    colors = [MODEL_COLORS[model] for model in MODEL_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8))
    for axis, column, ylabel in (
        (axes[0], "mae", "MAE (Token/5-min)"),
        (axes[1], "f1", "Fixed-P95 burst F1"),
    ):
        values = main[column].to_numpy(dtype="float64")
        bars = axis.bar(positions, values, color=colors, edgecolor=INK, linewidth=0.5)
        axis.set_xticks(positions, MODEL_ORDER, rotation=18, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_ylim(bottom=0)
        _style_axis(axis)
        for bar, value in zip(bars, values, strict=True):
            label = f"{value:,.0f}" if column == "mae" else f"{value:.3f}"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                max(bar.get_height(), axis.get_ylim()[1] * 0.01),
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK,
            )
    if np.allclose(main["f1"], 0.0):
        axes[1].set_ylim(0, 0.05)
        axes[1].text(
            0.5,
            0.75,
            "All four methods: F1 = 0.000",
            transform=axes[1].transAxes,
            ha="center",
            color=MUTED,
            fontsize=10,
        )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.97, bottom=0.20, wspace=0.28)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_pr_curves(predictions: pd.DataFrame, metrics: pd.DataFrame, output_path: Path) -> None:
    main = predictions.loc[predictions["horizon"].eq(15)]
    actual = main.groupby("timestamp", sort=True)["actual_burst"].first().astype(bool)
    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    for model in MODEL_ORDER:
        scores = (
            main.loc[main["model"].eq(model)]
            .set_index("timestamp")
            .reindex(actual.index)["y_pred"]
            .to_numpy(dtype="float64")
        )
        precision, recall, _ = precision_recall_curve(actual.to_numpy(), scores)
        ap = float(
            metrics.loc[
                metrics["horizon_minutes"].eq(15) & metrics["model"].eq(model), "pr_auc"
            ].iloc[0]
        )
        ax.step(
            recall,
            precision,
            where="post",
            color=MODEL_COLORS[model],
            linestyle=MODEL_LINESTYLES[model],
            linewidth=1.5,
            label=f"{model} (AP={ap:.3f})",
        )
    prevalence = float(actual.mean())
    ax.axhline(
        prevalence,
        color="#6B7280",
        linestyle=":",
        linewidth=1.2,
        label=f"No-skill prevalence ({prevalence:.4f})",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    _style_axis(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.97, bottom=0.12)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_confusion_matrices(predictions: pd.DataFrame, output_path: Path) -> None:
    selected_models = ("Seasonal Naive", "XGBoost", "LSTM")
    main = predictions.loc[predictions["horizon"].eq(15)]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.8), sharex=True, sharey=True)
    image = None
    for axis, model in zip(axes, selected_models, strict=True):
        part = main.loc[main["model"].eq(model)]
        matrix = confusion_matrix(
            part["actual_burst"].astype(bool),
            part["predicted_burst"].astype(bool),
            labels=[False, True],
        )
        row_sum = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(
            matrix,
            row_sum,
            out=np.zeros_like(matrix, dtype="float64"),
            where=row_sum != 0,
        )
        image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for row in range(2):
            for column in range(2):
                color = "white" if normalized[row, column] > 0.55 else INK
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:,}\n({normalized[row, column]:.1%})",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=10,
                )
        axis.set_xticks([0, 1], ["Non-burst", "Burst"])
        axis.set_yticks([0, 1], ["Non-burst", "Burst"])
        axis.set_xlabel(f"Predicted label - {model}")
    axes[0].set_ylabel("Actual label")
    if image is not None:
        colorbar_axis = fig.add_axes([0.925, 0.18, 0.015, 0.56])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Row-normalized share")
    fig.subplots_adjust(left=0.075, right=0.89, top=0.96, bottom=0.18, wspace=0.25)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_segment_burst(segment: pd.DataFrame, output_path: Path) -> None:
    part = segment.loc[segment["dimension"].eq("actual_burst")]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6))
    for axis, label in zip(axes, ("Non-burst", "Burst"), strict=True):
        values = (
            part.loc[part["segment"].eq(label)]
            .set_index("model")
            .reindex(MODEL_ORDER)["mae"]
        )
        bars = axis.bar(
            np.arange(len(MODEL_ORDER)),
            values,
            color=[MODEL_COLORS[model] for model in MODEL_ORDER],
            edgecolor=INK,
            linewidth=0.5,
        )
        axis.set_xticks(np.arange(len(MODEL_ORDER)), MODEL_ORDER, rotation=18, ha="right")
        axis.set_xlabel(f"Actual {label.lower()} windows")
        axis.set_ylabel("MAE (Token/5-min)")
        axis.set_ylim(bottom=0)
        axis.yaxis.set_major_formatter(mticker.FuncFormatter(_token_formatter))
        _style_axis(axis)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:,.0f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.97, bottom=0.20, wspace=0.27)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_segment_phases(segment: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8))
    dimensions = (
        ("relative_day_phase", ["00–06h", "06–12h", "12–18h", "18–24h"], "Relative within-day phase"),
        (
            "relative_week_phase",
            [f"Relative phase {value}" for value in range(1, 8)],
            "Relative within-week phase (not weekdays)",
        ),
    )
    for axis, (dimension, order, title) in zip(axes, dimensions, strict=True):
        part = segment.loc[segment["dimension"].eq(dimension)]
        for model in MODEL_ORDER:
            values = (
                part.loc[part["model"].eq(model)]
                .set_index("segment")
                .reindex(order)["mae"]
            )
            axis.plot(
                np.arange(len(order)),
                values,
                color=MODEL_COLORS[model],
                linestyle=MODEL_LINESTYLES[model],
                marker="o",
                markersize=3.5,
                linewidth=1.35,
                label=model,
            )
        axis.set_xticks(np.arange(len(order)), order, rotation=25, ha="right")
        axis.set_xlabel(title)
        axis.set_ylabel("MAE (Token/5-min)")
        axis.set_ylim(bottom=0)
        axis.yaxis.set_major_formatter(mticker.FuncFormatter(_token_formatter))
        _style_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", frameon=False)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.97, bottom=0.27, wspace=0.28)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_segment_composition(
    segment: pd.DataFrame, thresholds: dict[str, float], output_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8))
    width = 0.36
    positions = np.arange(len(MODEL_ORDER))
    for axis, dimension, title, cutoff_key in (
        (axes[0], "gpt4_share", "GPT-4 share", "gpt4_share_train_median"),
        (axes[1], "api_share", "API share", "api_share_train_median"),
    ):
        part = segment.loc[segment["dimension"].eq(dimension)]
        for offset, level, hatch, alpha in (
            (-width / 2, "Low", "", 0.48),
            (width / 2, "High", "///", 0.95),
        ):
            values = (
                part.loc[part["segment"].eq(level)]
                .set_index("model")
                .reindex(MODEL_ORDER)["mae"]
            )
            bars = axis.bar(
                positions + offset,
                values,
                width,
                color=[MODEL_COLORS[model] for model in MODEL_ORDER],
                edgecolor=INK,
                linewidth=0.6,
                alpha=alpha,
                hatch=hatch,
                label=level,
            )
            if level == "High":
                for bar, value in zip(bars, values, strict=True):
                    axis.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{value / 1000:.1f}k",
                        ha="center",
                        va="bottom",
                        fontsize=6.8,
                    )
        axis.set_xticks(positions, MODEL_ORDER, rotation=18, ha="right")
        axis.set_xlabel(f"{title} group (train median={thresholds[cutoff_key]:.3f})")
        axis.set_ylabel("MAE (Token/5-min)")
        axis.set_ylim(bottom=0)
        axis.yaxis.set_major_formatter(mticker.FuncFormatter(_token_formatter))
        _style_axis(axis)
        axis.legend(frameon=False, title="Target-time group", fontsize=8)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.97, bottom=0.23, wspace=0.28)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_core_figures(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    segments: pd.DataFrame,
    thresholds: dict[str, float],
    burst_threshold: float,
) -> dict[str, object]:
    figure_dir = project_path("outputs", "figures")
    week_start, week_end = plot_test_week(
        predictions, figure_dir / "fig_11_test_week_forecasts_h15.png"
    )
    burst_center = plot_burst_zoom(
        predictions, burst_threshold, figure_dir / "fig_12_burst_window_zoom_h15.png"
    )
    plot_metric_bars(metrics, figure_dir / "fig_13_model_mae_f1_h15.png")
    plot_pr_curves(predictions, metrics, figure_dir / "fig_14_pr_curves_h15.png")
    plot_confusion_matrices(
        predictions, figure_dir / "fig_15_burst_confusion_matrices_h15.png"
    )
    plot_segment_burst(segments, figure_dir / "fig_16_segment_error_burst_h15.png")
    plot_segment_phases(segments, figure_dir / "fig_17_segment_error_phases_h15.png")
    plot_segment_composition(
        segments, thresholds, figure_dir / "fig_18_segment_error_composition_h15.png"
    )
    return {
        "representative_week_start": week_start,
        "representative_week_end": week_end,
        "burst_center": burst_center,
    }


def _sig(value: float) -> str:
    return f"{float(value):.3g}"


def _legacy_update_experiment_log_unused(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    segments: pd.DataFrame,
    thresholds: dict[str, float],
    figure_context: dict[str, object],
    bootstrap_replicates: int,
) -> None:
    """Insert an idempotent Week-5 freeze and reporting record."""
    marker_start = "<!-- WEEK5_FINAL_EVALUATION_START -->"
    marker_end = "<!-- WEEK5_FINAL_EVALUATION_END -->"
    main = metrics.loc[metrics["horizon_minutes"].eq(15)].set_index("model").loc[list(MODEL_ORDER)]
    mae_winner = str(main["mae"].idxmin())
    ap_winner = str(main["pr_auc"].idxmax())
    burst_part = segments.loc[
        segments["dimension"].eq("actual_burst") & segments["segment"].eq("Burst")
    ].set_index("model")
    nonburst_part = segments.loc[
        segments["dimension"].eq("actual_burst") & segments["segment"].eq("Non-burst")
    ].set_index("model")
    burst_best = str(burst_part["mae"].idxmin())
    nonburst_best = str(nonburst_part["mae"].idxmin())
    week_start = pd.Timestamp(figure_context["representative_week_start"])
    week_end = pd.Timestamp(figure_context["representative_week_end"])
    burst_center = pd.Timestamp(figure_context["burst_center"])

    metric_lines = [
        "| Model | MAE | RMSE | sMAPE | Seasonal-Naive MAE improvement | F1 | PR-AUC/AP | TP/FP/FN/TN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, row in main.iterrows():
        metric_lines.append(
            f"| {model} | {_sig(row['mae'])} | {_sig(row['rmse'])} | "
            f"{_sig(row['smape_percent'])}% | {_sig(row['mae_improvement_vs_seasonal_percent'])}% | "
            f"{_sig(row['f1'])} | {_sig(row['pr_auc'])} | "
            f"{int(row['true_positive'])}/{int(row['false_positive'])}/"
            f"{int(row['false_negative'])}/{int(row['true_negative'])} |"
        )

    block = "\n".join(
        [
            marker_start,
            "## Week 5: Final freeze, unified test evaluation, and figure conclusions",
            "",
            "- Freeze date: 2026-07-21; random seed: 42; primary task: 15 minutes ahead; supporting tasks: 5 and 60 minutes ahead.",
            "- Freeze rule: Week 3 XGBoost and Week 4 LSTM configurations were selected only with validation data. The primary test set is now limited to one-time prediction summaries, metrics, figures, and paper reporting. It cannot change parameters, features, thresholds, or models.",
            "- Unified test set: `data/processed/features_h{5,15,60}_test.parquet`, with the same 5,228 target times per horizon. The training P95 is fixed at `315246.2` Token/5min, and labels always use `token_load >= threshold`.",
            "- Frozen baselines: at forecast origin t, Persistence uses the last complete window `y[t-5min]`, while Seasonal Naive uses the window 24 hours before the target; file: `outputs/tables/pred_baselines_test.csv`.",
            "- Frozen XGBoost: configurations `configs/xgb_h5.yaml`, `xgb_h15.yaml`, and `xgb_h60.yaml`; models `models/xgb_h5.joblib`, `xgb_h15.joblib`, and `xgb_h60.joblib`; predictions `pred_xgb_test_h{5,15,60}.csv`.",
            "- Frozen LSTM: configurations `configs/lstm_h5.yaml`, `lstm_h15.yaml`, and `lstm_h60.yaml`; models `models/lstm_h{5,15,60}.keras` with corresponding feature and target scalers; predictions `pred_lstm_test_h{5,15,60}.csv`.",
            "- Alignment audit: the unified long table has " + f"{len(predictions):,}" + " rows. Every `timestamp+horizon` has exactly four methods, identical actual values and burst labels, and no missing or duplicate keys.",
            "- Confidence information: 95% percentile bootstrap using complete synthetic UTC calendar-day blocks, "
            + f"seed 42, and {bootstrap_replicates:,} replicates. This preserves within-day dependence. The test period has only five burst windows, so burst confidence intervals require caution.",
            "",
            "### Primary 15-minute results (three significant digits)",
            "",
            *metric_lines,
            "",
            f"Overall conclusion: {mae_winner} has the lowest 15-minute MAE ({_sig(main.loc[mae_winner, 'mae'])} Token/5min), "
            f"an improvement of {_sig(main.loc[mae_winner, 'mae_improvement_vs_seasonal_percent'])}% over Seasonal Naive. "
            "No method detects any of the five true test bursts at the fixed threshold, so all four methods have F1=0. "
            f"{ap_winner} has the highest PR-AUC/AP ({_sig(main.loc[ap_winner, 'pr_auc'])}), but that ranking does not change the zero-hit threshold result.",
            "",
            "### Predefined grouping rules",
            "",
            "- Relative time-of-day groups are 00-06h, 06-12h, 12-18h, and 18-24h based on relative seconds. Relative week phase uses seven consecutive 24-hour phases. Neither grouping represents real weekdays or weekends.",
            f"- The high/low GPT-4 share threshold is the median among non-empty training windows, {_sig(thresholds['gpt4_share_train_median'])}; the API share threshold is {_sig(thresholds['api_share_train_median'])}. Zero-request windows have no service composition and are excluded from these groups.",
            "- Target-time GPT-4/API composition is used only for retrospective error interpretation. It is not a future prediction feature, and model inputs use only historical composition available at forecast time.",
            "",
            "### Figure notes",
            "",
            f"- Figure 11 `fig_11_test_week_forecasts_h15.png`: the week from {week_start:%Y-%m-%d} to {(week_end - pd.Timedelta(minutes=5)):%Y-%m-%d} has total load closest to the median among complete seven-day windows and shows how three primary forecasts follow ordinary short-term test patterns.",
            f"- Figure 12 `fig_12_burst_window_zoom_h15.png`: the objectively selected maximum test load at {burst_center.isoformat()} shows that none of the three predictions crosses the training P95, explaining TP=0 at the fixed threshold.",
            f"- Figure 13 `fig_13_model_mae_f1_h15.png`: {mae_winner} has the lowest overall MAE, but every method has F1=0, so average accuracy does not establish effective burst detection.",
            f"- Figure 14 `fig_14_pr_curves_h15.png`: {ap_winner} has the highest AP with only five positive cases, but the curve is discrete and uncertain, so it provides ranking evidence only.",
            "- Figure 15 `fig_15_burst_confusion_matrices_h15.png`: Seasonal Naive, XGBoost, and LSTM each have five false negatives for the five true bursts, with no hidden hits at the fixed capacity threshold.",
            f"- Figure 16 `fig_16_segment_error_burst_h15.png`: {nonburst_best} has the lowest non-burst MAE and {burst_best} has the lowest burst MAE, while burst errors are substantially larger.",
            "- Figure 17 `fig_17_segment_error_phases_h15.png`: error changes across relative daily and weekly phases, but these phases describe anonymous trace positions and cannot be named as real dates or weekends.",
            "- Figure 18 `fig_18_segment_error_composition_h15.png`: errors differ between high and low service-composition groups, but this target-time retrospective grouping is not a deployable future-composition feature.",
            "",
            "### Primary test-set usage record",
            "",
            "- Frozen prediction stage: the baseline script generates all horizons once. XGBoost and LSTM each make one final test prediction per horizon, three calls each, after all validation configurations are frozen.",
            "- Unified evaluation: `src/07_evaluate.py` reads existing prediction files for one reporting summary and does not refit or select models.",
            "- Later minimal ablation: the full feature set reuses the frozen h15 test prediction. Only two predefined ablation variants receive one test prediction each for final reporting.",
            "",
            marker_end,
        ]
    )
    log_path = project_path("docs", "experiment_log.md")
    existing = log_path.read_text(encoding="utf-8")
    if marker_start in existing and marker_end in existing:
        prefix, remainder = existing.split(marker_start, maxsplit=1)
        _, suffix = remainder.split(marker_end, maxsplit=1)
        updated = prefix.rstrip() + "\n\n" + block + suffix
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    log_path.write_text(updated, encoding="utf-8", newline="\n")


# Keep the historical verbose logger above under an explicitly unused name for
# provenance.  The public definition below is the compact UTF-8-safe record
# used by ``main``; both descriptions now state the same strict timing contract.
def update_experiment_log(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    segments: pd.DataFrame,
    thresholds: dict[str, float],
    figure_context: dict[str, object],
    bootstrap_replicates: int,
) -> None:
    marker_start = "<!-- WEEK5_FINAL_EVALUATION_START -->"
    marker_end = "<!-- WEEK5_FINAL_EVALUATION_END -->"
    main = (
        metrics.loc[metrics["horizon_minutes"].eq(15)]
        .set_index("model")
        .loc[list(MODEL_ORDER)]
    )
    mae_winner = str(main["mae"].idxmin())
    ap_winner = str(main["pr_auc"].idxmax())
    best_f1 = float(main["f1"].max())
    f1_winners = [
        str(model)
        for model, value in main["f1"].items()
        if np.isclose(float(value), best_f1, rtol=1e-9, atol=1e-12)
    ]
    f1_winner_text = ", ".join(f1_winners) + (" (tie)" if len(f1_winners) > 1 else "")
    actual_bursts = int(main["actual_burst_windows"].iloc[0])
    threshold = float(main["train_p95_threshold"].iloc[0])
    targets_per_horizon = int(
        predictions[["timestamp", "horizon"]].drop_duplicates().groupby("horizon").size().min()
    )
    burst_part = segments.loc[
        segments["dimension"].eq("actual_burst") & segments["segment"].eq("Burst")
    ].set_index("model")
    nonburst_part = segments.loc[
        segments["dimension"].eq("actual_burst") & segments["segment"].eq("Non-burst")
    ].set_index("model")
    burst_best = str(burst_part["mae"].idxmin())
    nonburst_best = str(nonburst_part["mae"].idxmin())
    week_start = pd.Timestamp(figure_context["representative_week_start"])
    week_end = pd.Timestamp(figure_context["representative_week_end"])
    burst_center = pd.Timestamp(figure_context["burst_center"])

    metric_lines = [
        "| Model | MAE | RMSE | sMAPE | vs Seasonal MAE | F1 | AP | TP/FP/FN/TN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, row in main.iterrows():
        metric_lines.append(
            f"| {model} | {_sig(row['mae'])} | {_sig(row['rmse'])} | "
            f"{_sig(row['smape_percent'])}% | {_sig(row['mae_improvement_vs_seasonal_percent'])}% | "
            f"{_sig(row['f1'])} | {_sig(row['pr_auc'])} | "
            f"{int(row['true_positive'])}/{int(row['false_positive'])}/"
            f"{int(row['false_negative'])}/{int(row['true_negative'])} |"
        )

    block = "\n".join(
        [
            marker_start,
            "## Week 5: frozen unified test evaluation",
            "",
            "- Freeze rule: baseline, XGBoost, and LSTM configurations were selected using validation evidence only; the test set is reporting-only.",
            f"- Common coverage: {len(predictions):,} long-form rows; four models at every timestamp+horizon; at least {targets_per_horizon:,} common target timestamps per horizon.",
            f"- Burst rule: one train-only P95 = {_sig(threshold)} Token/5-min; labels use token_load >= threshold for every model and dataset.",
            f"- Confidence information: 95% synthetic-UTC calendar-day block bootstrap, seed={SEED}, {bootstrap_replicates:,} replicates.",
            "- Test-use rule: src/07_evaluate.py reads frozen predictions and does not refit, select, or tune a model.",
            "",
            "### Main 15-minute results (three significant digits)",
            "",
            *metric_lines,
            "",
            f"- Overall result: {mae_winner} has the lowest MAE ({_sig(main.loc[mae_winner, 'mae'])} Token/5-min; {_sig(main.loc[mae_winner, 'mae_improvement_vs_seasonal_percent'])}% versus Seasonal Naive).",
            f"- Burst result: {actual_bursts} actual burst windows occur in the test set. Best fixed-threshold F1 is {f1_winner_text} ({_sig(best_f1)}); best ranking AP is {ap_winner} ({_sig(main.loc[ap_winner, 'pr_auc'])}).",
            f"- Segment result: non-burst MAE is lowest for {nonburst_best}; burst-window MAE is lowest for {burst_best}. Target-time GPT-4/API shares are used only for retrospective grouping.",
            f"- Composition cutoffs: train nonzero-window medians are GPT-4={_sig(thresholds['gpt4_share_train_median'])}, API={_sig(thresholds['api_share_train_median'])}.",
            "- Relative day/week phases describe the anonymized synthetic time axis and are not interpreted as real weekdays or weekends.",
            "",
            "### Figure conclusions",
            "",
            f"- Fig. 11: the objectively selected representative week ({week_start:%Y-%m-%d} to {(week_end - pd.Timedelta(minutes=5)):%Y-%m-%d}) compares ordinary-load tracking under one common unit and legend.",
            f"- Fig. 12: the local window is centered on the maximum observed test target ({burst_center.isoformat()}) and shows every forecast against the train P95.",
            f"- Fig. 13: {mae_winner} leads overall MAE, while the fixed-threshold F1 result must be interpreted separately.",
            f"- Fig. 14: {ap_winner} has the highest AP, which is ranking evidence rather than proof of adequate fixed-threshold detection.",
            "- Fig. 15: confusion matrices expose TP/FP/FN/TN counts and the effect of rare positive windows.",
            "- Fig. 16: burst-window errors are reported separately from non-burst errors rather than hidden by the overall average.",
            "- Fig. 17: relative phase profiles are diagnostic positions on the synthetic axis, not real calendar semantics.",
            "- Fig. 18: service-composition groups explain target-time errors only; they are not deployable future composition features.",
            "",
            marker_end,
        ]
    )
    log_path = project_path("docs", "experiment_log.md")
    existing = log_path.read_text(encoding="utf-8")
    if marker_start in existing and marker_end in existing:
        prefix, remainder = existing.split(marker_start, maxsplit=1)
        _, suffix = remainder.split(marker_end, maxsplit=1)
        updated = prefix.rstrip() + "\n\n" + block + suffix
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    log_path.write_text(updated, encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=1000,
        help="Calendar-day block bootstrap repetitions (default: 1000)",
    )
    parser.add_argument(
        "--skip-figures", action="store_true", help="Development-only table run"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive for the final run")
    ensure_output_directories()
    threshold = load_threshold()
    predictions, audit = build_unified_test_predictions(threshold)
    metrics = compute_final_metrics(predictions, threshold, args.bootstrap_replicates)
    save_result_tables(metrics)
    segments, composition_thresholds = build_segment_errors(predictions)
    if args.skip_figures:
        figure_context = {
            "representative_week_start": pd.NaT,
            "representative_week_end": pd.NaT,
            "burst_center": pd.NaT,
        }
    else:
        figure_context = create_core_figures(
            predictions,
            metrics,
            segments,
            composition_thresholds,
            threshold,
        )
        update_experiment_log(
            predictions,
            metrics,
            segments,
            composition_thresholds,
            figure_context,
            args.bootstrap_replicates,
        )
    print(
        f"Saved unified predictions: {project_path('outputs', 'tables', 'test_predictions_all_models.csv')} "
        f"({len(predictions):,} rows)"
    )
    print(f"Alignment checks passed for {len(audit)} model-horizon sources")
    print("\n15-minute final results:")
    print(
        metrics.loc[metrics["horizon_minutes"].eq(15), [
            "model",
            "mae",
            "rmse",
            "smape_percent",
            "mae_improvement_vs_seasonal_percent",
            "f1",
            "pr_auc",
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
        ]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
