"""Reproducible preliminary EDA and XGBoost results for the paper draft.

The script only reads the two primary ``without_fails`` BurstGPT files and
writes derived tables/figures under ``outputs/``.  It deliberately leaves the
raw request records untouched.  A 15-minute-ahead task (three 5-minute bins)
is evaluated with chronological 70/15/15 splits.  The LSTM comparison remains
outside this preliminary run because TensorFlow is not installed yet.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_MPL_CONFIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / ".matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)
from xgboost import XGBRegressor

from utils import ensure_output_directories, project_path, set_global_seed


FREQUENCY_SECONDS = 300
HORIZON_BINS = 3  # 15 minutes ahead at 5-minute resolution
SEASONAL_PERIOD_BINS = 288  # 24 hours / 5 minutes
PRIMARY_FILES = (
    "level_5_BurstGPT_v2.0_without_fails_1_Undisclosed_GitHub.csv",
    "level_5_BurstGPT_v2.0_without_fails_2_Undisclosed_GitHub.csv",
)
REQUIRED_COLUMNS = {
    "Timestamp",
    "Model",
    "Request tokens",
    "Response tokens",
    "Total tokens",
    "Log Type",
}


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def load_5min_series() -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Aggregate request records to consecutive 5-minute bins in chunks."""
    partials: list[pd.DataFrame] = []
    source_rows = 0
    invalid_rows = 0
    model_counts: dict[str, int] = {}
    log_type_counts: dict[str, int] = {}

    for filename in PRIMARY_FILES:
        path = project_path("Dataset", filename)
        if not path.exists():
            raise FileNotFoundError(f"Missing primary input file: {path}")

        for chunk in pd.read_csv(path, chunksize=200_000):
            missing = REQUIRED_COLUMNS.difference(chunk.columns)
            if missing:
                raise ValueError(f"{filename} is missing columns: {sorted(missing)}")

            source_rows += len(chunk)
            numeric_columns = ["Timestamp", "Request tokens", "Response tokens", "Total tokens"]
            chunk[numeric_columns] = chunk[numeric_columns].apply(
                pd.to_numeric, errors="coerce"
            )
            valid = (
                chunk[numeric_columns].notna().all(axis=1)
                & (chunk["Request tokens"] >= 0)
                & (chunk["Response tokens"] >= 0)
                & (chunk["Total tokens"] >= 0)
                & (
                    chunk["Request tokens"] + chunk["Response tokens"]
                    == chunk["Total tokens"]
                )
            )
            invalid_rows += int((~valid).sum())
            chunk = chunk.loc[valid].copy()

            for value, count in chunk["Model"].value_counts(dropna=False).items():
                label = str(value)
                model_counts[label] = model_counts.get(label, 0) + int(count)
            for value, count in chunk["Log Type"].value_counts(dropna=False).items():
                label = str(value)
                log_type_counts[label] = log_type_counts.get(label, 0) + int(count)

            bins = (chunk["Timestamp"].astype("int64") // FREQUENCY_SECONDS) * FREQUENCY_SECONDS
            prepared = pd.DataFrame(
                {
                    "bin_start_s": bins,
                    "token_load": chunk["Total tokens"].astype("float64"),
                    "request_count": 1,
                    "input_tokens": chunk["Request tokens"].astype("float64"),
                    "output_tokens": chunk["Response tokens"].astype("float64"),
                    "gpt4_request_count": chunk["Model"].eq("GPT-4").astype("int64"),
                    "api_request_count": chunk["Log Type"].eq("API log").astype("int64"),
                }
            )
            partials.append(prepared.groupby("bin_start_s", as_index=False).sum())

    grouped = pd.concat(partials, ignore_index=True).groupby("bin_start_s", as_index=False).sum()
    first_bin = int(grouped["bin_start_s"].min())
    last_bin = int(grouped["bin_start_s"].max())
    full_index = np.arange(first_bin, last_bin + FREQUENCY_SECONDS, FREQUENCY_SECONDS)
    series = grouped.set_index("bin_start_s").reindex(full_index, fill_value=0.0)
    series.index.name = "bin_start_s"
    series["gpt4_share"] = np.divide(
        series["gpt4_request_count"],
        series["request_count"],
        out=np.zeros(len(series), dtype=float),
        where=series["request_count"].to_numpy() > 0,
    )
    series["api_share"] = np.divide(
        series["api_request_count"],
        series["request_count"],
        out=np.zeros(len(series), dtype=float),
        where=series["request_count"].to_numpy() > 0,
    )

    metadata: dict[str, float | int] = {
        "source_rows": source_rows,
        "invalid_rows": invalid_rows,
        "first_timestamp_s": first_bin,
        "last_timestamp_s": last_bin,
        "duration_days": (last_bin - first_bin + FREQUENCY_SECONDS) / 86400,
        "model_chatgpt_requests": model_counts.get("ChatGPT", 0),
        "model_gpt4_requests": model_counts.get("GPT-4", 0),
        "api_requests": log_type_counts.get("API log", 0),
        "conversation_requests": log_type_counts.get("Conversation log", 0),
    }
    return series, metadata


def describe_series(series: pd.DataFrame, metadata: dict[str, float | int]) -> dict[str, float | int]:
    """Create paper-ready descriptive statistics for 5-minute token load."""
    load = series["token_load"]
    daily = load.groupby(load.index // 86400).sum()
    hourly_mean = load.groupby((load.index % 86400) // 3600).mean()
    stats: dict[str, float | int] = {
        **metadata,
        "five_minute_bins": int(len(series)),
        "total_tokens": float(load.sum()),
        "total_requests": int(series["request_count"].sum()),
        "mean_tokens_per_5min": float(load.mean()),
        "median_tokens_per_5min": float(load.median()),
        "p95_tokens_per_5min": float(load.quantile(0.95)),
        "p99_tokens_per_5min": float(load.quantile(0.99)),
        "max_tokens_per_5min": float(load.max()),
        "coefficient_of_variation": float(load.std(ddof=1) / load.mean()),
        "zero_load_bin_share": float((load == 0).mean()),
        "max_daily_tokens": float(daily.max()),
        "median_daily_tokens": float(daily.median()),
        "peak_hour_phase": int(hourly_mean.idxmax()),
        "trough_hour_phase": int(hourly_mean.idxmin()),
        "peak_to_trough_hourly_mean_ratio": _safe_rate(hourly_mean.max(), hourly_mean.min()),
        "gpt4_request_share": _safe_rate(
            metadata["model_gpt4_requests"], metadata["source_rows"]
        ),
        "api_request_share": _safe_rate(metadata["api_requests"], metadata["source_rows"]),
    }
    return stats


def build_feature_frame(series: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build only features observable at the end of the forecast-origin bin."""
    frame = series.copy()
    for lag in (1, 2, 3, 6, 12, 24, 48, SEASONAL_PERIOD_BINS):
        frame[f"load_lag_{lag}"] = frame["token_load"].shift(lag)
    for window in (3, 6, 12, 24):
        shifted = frame["token_load"].shift(1)
        frame[f"load_rollmean_{window}"] = shifted.rolling(window).mean()
        frame[f"load_rollstd_{window}"] = shifted.rolling(window).std()
    frame["load_change_1"] = frame["token_load"].diff(1)
    phase = frame.index.to_numpy(dtype=float)
    frame["tod_sin"] = np.sin(2 * np.pi * (phase % 86400) / 86400)
    frame["tod_cos"] = np.cos(2 * np.pi * (phase % 86400) / 86400)
    frame["dow_sin"] = np.sin(2 * np.pi * (phase % 604800) / 604800)
    frame["dow_cos"] = np.cos(2 * np.pi * (phase % 604800) / 604800)
    frame["target"] = frame["token_load"].shift(-HORIZON_BINS)
    # The seasonal baseline uses the matching bin from the prior day, not a
    # contemporaneous or future observation.
    frame["seasonal_naive"] = frame["token_load"].shift(
        SEASONAL_PERIOD_BINS - HORIZON_BINS
    )
    frame["persistence"] = frame["token_load"]
    feature_columns = [
        column
        for column in frame.columns
        if column.startswith("load_")
        or column in {"request_count", "input_tokens", "output_tokens", "gpt4_share", "api_share", "tod_sin", "tod_cos", "dow_sin", "dow_cos"}
    ]
    return frame.dropna().copy(), feature_columns


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, burst_threshold: float) -> dict[str, float]:
    """Calculate continuous-load and thresholded-burst metrics."""
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    y_true = np.asarray(y_true, dtype=float)
    actual_burst = y_true >= burst_threshold
    predicted_burst = y_pred >= burst_threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual_burst, predicted_burst, average="binary", zero_division=0
    )
    # Scores, rather than only thresholded labels, are used for PR-AUC.
    pr_auc = average_precision_score(actual_burst, y_pred)
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "smape": float(smape),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "actual_burst_windows": int(actual_burst.sum()),
        "predicted_burst_windows": int(predicted_burst.sum()),
        "true_positive_windows": int((actual_burst & predicted_burst).sum()),
    }


def run_models(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit a fixed XGBoost specification and evaluate chronological splits."""
    n_rows = len(frame)
    train_end = int(n_rows * 0.70)
    validation_end = int(n_rows * 0.85)
    train = frame.iloc[:train_end]
    validation = frame.iloc[train_end:validation_end]
    test = frame.iloc[validation_end:]
    burst_threshold = float(train["target"].quantile(0.95))

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[feature_columns], np.log1p(train["target"]))

    result_rows: list[dict[str, float | str | int]] = []
    prediction_rows: list[pd.DataFrame] = []
    for split_name, split in (("validation", validation), ("test", test)):
        predictions = {
            "persistence": split["persistence"].to_numpy(),
            "seasonal_naive": split["seasonal_naive"].to_numpy(),
            "xgboost": np.expm1(model.predict(split[feature_columns])),
        }
        for model_name, y_pred in predictions.items():
            metrics = calculate_metrics(split["target"].to_numpy(), y_pred, burst_threshold)
            result_rows.append(
                {
                    "split": split_name,
                    "model": model_name,
                    "horizon_minutes": HORIZON_BINS * 5,
                    "n_observations": len(split),
                    "train_burst_p95": burst_threshold,
                    **metrics,
                }
            )
            if split_name == "test":
                prediction_rows.append(
                    pd.DataFrame(
                        {
                            "forecast_origin_s": split.index,
                            "target_s": split.index + HORIZON_BINS * FREQUENCY_SECONDS,
                            "model": model_name,
                            "y_true": split["target"].to_numpy(),
                            "y_pred": np.clip(y_pred, 0, None),
                            "actual_burst": split["target"].to_numpy() >= burst_threshold,
                            "predicted_burst": np.clip(y_pred, 0, None) >= burst_threshold,
                        }
                    )
                )

    results = pd.DataFrame(result_rows)
    seasonal_mae_by_split = results.loc[
        results["model"] == "seasonal_naive", ["split", "mae"]
    ].set_index("split")["mae"]
    results["mae_improvement_vs_seasonal"] = results.apply(
        lambda row: 1 - row["mae"] / seasonal_mae_by_split[row["split"]], axis=1
    )
    importance = pd.DataFrame(
        {"feature": feature_columns, "gain_importance": model.feature_importances_}
    ).sort_values("gain_importance", ascending=False)
    return results, pd.concat(prediction_rows, ignore_index=True), importance


def save_plot(predictions: pd.DataFrame, output_path: Path) -> None:
    """Save a short, interpretable test-window comparison plot."""
    wide = predictions.pivot(index="target_s", columns="model", values="y_pred")
    observed = predictions.drop_duplicates("target_s").set_index("target_s")["y_true"]
    window = min(576, len(wide))  # Two days at 5-minute resolution.
    wide = wide.iloc[:window]
    observed = observed.loc[wide.index]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(observed.index, observed.values, color="black", linewidth=1.4, label="Observed")
    for name, color in (("persistence", "#9e9e9e"), ("seasonal_naive", "#4575b4"), ("xgboost", "#d73027")):
        ax.plot(wide.index, wide[name], linewidth=1.0, alpha=0.9, label=name)
    ax.set_xlabel("Relative timestamp (seconds)")
    ax.set_ylabel("Token load per 5 minutes")
    ax.set_title("Preliminary 15-minute-ahead token-load forecasts (first two test days)")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    set_global_seed(42)
    ensure_output_directories()
    series, metadata = load_5min_series()
    descriptive = describe_series(series, metadata)
    frame, feature_columns = build_feature_frame(series)
    results, predictions, importance = run_models(frame, feature_columns)

    tables_dir = project_path("outputs", "tables")
    figures_dir = project_path("outputs", "figures")
    series.reset_index().to_csv(tables_dir / "preliminary_5min_series.csv", index=False)
    results.round(6).to_csv(tables_dir / "preliminary_model_metrics_h15.csv", index=False)
    predictions.to_csv(tables_dir / "preliminary_test_predictions_h15.csv", index=False)
    importance.round(6).to_csv(tables_dir / "preliminary_xgb_feature_importance_h15.csv", index=False)
    with (tables_dir / "preliminary_descriptive_statistics.json").open("w", encoding="utf-8") as handle:
        json.dump(descriptive, handle, ensure_ascii=False, indent=2)
    save_plot(predictions, figures_dir / "fig_preliminary_h15_test_window.png")

    print("Preliminary analysis complete.")
    print(json.dumps(descriptive, ensure_ascii=False, indent=2))
    print(results.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
