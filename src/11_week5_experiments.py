"""Run the frozen Week-5 XGBoost ablation and BurstGPT-3 robustness test.

The ablation keeps the h15 XGBoost hyperparameters and chronological boundaries
fixed.  Only the two pre-specified reduced feature sets make new test prediction
calls; the full-feature result reuses the frozen main prediction.

The robustness design is an external zero-shot cross-period test: models fitted
on BurstGPT batches 1/2 are applied directly to separately cleaned and aggregated
batch 3.  No batch-3 labels are used for fitting, selection, scaling, thresholding,
or feature definition.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / "outputs" / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import joblib
import numpy as np
import pandas as pd
import yaml
from xgboost import XGBRegressor

from utils import ensure_output_directories, project_path


SEED = 42
HORIZONS = (5, 15, 60)
HORIZON_ORDER = {15: 0, 5: 1, 60: 2}
MODEL_ORDER = ("Persistence", "Seasonal Naive", "XGBoost", "LSTM")
MODEL_RANK = {model: rank for rank, model in enumerate(MODEL_ORDER)}
BATCH3_RAW = project_path(
    "Dataset", "level_5_BurstGPT_v2.0_without_fails_3_Undisclosed_GitHub.csv"
)
BATCH3_REQUESTS = project_path("data", "processed", "requests_3.parquet")
BATCH3_SERIES = project_path("data", "processed", "series_5min_burstgpt3.parquet")


def load_numeric_module(alias: str, filename: str) -> ModuleType:
    path = project_path("src", filename)
    specification = importlib.util.spec_from_file_location(alias, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(specification)
    # dataclasses and a few serialization helpers resolve annotations through
    # sys.modules while the module body is executing.
    sys.modules[alias] = module
    specification.loader.exec_module(module)
    return module


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_threshold() -> float:
    table = pd.read_csv(project_path("outputs", "tables", "table_01_burst_threshold.csv"))
    if len(table) != 1:
        raise AssertionError("Expected one frozen threshold row")
    return float(table.loc[0, "p95_threshold"])


def _write_three_significant(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.3g")


def ensure_clean_batch3(force: bool, chunksize: int) -> dict[str, object]:
    """Use the exact Week-1 cleaning function and preserve a separate artifact."""
    clean_module = load_numeric_module("clean_week1", "01_clean.py")
    if force or not BATCH3_REQUESTS.exists():
        print(f"Cleaning batch 3 with the frozen Week-1 rules: {BATCH3_RAW.name}", flush=True)
        record = clean_module.clean_and_audit_without_fails(
            batch="3",
            path=BATCH3_RAW,
            output_path=BATCH3_REQUESTS,
            chunksize=chunksize,
        )
    else:
        audit = pd.read_csv(project_path("outputs", "tables", "table_00_data_quality.csv"))
        match = audit.loc[audit["file_id"].eq("without_fails_3")]
        if len(match) != 1:
            raise AssertionError("Could not recover the frozen batch-3 cleaning audit")
        record = match.iloc[0].to_dict()
        record["clean_rows"] = int(pd.read_parquet(BATCH3_REQUESTS, columns=["timestamp"]).shape[0])
    audit_table = pd.DataFrame([record])
    _write_three_significant(
        audit_table, project_path("outputs", "tables", "table_07a_burstgpt3_cleaning_audit.csv")
    )
    return record


def build_batch3_series(threshold: float, force: bool) -> pd.DataFrame:
    """Reuse the Week-2 left-closed 5-minute aggregation without mixing batches."""
    if BATCH3_SERIES.exists() and not force:
        series = pd.read_parquet(BATCH3_SERIES)
        series.index = pd.DatetimeIndex(pd.to_datetime(series.index, utc=True), name="timestamp")
    else:
        build_module = load_numeric_module("build_series_week2", "02_build_series.py")
        columns = [
            "timestamp",
            "total_tokens",
            "request_tokens",
            "response_tokens",
            "model",
            "log_type",
            "source_batch",
        ]
        requests = pd.read_parquet(BATCH3_REQUESTS, columns=columns)
        requests["timestamp"] = pd.to_datetime(requests["timestamp"], utc=True, errors="raise")
        requests.sort_values("timestamp", kind="mergesort", inplace=True, ignore_index=True)
        if not requests["timestamp"].is_monotonic_increasing:
            raise AssertionError("Batch-3 cleaned requests are not chronological")
        if set(requests["source_batch"].astype(str).unique()) != {"3"}:
            raise AssertionError("Batch-3 robustness data contains another source batch")
        series = build_module.aggregate_complete_grid(requests, "5min")
        # aggregate_complete_grid has main-specific batch-1/2 provenance fields;
        # override only the provenance label, never an analytical measure.
        series["source_batch"] = "3"
        series.to_parquet(BATCH3_SERIES, compression="zstd")
        del requests
        gc.collect()
    series["split"] = "robustness_zero_shot"
    series["actual_burst"] = series["token_load"].ge(threshold)
    if not series.index.is_unique or not series.index.is_monotonic_increasing:
        raise AssertionError("Batch-3 series index must be unique and increasing")
    if not series.index.to_series().diff().dropna().eq(pd.Timedelta(minutes=5)).all():
        raise AssertionError("Batch-3 series is not a complete 5-minute grid")
    # Persist the fixed-threshold label and the explicit zero-shot role as part
    # of the final robustness series, not merely as in-memory evaluation state.
    series.to_parquet(BATCH3_SERIES, compression="zstd")
    return series


def ablation_feature_sets(full_features: list[str]) -> dict[str, list[str]]:
    feature_sets = {
        "full_features": list(full_features),
        "without_service_structure": [
            name
            for name in full_features
            if not name.startswith(("gpt4_share_", "api_share_"))
        ],
        "lags_plus_relative_calendar": [
            name
            for name in full_features
            if name.startswith("lag_") or name.startswith("relative_")
        ],
    }
    if len(feature_sets["full_features"]) != 36:
        raise AssertionError("Unexpected frozen full-feature count")
    if len(feature_sets["without_service_structure"]) != 30:
        raise AssertionError("Service-structure ablation must remove six GPT-4/API shares")
    if len(feature_sets["lags_plus_relative_calendar"]) != 12:
        raise AssertionError("Lag+calendar ablation must contain 8 lags and 4 phases")
    return feature_sets


def _ablation_metrics(
    evaluate: ModuleType,
    frame: pd.DataFrame,
    feature_set: str,
    split: str,
    n_features: int,
    threshold: float,
    seasonal_mae: float,
    test_prediction_calls: int,
) -> dict[str, object]:
    regression = evaluate.regression_metrics(frame["actual_token_load"], frame["predicted_token_load"])
    burst = evaluate.burst_metrics(
        frame["actual_token_load"], frame["predicted_token_load"], threshold
    )
    return {
        "horizon_minutes": 15,
        "feature_set": feature_set,
        "split": split,
        "n_features": n_features,
        "n_observations": len(frame),
        **regression,
        **burst,
        "seasonal_naive_mae": seasonal_mae,
        "mae_improvement_vs_seasonal_percent": 100.0
        * (seasonal_mae - float(regression["mae"]))
        / seasonal_mae,
        "hyperparameters": "frozen configs/xgb_h15.yaml",
        "split_boundaries": "unchanged target-time train/valid/test",
        "test_prediction_calls_in_week5_ablation": test_prediction_calls,
        "test_role": "final report only; never used to select the feature set",
    }


def run_xgb_h15_ablation(threshold: float) -> pd.DataFrame:
    """Evaluate exactly three pre-specified feature sets with frozen parameters."""
    evaluate = load_numeric_module("evaluate_week5_ablation", "07_evaluate.py")
    config = read_yaml(project_path("configs", "xgb_h15.yaml"))
    full_features = list(config["features"])
    feature_sets = ablation_feature_sets(full_features)
    fixed = dict(config["fixed_params"])
    selected = dict(config["selected_params"])
    target = "target_h15"
    train = pd.read_parquet(project_path("data", "processed", "features_h15_train.parquet"))
    valid = pd.read_parquet(project_path("data", "processed", "features_h15_valid.parquet"))
    test = pd.read_parquet(project_path("data", "processed", "features_h15_test.parquet"))
    baseline = pd.read_csv(project_path("outputs", "tables", "table_02_baseline_results.csv"))
    seasonal = baseline.loc[
        baseline["model"].eq("seasonal_naive") & baseline["horizon_minutes"].eq(15)
    ].set_index("split")["mae"]
    rows: list[dict[str, object]] = []

    for feature_set, features in feature_sets.items():
        if feature_set == "full_features":
            valid_prediction = pd.read_csv(
                project_path("outputs", "tables", "pred_xgb_tuned_valid_h15.csv")
            )
            test_prediction = pd.read_csv(
                project_path("outputs", "tables", "pred_xgb_test_h15.csv")
            )
            test_calls = 0
        else:
            validation_model = XGBRegressor(**fixed, **selected)
            validation_model.fit(train[features], train[target])
            valid_values = validation_model.predict(valid[features])
            valid_prediction = pd.DataFrame(
                {
                    "feature_time": valid["feature_time"],
                    "target_time": valid["target_time"],
                    "horizon_minutes": 15,
                    "model_stage": "ablation_validation",
                    "actual_token_load": valid[target],
                    "predicted_token_load": valid_values,
                    "actual_burst": valid["actual_burst"],
                    "predicted_burst": valid_values >= threshold,
                }
            )
            combined = pd.concat([train, valid], ignore_index=True)
            final_model = XGBRegressor(**fixed, **selected)
            final_model.fit(combined[features], combined[target])
            test_values = final_model.predict(test[features])  # one final call per reduced set
            test_prediction = pd.DataFrame(
                {
                    "feature_time": test["feature_time"],
                    "target_time": test["target_time"],
                    "horizon_minutes": 15,
                    "model_stage": "ablation_final",
                    "actual_token_load": test[target],
                    "predicted_token_load": test_values,
                    "actual_burst": test["actual_burst"],
                    "predicted_burst": test_values >= threshold,
                }
            )
            suffix = (
                "no_service_structure"
                if feature_set == "without_service_structure"
                else "lags_calendar"
            )
            joblib.dump(
                final_model,
                project_path("models", f"xgb_h15_ablation_{suffix}.joblib"),
            )
            test_prediction.to_csv(
                project_path("outputs", "tables", f"pred_xgb_ablation_test_h15_{suffix}.csv"),
                index=False,
            )
            test_calls = 1
            del validation_model, final_model, combined

        for split_name, frame in (("valid", valid_prediction), ("test", test_prediction)):
            expected = valid if split_name == "valid" else test
            np.testing.assert_array_equal(
                pd.to_datetime(frame["target_time"], utc=True).to_numpy(),
                pd.to_datetime(expected["target_time"], utc=True).to_numpy(),
            )
            np.testing.assert_array_equal(
                frame["actual_token_load"].to_numpy(dtype="float64"),
                expected[target].to_numpy(dtype="float64"),
            )
            rows.append(
                _ablation_metrics(
                    evaluate,
                    frame,
                    feature_set,
                    split_name,
                    len(features),
                    threshold,
                    float(seasonal.loc[split_name]),
                    test_calls if split_name == "test" else 0,
                )
            )
        gc.collect()

    result = pd.DataFrame(rows)
    validation_order = (
        result.loc[result["split"].eq("valid")]
        .sort_values("mae", kind="mergesort")["feature_set"]
        .tolist()
    )
    rank = {feature_set: position + 1 for position, feature_set in enumerate(validation_order)}
    selected_feature_set = validation_order[0]
    result["validation_mae_rank"] = result["feature_set"].map(rank)
    result["validation_selected_feature_set"] = selected_feature_set
    result["selected_on_validation"] = result["feature_set"].eq(selected_feature_set)
    result["_split_rank"] = result["split"].map({"valid": 0, "test": 1})
    result["_feature_rank"] = result["feature_set"].map(
        {"full_features": 0, "without_service_structure": 1, "lags_plus_relative_calendar": 2}
    )
    result.sort_values(["_split_rank", "_feature_rank"], kind="mergesort", inplace=True)
    result.drop(columns=["_split_rank", "_feature_rank"], inplace=True)
    _write_three_significant(
        result, project_path("outputs", "tables", "table_06a_xgb_h15_ablation.csv")
    )
    return result.reset_index(drop=True)


def build_batch3_horizon_dataset(
    series: pd.DataFrame, features: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    steps = horizon // 5
    target_name = f"target_h{horizon}"
    dataset = features.copy()
    dataset.insert(0, "feature_time", series.index)
    dataset.insert(1, "target_time", series.index + pd.Timedelta(minutes=horizon))
    dataset.insert(2, target_name, series["token_load"].shift(-steps).to_numpy())
    dataset.insert(3, "actual_burst", series["actual_burst"].shift(-steps).to_numpy())
    dataset = dataset.dropna().reset_index(drop=True)
    dataset["actual_burst"] = dataset["actual_burst"].astype(bool)
    expected = series["token_load"].reindex(pd.DatetimeIndex(dataset["target_time"]))
    np.testing.assert_array_equal(dataset[target_name].to_numpy(), expected.to_numpy())
    if not (dataset["target_time"] - dataset["feature_time"]).eq(
        pd.Timedelta(minutes=horizon)
    ).all():
        raise AssertionError(f"Batch-3 target alignment failed for h{horizon}")
    return dataset


def _robust_long_part(
    dataset: pd.DataFrame,
    horizon: int,
    model: str,
    predictions: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    target = f"target_h{horizon}"
    values = np.asarray(predictions, dtype="float64").reshape(-1)
    if len(values) != len(dataset) or not np.isfinite(values).all():
        raise AssertionError(f"Invalid robustness predictions for {model} h{horizon}")
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(dataset["target_time"], utc=True),
            "horizon": horizon,
            "model": model,
            "y_true": dataset[target].to_numpy(dtype="float64"),
            "y_pred": values,
            "actual_burst": dataset[target].ge(threshold).to_numpy(),
            "predicted_burst": values >= threshold,
        }
    )


def run_burstgpt3_zero_shot(
    series: pd.DataFrame, threshold: float, bootstrap_replicates: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluate = load_numeric_module("evaluate_week5_robust", "07_evaluate.py")
    features_module = load_numeric_module("features_week3_robust", "03_features.py")
    lstm_module = load_numeric_module("lstm_week4_robust", "06_lstm.py")
    features, _definitions = features_module.build_feature_matrix(series)
    lstm_features = lstm_module.build_lstm_features(series)
    tensorflow = lstm_module.import_tensorflow()
    parts: list[pd.DataFrame] = []

    for horizon in HORIZONS:
        dataset = build_batch3_horizon_dataset(series, features, horizon)
        target_times = pd.DatetimeIndex(dataset["target_time"])
        feature_times = pd.DatetimeIndex(dataset["feature_time"])
        persistence_source_times = feature_times - pd.Timedelta(minutes=5)
        persistence = series["token_load"].reindex(
            persistence_source_times
        ).to_numpy(dtype="float64")
        if not np.isfinite(persistence).all():
            raise AssertionError("Batch-3 persistence warm-up did not cover forecast origins")
        seasonal = series["token_load"].reindex(
            target_times - pd.Timedelta(days=1)
        ).to_numpy(dtype="float64")
        if not np.isfinite(seasonal).all():
            raise AssertionError("Batch-3 warm-up did not cover daily seasonal baseline")
        parts.append(_robust_long_part(dataset, horizon, "Persistence", persistence, threshold))
        parts.append(_robust_long_part(dataset, horizon, "Seasonal Naive", seasonal, threshold))

        xgb_config = read_yaml(project_path("configs", f"xgb_h{horizon}.yaml"))
        xgb_features = list(xgb_config["features"])
        xgb_model = joblib.load(project_path("models", f"xgb_h{horizon}.joblib"))
        xgb_prediction = xgb_model.predict(dataset[xgb_features])
        parts.append(_robust_long_part(dataset, horizon, "XGBoost", xgb_prediction, threshold))

        lstm_config = read_yaml(project_path("configs", f"lstm_h{horizon}.yaml"))
        sequence_features = list(lstm_config["features"])
        feature_scaler = joblib.load(
            project_path("models", f"lstm_feature_scaler_h{horizon}.joblib")
        )
        target_scaler = joblib.load(
            project_path("models", f"lstm_target_scaler_h{horizon}.joblib")
        )
        anchor = dataset[["feature_time", "target_time", f"target_h{horizon}", "actual_burst"]].copy()
        anchor.insert(2, "split", "robustness_zero_shot")
        sequence_set = lstm_module.prepare_sequence_set(
            lstm_features,
            anchor,
            "robustness_zero_shot",
            sequence_features,
            int(lstm_config["lookback"]),
            horizon // 5,
            feature_scaler,
            target_scaler,
            fit_scalers=False,
        )
        lstm_model = tensorflow.keras.models.load_model(
            project_path("models", f"lstm_h{horizon}.keras"), compile=False
        )
        scaled_prediction = lstm_model.predict(
            sequence_set.X,
            batch_size=int(lstm_config["training"]["batch_size"]),
            verbose=0,
        ).reshape(-1)
        lstm_prediction = lstm_module.inverse_predictions(target_scaler, scaled_prediction)
        np.testing.assert_array_equal(
            sequence_set.y_raw, dataset[f"target_h{horizon}"].to_numpy(dtype="float64")
        )
        parts.append(_robust_long_part(dataset, horizon, "LSTM", lstm_prediction, threshold))
        del dataset, xgb_model, xgb_prediction, lstm_model, sequence_set, scaled_prediction
        tensorflow.keras.backend.clear_session()
        gc.collect()

    long = pd.concat(parts, ignore_index=True)
    long["_model_rank"] = long["model"].map(MODEL_RANK)
    long.sort_values(["timestamp", "horizon", "_model_rank"], kind="mergesort", inplace=True)
    long.drop(columns="_model_rank", inplace=True)
    coverage = long.groupby(["timestamp", "horizon"]).agg(
        models=("model", "nunique"),
        y_true_values=("y_true", "nunique"),
        actual_label_values=("actual_burst", "nunique"),
    )
    if not coverage["models"].eq(4).all():
        raise AssertionError("Batch-3 robustness rows lack one of four models")
    if not coverage[["y_true_values", "actual_label_values"]].eq(1).all().all():
        raise AssertionError("Batch-3 models disagree on common target labels")
    long.to_csv(
        project_path("outputs", "tables", "burstgpt3_predictions_all_models.csv"),
        index=False,
        float_format="%.10g",
    )

    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        for model in MODEL_ORDER:
            part = long.loc[long["horizon"].eq(horizon) & long["model"].eq(model)].copy()
            intervals = evaluate.block_bootstrap_intervals(
                part,
                threshold,
                bootstrap_replicates,
                seed=SEED + 10_000 + horizon * 100 + MODEL_RANK[model],
            )
            rows.append(
                {
                    "dataset": "BurstGPT_without_fails_3",
                    "design": "external zero-shot cross-period test",
                    "training_data": "BurstGPT_without_fails_1/2 only",
                    "horizon_minutes": horizon,
                    "model": model,
                    "n_observations": len(part),
                    "train_1_2_p95_threshold": threshold,
                    **evaluate.regression_metrics(part["y_true"], part["y_pred"]),
                    **evaluate.burst_metrics(part["y_true"], part["y_pred"], threshold),
                    **intervals,
                    "confidence_level": 0.95,
                    "ci_method": "synthetic-UTC calendar-day block bootstrap",
                    "batch3_fit_or_selection_rows": 0,
                    "service_structure_fields": "same GPT-4/API definitions; Session ID and Elapsed time unused",
                    "relative_phase_anchor": "published relative seconds from synthetic UTC epoch",
                }
            )
    result = pd.DataFrame(rows)
    seasonal_mae = (
        result.loc[result["model"].eq("Seasonal Naive")]
        .set_index("horizon_minutes")["mae"]
        .to_dict()
    )
    result["seasonal_naive_mae"] = result["horizon_minutes"].map(seasonal_mae)
    result["mae_improvement_vs_seasonal_percent"] = 100.0 * (
        result["seasonal_naive_mae"] - result["mae"]
    ) / result["seasonal_naive_mae"]
    result["_horizon_rank"] = result["horizon_minutes"].map(HORIZON_ORDER)
    result["_model_rank"] = result["model"].map(MODEL_RANK)
    result.sort_values(["_horizon_rank", "_model_rank"], kind="mergesort", inplace=True)
    result.drop(columns=["_horizon_rank", "_model_rank"], inplace=True)
    _write_three_significant(
        result, project_path("outputs", "tables", "table_07_robustness_burstgpt3.csv")
    )
    return long.reset_index(drop=True), result.reset_index(drop=True)


def _sig(value: float) -> str:
    return f"{float(value):.3g}"


def update_experiment_log(
    cleaning: dict[str, object],
    series: pd.DataFrame,
    ablation: pd.DataFrame,
    robustness: pd.DataFrame,
    robustness_bootstrap_replicates: int,
) -> None:
    marker_start = "<!-- WEEK5_ABLATION_ROBUSTNESS_START -->"
    marker_end = "<!-- WEEK5_ABLATION_ROBUSTNESS_END -->"
    valid = ablation.loc[ablation["split"].eq("valid")].set_index("feature_set")
    test = ablation.loc[ablation["split"].eq("test")].set_index("feature_set")
    selected = str(valid["mae"].idxmin())
    robust_h15 = robustness.loc[robustness["horizon_minutes"].eq(15)].set_index("model").loc[
        list(MODEL_ORDER)
    ]
    robust_winner = str(robust_h15["mae"].idxmin())

    ablation_lines = [
        "| Feature set | Features | Validation MAE | Test MAE | Test improvement vs Seasonal Naive | Test F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for feature_set in (
        "full_features",
        "without_service_structure",
        "lags_plus_relative_calendar",
    ):
        ablation_lines.append(
            f"| {feature_set} | {int(valid.loc[feature_set, 'n_features'])} | "
            f"{_sig(valid.loc[feature_set, 'mae'])} | {_sig(test.loc[feature_set, 'mae'])} | "
            f"{_sig(test.loc[feature_set, 'mae_improvement_vs_seasonal_percent'])}% | "
            f"{_sig(test.loc[feature_set, 'f1'])} |"
        )

    robustness_lines = [
        "| Model | MAE | RMSE | sMAPE | Seasonal-Naive improvement | Precision | Recall | F1 | PR-AUC/AP | TP/FP/FN/TN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, row in robust_h15.iterrows():
        robustness_lines.append(
            f"| {model} | {_sig(row['mae'])} | {_sig(row['rmse'])} | {_sig(row['smape_percent'])}% | "
            f"{_sig(row['mae_improvement_vs_seasonal_percent'])}% | {_sig(row['precision'])} | "
            f"{_sig(row['recall'])} | {_sig(row['f1'])} | {_sig(row['pr_auc'])} | "
            f"{int(row['true_positive'])}/{int(row['false_positive'])}/"
            f"{int(row['false_negative'])}/{int(row['true_negative'])} |"
        )

    block = "\n".join(
        [
            marker_start,
            "## Week 5: Minimal ablation and cross-period robustness on BurstGPT_3",
            "",
            "### Minimal 15-minute XGBoost ablation",
            "",
            "- The predefined groups are the full feature set, the set without six historical GPT-4/API share features, and eight lags plus four relative phase features. All groups use the fixed `configs/xgb_h15.yaml` parameters and unchanged target-time boundaries.",
            "- Variants are ranked only by validation MAE, while the test set reports final results for all three groups. The full-feature test result reuses the primary prediction, and each reduced group receives one final test prediction. No configuration is changed from test results.",
            "",
            *ablation_lines,
            "",
            f"Ablation conclusion: `{selected}` ranks first on validation MAE. This ranking does not replace the frozen primary paper model. Test results describe feature-group contributions and are not used for another selection round.",
            "",
            "### BurstGPT_3 robustness design (single predefined protocol)",
            "",
            "- Design: **external zero-shot cross-period test**. Persistence, Seasonal Naive, XGBoost, LSTM, and scalers frozen on batches 1 and 2 are applied directly to batch 3. Batch 3 is not used for training, validation, or model selection.",
            f"- Cleaning: reuse `src/01_clean.py::clean_and_audit_without_fails`; clean `{BATCH3_RAW.name}` to {int(float(cleaning['clean_rows'])):,} requests and save `data/processed/requests_3.parquet`.",
            f"- Aggregation: reuse the left-closed, right-open five-minute contract from `src/02_build_series.py::aggregate_complete_grid`; create {len(series):,} independent windows from {series.index.min().isoformat()} to {series.index.max().isoformat()} without joining batches 1 and 2.",
            "- Threshold and features: keep the batches 1 and 2 training P95 of 315246.2 Token/5min; anchor relative daily and weekly phase to the synthetic UTC origin derived from public relative seconds. Do not use the batch 3 Session ID or Elapsed time fields.",
            f"- Confidence information: 95% synthetic UTC day-block bootstrap with seed 42 and {robustness_bootstrap_replicates:,} replicates.",
            "",
            "#### BurstGPT_3 15-minute results (three significant digits)",
            "",
            *robustness_lines,
            "",
            f"Robustness conclusion: {robust_winner} has the lowest zero-shot 15-minute MAE on batch 3 ({_sig(robust_h15.loc[robust_winner, 'mae'])} Token/5min). "
            "This result tests only cross-period transfer and does not estimate performance after retraining on batch 3. Burst results still use the fixed batches 1 and 2 training P95 as the capacity threshold.",
            "",
            "### Additional test-usage record",
            "",
            "- The primary test set receives only two new prediction calls, one for each predefined reduced XGBoost feature group. The full group reuses its frozen prediction. Ablation test labels never affect ranking.",
            "- BurstGPT_3 is independent cross-period data and does not count as use of the batches 1 and 2 primary test set. Each frozen model and horizon receives one zero-shot prediction call.",
            "- Parameter tuning and threshold changes stop at the end of Week 5. Later work is limited to code-error fixes, table and figure checks, and paper writing.",
            "",
            marker_end,
        ]
    )
    path = project_path("docs", "experiment_log.md")
    existing = path.read_text(encoding="utf-8")
    if marker_start in existing and marker_end in existing:
        prefix, remainder = existing.split(marker_start, maxsplit=1)
        _, suffix = remainder.split(marker_end, maxsplit=1)
        updated = prefix.rstrip() + "\n\n" + block + suffix
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    path.write_text(updated, encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-clean", action="store_true")
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunksize <= 0 or args.bootstrap_replicates <= 0:
        raise ValueError("chunksize and bootstrap replicates must be positive")
    ensure_output_directories()
    np.random.seed(SEED)
    threshold = load_threshold()
    cleaning = ensure_clean_batch3(args.force_clean, args.chunksize)
    series = build_batch3_series(threshold, args.force_clean)
    ablation = run_xgb_h15_ablation(threshold)
    _robust_long, robustness = run_burstgpt3_zero_shot(
        series, threshold, args.bootstrap_replicates
    )
    update_experiment_log(
        cleaning,
        series,
        ablation,
        robustness,
        args.bootstrap_replicates,
    )
    print("\nValidation-ranked XGBoost h15 ablation:")
    print(
        ablation.loc[:, [
            "split",
            "feature_set",
            "n_features",
            "mae",
            "f1",
            "validation_mae_rank",
        ]].to_string(index=False)
    )
    print("\nBurstGPT-3 zero-shot h15 robustness:")
    print(
        robustness.loc[robustness["horizon_minutes"].eq(15), [
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
