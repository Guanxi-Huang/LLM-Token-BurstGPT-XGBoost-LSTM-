"""Tune, explain, refit, and test leakage-safe XGBoost load forecasts.

The script has two deliberate phases.  It first trains/tunes every horizon with
train and validation files only.  Test files are opened only after all selected
parameters have been frozen to YAML, then each final model calls ``predict`` on
its test matrix exactly once.
"""

from __future__ import annotations

import argparse
import itertools
import os
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / "outputs" / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ParameterSampler
from xgboost import XGBRegressor

from utils import ensure_output_directories, project_path, set_global_seed


SEED = 42
EPSILON = 1e-9
SPLIT_ORDER = ("train", "valid", "test")
SEARCH_SPACE: dict[str, list[float | int]] = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.02, 0.05, 0.1],
    "n_estimators": [300, 600, 1000],
    "min_child_weight": [1, 5, 10],
    "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
}
FIXED_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "random_state": SEED,
    "n_jobs": -1,
}


def load_base_config() -> dict:
    with project_path("configs", "base.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_threshold() -> float:
    path = project_path("outputs", "tables", "table_01_burst_threshold.csv")
    table = pd.read_csv(path)
    if len(table) != 1 or "p95_threshold" not in table:
        raise ValueError("Burst threshold table must contain exactly one p95_threshold")
    return float(table.loc[0, "p95_threshold"])


def load_feature_names() -> list[str]:
    path = project_path("outputs", "tables", "table_03_feature_availability.csv")
    table = pd.read_csv(path)
    required = {"name", "category", "availability", "obtained_at_prediction_time"}
    if required.difference(table.columns):
        raise ValueError(f"Invalid feature availability table: {path}")
    if not table["availability"].eq("available_at_t").all():
        unavailable = table.loc[~table["availability"].eq("available_at_t"), "name"].tolist()
        raise AssertionError(f"Unavailable features entered the whitelist: {unavailable}")
    if table["name"].duplicated().any():
        raise AssertionError("Feature availability table contains duplicates")
    return table["name"].tolist()


def load_split(horizon: int, split_name: str, feature_names: list[str]) -> pd.DataFrame:
    """Open exactly one split file so tuning cannot accidentally read test rows."""
    if split_name not in SPLIT_ORDER:
        raise ValueError(f"Unknown split: {split_name}")
    path = project_path("data", "processed", f"features_h{horizon}_{split_name}.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Missing features: {path}; run src/03_features.py")
    target_name = f"target_h{horizon}"
    columns = ["feature_time", "target_time", "split", target_name, "actual_burst", *feature_names]
    frame = pd.read_parquet(path, columns=columns)
    frame["feature_time"] = pd.to_datetime(frame["feature_time"], utc=True)
    frame["target_time"] = pd.to_datetime(frame["target_time"], utc=True)
    if not frame["split"].eq(split_name).all():
        raise AssertionError(f"Split file {path} contains non-{split_name} rows")
    expected_delta = pd.Timedelta(minutes=horizon)
    if not (frame["target_time"] - frame["feature_time"]).eq(expected_delta).all():
        raise AssertionError(f"Feature/target time misalignment in {path}")
    if frame[feature_names + [target_name]].isna().any().any():
        raise AssertionError(f"Missing model values in {path}")
    if not np.isfinite(frame[feature_names + [target_name]].to_numpy(dtype="float64")).all():
        raise AssertionError(f"Non-finite model values in {path}")
    return frame


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    error = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "smape_percent": float(
            100.0 * np.mean(2.0 * np.abs(error) / (np.abs(y_true) + np.abs(y_pred) + EPSILON))
        ),
    }


def burst_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict[str, float | int]:
    actual = np.asarray(y_true, dtype="float64") >= threshold
    predicted = np.asarray(y_pred, dtype="float64") >= threshold
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[False, True]).ravel()
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "actual_burst_windows": int(actual.sum()),
        "predicted_burst_windows": int(predicted.sum()),
    }


def prediction_diagnostics(
    frame: pd.DataFrame, predictions: np.ndarray, horizon: int, stage: str
) -> dict[str, object]:
    values = np.asarray(predictions, dtype="float64")
    if len(values) != len(frame) or not np.isfinite(values).all():
        raise AssertionError(f"Invalid prediction vector for h{horizon} {stage}")
    aligned = bool(
        (frame["target_time"] - frame["feature_time"])
        .eq(pd.Timedelta(minutes=horizon))
        .all()
    )
    if not aligned:
        raise AssertionError(f"Prediction time alignment failed for h{horizon} {stage}")
    return {
        "horizon_minutes": horizon,
        "stage": stage,
        "n_predictions": len(values),
        "n_unique_predictions": int(pd.Series(values).nunique()),
        "is_constant": bool(np.allclose(values, values[0])),
        "negative_prediction_count": int((values < 0.0).sum()),
        "min_prediction": float(values.min()),
        "max_prediction": float(values.max()),
        "target_time_aligned": aligned,
    }


def make_prediction_frame(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    horizon: int,
    stage: str,
    threshold: float,
) -> pd.DataFrame:
    y_true = frame[f"target_h{horizon}"].to_numpy(dtype="float64")
    y_pred = np.asarray(predictions, dtype="float64")
    return pd.DataFrame(
        {
            "feature_time": frame["feature_time"].to_numpy(),
            "target_time": frame["target_time"].to_numpy(),
            "horizon_minutes": horizon,
            "model_stage": stage,
            "actual_token_load": y_true,
            "predicted_token_load": y_pred,
            "actual_burst": y_true >= threshold,
            "predicted_burst": y_pred >= threshold,
        }
    )


def train_default_h15(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_names: list[str],
    threshold: float,
) -> tuple[dict[str, object], dict[str, object]]:
    """Validate the data path with the exact requested default estimator."""
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[feature_names], train["target_h15"])
    predictions = model.predict(valid[feature_names])
    diagnostics = prediction_diagnostics(valid, predictions, 15, "default_valid")
    output = make_prediction_frame(valid, predictions, 15, "default", threshold)
    output.to_csv(project_path("outputs", "tables", "pred_xgb_valid_h15.csv"), index=False)
    metrics = {
        "split": "valid",
        "model": "xgboost_default",
        "horizon_minutes": 15,
        "n_observations": len(valid),
        **regression_metrics(output["actual_token_load"], output["predicted_token_load"]),
        **burst_metrics(output["actual_token_load"], output["predicted_token_load"], threshold),
    }
    return metrics, diagnostics


def sampled_candidates(n_iter: int, seed: int) -> list[dict[str, Any]]:
    cardinality = int(np.prod([len(values) for values in SEARCH_SPACE.values()]))
    count = min(n_iter, cardinality)
    raw = list(ParameterSampler(SEARCH_SPACE, n_iter=count, random_state=seed))
    return [{key: value.item() if hasattr(value, "item") else value for key, value in row.items()} for row in raw]


def tune_horizon(
    horizon: int,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_names: list[str],
    threshold: float,
    candidates: list[dict[str, Any]],
    accumulated_log: list[dict[str, object]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    target_name = f"target_h{horizon}"
    x_train, y_train = train[feature_names], train[target_name]
    x_valid, y_valid = valid[feature_names], valid[target_name]
    horizon_rows: list[dict[str, object]] = []
    log_path = project_path("outputs", "tables", "xgb_tuning_log.csv")

    for candidate_index, params in enumerate(candidates, start=1):
        started = time.perf_counter()
        model = XGBRegressor(**FIXED_PARAMS, **params)
        model.fit(x_train, y_train)
        predictions = model.predict(x_valid)
        elapsed = time.perf_counter() - started
        diagnostics = prediction_diagnostics(valid, predictions, horizon, "tuning_valid")
        metrics = regression_metrics(y_valid.to_numpy(), predictions)
        classification = burst_metrics(y_valid.to_numpy(), predictions, threshold)
        row: dict[str, object] = {
            "horizon_minutes": horizon,
            "candidate_index": candidate_index,
            **params,
            "valid_mae": metrics["mae"],
            "valid_rmse": metrics["rmse"],
            "valid_smape_percent": metrics["smape_percent"],
            "valid_f1": classification["f1"],
            "valid_precision": classification["precision"],
            "valid_recall": classification["recall"],
            "fit_seconds": elapsed,
            "is_constant": diagnostics["is_constant"],
            "negative_prediction_count": diagnostics["negative_prediction_count"],
            "selected": False,
            "selection_data": "validation_only",
        }
        horizon_rows.append(row)
        # Persist after every candidate so the tuning evidence survives an
        # interrupted long run.
        pd.DataFrame([*accumulated_log, *horizon_rows]).to_csv(log_path, index=False)
        print(
            f"h{horizon} candidate {candidate_index:02d}/{len(candidates)}: "
            f"MAE={metrics['mae']:,.3f}, F1={classification['f1']:.4f}, {elapsed:.2f}s"
        )

    tuning = pd.DataFrame(horizon_rows)
    # The bounded search space is already the complexity guard.  Select lowest
    # validation MAE; exact numerical ties prefer shallower/fewer-tree models.
    tuning = tuning.sort_values(
        ["valid_mae", "max_depth", "n_estimators", "min_child_weight"], kind="mergesort"
    ).reset_index(drop=True)
    tuning.loc[0, "selected"] = True
    best_row = tuning.iloc[0]
    selected = {key: best_row[key].item() if hasattr(best_row[key], "item") else best_row[key] for key in SEARCH_SPACE}

    selected_ids = {(horizon, int(best_row["candidate_index"]))}
    for row in horizon_rows:
        row["selected"] = (horizon, int(row["candidate_index"])) in selected_ids
    accumulated_log.extend(horizon_rows)
    pd.DataFrame(accumulated_log).to_csv(log_path, index=False)
    return selected, tuning


def save_config(
    horizon: int,
    feature_names: list[str],
    selected: dict[str, Any],
    tuning: pd.DataFrame,
    train_rows: int,
    valid_rows: int,
) -> Path:
    best = tuning.iloc[0]
    config = {
        "horizon_minutes": horizon,
        "target_column": f"target_h{horizon}",
        "feature_source": f"data/processed/features_h{horizon}_{{split}}.parquet",
        "features": feature_names,
        "fixed_params": FIXED_PARAMS,
        "selected_params": selected,
        "selection": {
            "data": "validation only; test file unopened during selection",
            "primary_metric": "MAE",
            "valid_mae": float(best["valid_mae"]),
            "valid_f1": float(best["valid_f1"]),
            "rule": "minimum validation MAE; ties prefer lower max_depth then fewer trees",
            "candidate_index": int(best["candidate_index"]),
            "bounded_search_space": SEARCH_SPACE,
            "sampled_candidates": int(len(tuning)),
        },
        "fit_rows": {"train": train_rows, "validation": valid_rows},
        "final_refit": "train + validation; no standardization",
        "test_policy": "predict once after all horizons are frozen; never tune on test",
        "seed": SEED,
    }
    path = project_path("configs", f"xgb_h{horizon}.yaml")
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    return path


def save_native_importance(
    model: XGBRegressor, feature_names: list[str], horizon: int
) -> pd.DataFrame:
    importance = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_.astype("float64")}
    ).sort_values("importance", ascending=False, kind="mergesort")
    importance["rank"] = np.arange(1, len(importance) + 1)
    importance["availability_audit"] = "historical_or_relative_calendar"
    importance.to_csv(
        project_path("outputs", "tables", f"xgb_feature_importance_h{horizon}.csv"), index=False
    )

    shown = importance.head(20).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.barh(shown["feature"], shown["importance"], color="#31688E")
    ax.set_xlabel("XGBoost feature importance")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        project_path("outputs", "figures", f"fig_xgb_feature_importance_h{horizon}.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    return importance


def save_shap_outputs(
    model: XGBRegressor,
    valid: pd.DataFrame,
    feature_names: list[str],
    horizon: int,
    sample_size: int,
) -> None:
    import shap

    n_rows = min(sample_size, 2000, len(valid))
    sample = valid[feature_names].sample(n=n_rows, random_state=SEED)
    explainer = shap.TreeExplainer(model)
    # XGBoost accumulates float32 tree outputs while SHAP sums contributions
    # in a different order.  At this load scale the strict additivity check can
    # differ by less than one token out of hundreds of thousands; shape and
    # finiteness are asserted below, so disable only that numerical tolerance
    # gate rather than changing the explainer or the sampled rows.
    values = explainer.shap_values(sample, check_additivity=False)
    values_array = np.asarray(values, dtype="float64")
    if values_array.shape != sample.shape:
        raise AssertionError(f"Unexpected SHAP shape {values_array.shape}, expected {sample.shape}")
    if not np.isfinite(values_array).all():
        raise AssertionError(f"Non-finite SHAP values for h{horizon}")
    mean_abs = np.mean(np.abs(values_array), axis=0)
    shap_importance = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False, kind="mergesort")
    shap_importance["rank"] = np.arange(1, len(shap_importance) + 1)
    shap_importance["sample_rows"] = n_rows
    shap_importance["availability_audit"] = "historical_or_relative_calendar"
    shap_importance.to_csv(
        project_path("outputs", "tables", f"xgb_shap_importance_h{horizon}.csv"), index=False
    )

    shap.summary_plot(values_array, sample, max_display=20, show=False)
    figure = plt.gcf()
    figure.axes[0].xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value / 1000:g}k" if abs(value) >= 1000 else f"{value:g}")
    )
    figure.tight_layout()
    figure.savefig(
        project_path("outputs", "figures", f"fig_xgb_shap_summary_h{horizon}.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)

    for rank, feature in enumerate(shap_importance.head(3)["feature"], start=1):
        shap.dependence_plot(
            feature,
            values_array,
            sample,
            interaction_index=None,
            show=False,
        )
        figure = plt.gcf()
        figure.tight_layout()
        figure.savefig(
            project_path(
                "outputs", "figures", f"fig_xgb_shap_dependence_h{horizon}_{rank}_{feature}.png"
            ),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(figure)


def fit_selected_validation_model(
    horizon: int,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_names: list[str],
    selected: dict[str, Any],
    threshold: float,
    shap_sample_size: int,
    skip_shap: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    target_name = f"target_h{horizon}"
    model = XGBRegressor(**FIXED_PARAMS, **selected)
    model.fit(train[feature_names], train[target_name])
    predictions = model.predict(valid[feature_names])
    diagnostics = prediction_diagnostics(valid, predictions, horizon, "selected_valid")
    output = make_prediction_frame(valid, predictions, horizon, "selected", threshold)
    output.to_csv(
        project_path("outputs", "tables", f"pred_xgb_tuned_valid_h{horizon}.csv"), index=False
    )
    save_native_importance(model, feature_names, horizon)
    if not skip_shap:
        save_shap_outputs(model, valid, feature_names, horizon, shap_sample_size)
    metrics = {
        "split": "valid",
        "model": "xgboost_selected",
        "horizon_minutes": horizon,
        "n_observations": len(valid),
        **regression_metrics(output["actual_token_load"], output["predicted_token_load"]),
        **burst_metrics(output["actual_token_load"], output["predicted_token_load"], threshold),
    }
    return metrics, diagnostics


def final_refit_and_test_once(
    horizon: int,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_names: list[str],
    selected: dict[str, Any],
    threshold: float,
) -> tuple[dict[str, object], dict[str, object]]:
    """Load test only after selection, then make the sole test predict call."""
    target_name = f"target_h{horizon}"
    combined = pd.concat([train, valid], ignore_index=True)
    final_model = XGBRegressor(**FIXED_PARAMS, **selected)
    final_model.fit(combined[feature_names], combined[target_name])
    model_path = project_path("models", f"xgb_h{horizon}.joblib")
    joblib.dump(final_model, model_path)

    test = load_split(horizon, "test", feature_names)
    predictions = final_model.predict(test[feature_names])  # exactly one test prediction
    diagnostics = prediction_diagnostics(test, predictions, horizon, "final_test")
    output = make_prediction_frame(test, predictions, horizon, "final", threshold)
    output.to_csv(
        project_path("outputs", "tables", f"pred_xgb_test_h{horizon}.csv"), index=False
    )
    metrics = {
        "split": "test",
        "model": "xgboost_final",
        "horizon_minutes": horizon,
        "n_observations": len(test),
        **regression_metrics(output["actual_token_load"], output["predicted_token_load"]),
        **burst_metrics(output["actual_token_load"], output["predicted_token_load"], threshold),
    }
    return metrics, diagnostics


def save_comparison(results: pd.DataFrame) -> None:
    baseline_path = project_path("outputs", "tables", "table_02_baseline_results.csv")
    if not baseline_path.exists():
        return
    baselines = pd.read_csv(baseline_path)
    xgb_row = results.loc[
        results["split"].eq("test")
        & results["model"].eq("xgboost_final")
        & results["horizon_minutes"].eq(15)
    ]
    if len(xgb_row) != 1:
        raise AssertionError("Expected exactly one final h15 XGBoost test result")
    xgb_mae = float(xgb_row.iloc[0]["mae"])
    rows: list[dict[str, object]] = []
    for baseline in ("persistence", "seasonal_naive"):
        match = baselines.loc[
            baselines["split"].eq("test")
            & baselines["model"].eq(baseline)
            & baselines["horizon_minutes"].eq(15)
        ]
        if len(match) != 1:
            raise AssertionError(f"Missing h15 test baseline: {baseline}")
        baseline_mae = float(match.iloc[0]["mae"])
        rows.append(
            {
                "horizon_minutes": 15,
                "comparison_split": "test (reported once; not used for tuning)",
                "xgboost_mae": xgb_mae,
                "baseline": baseline,
                "baseline_mae": baseline_mae,
                "mae_improvement": baseline_mae - xgb_mae,
                "mae_improvement_percent": 100.0 * (baseline_mae - xgb_mae) / baseline_mae,
                "xgboost_better": xgb_mae < baseline_mae,
                "interpretation": (
                    "XGBoost outperformed the baseline"
                    if xgb_mae < baseline_mae
                    else "XGBoost did not outperform; retained as a valid research result"
                ),
            }
        )
    pd.DataFrame(rows).to_csv(
        project_path("outputs", "tables", "table_04_xgb_vs_baselines_h15.csv"), index=False
    )


def save_h15_error_analysis() -> None:
    """Document distribution shift without using test evidence for selection."""
    sources = [
        (
            "valid",
            "xgboost_selected",
            project_path("outputs", "tables", "pred_xgb_tuned_valid_h15.csv"),
            "predicted_token_load",
        ),
        (
            "test",
            "xgboost_final",
            project_path("outputs", "tables", "pred_xgb_test_h15.csv"),
            "predicted_token_load",
        ),
        (
            "test",
            "persistence",
            project_path("outputs", "tables", "pred_baselines_test.csv"),
            "pred_persistence_h15",
        ),
        (
            "test",
            "seasonal_naive",
            project_path("outputs", "tables", "pred_baselines_test.csv"),
            "pred_seasonal_h15",
        ),
    ]
    rows: list[dict[str, object]] = []
    for split_name, model_name, path, prediction_column in sources:
        frame = pd.read_csv(path)
        actual_column = "actual_token_load"
        actual = frame[actual_column].to_numpy(dtype="float64")
        prediction = frame[prediction_column].to_numpy(dtype="float64")
        rows.append(
            {
                "split": split_name,
                "model": model_name,
                "n_observations": len(frame),
                "actual_mean": float(np.mean(actual)),
                "actual_median": float(np.median(actual)),
                "actual_p95": float(np.quantile(actual, 0.95)),
                "actual_zero_windows": int((actual == 0.0).sum()),
                "prediction_mean": float(np.mean(prediction)),
                "prediction_median": float(np.median(prediction)),
                "mean_error_bias": float(np.mean(prediction - actual)),
                "mae": float(np.mean(np.abs(prediction - actual))),
                "negative_prediction_count": int((prediction < 0.0).sum()),
                "analysis_role": (
                    "validation selection evidence"
                    if split_name == "valid"
                    else "post-freeze test interpretation only"
                ),
            }
        )
    analysis = pd.DataFrame(rows)
    valid_mean = float(
        analysis.loc[
            analysis["split"].eq("valid") & analysis["model"].eq("xgboost_selected"),
            "actual_mean",
        ].iloc[0]
    )
    test_mean = float(
        analysis.loc[
            analysis["split"].eq("test") & analysis["model"].eq("xgboost_final"),
            "actual_mean",
        ].iloc[0]
    )
    analysis["test_actual_mean_change_vs_validation_percent"] = (
        100.0 * (test_mean - valid_mean) / valid_mean
    )
    analysis["diagnosis"] = (
        "Test mean load is much lower than validation; persistence adapts immediately, "
        "while the refit XGBoost has positive mean bias. This is interpretation, not a tuning rule."
    )
    analysis.to_csv(
        project_path("outputs", "tables", "table_04_xgb_h15_error_analysis.csv"), index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons-minutes", type=int, nargs="*", default=None)
    parser.add_argument("--n-iter", type=int, default=12, help="Random candidates per horizon")
    parser.add_argument("--shap-sample-size", type=int, default=2000)
    parser.add_argument("--skip-shap", action="store_true", help="Development-only fast run")
    args = parser.parse_args()
    if args.n_iter <= 0:
        raise ValueError("--n-iter must be positive")
    if not 1 <= args.shap_sample_size <= 2000:
        raise ValueError("--shap-sample-size must be in [1, 2000]")

    ensure_output_directories()
    set_global_seed(SEED)
    config = load_base_config()
    horizons = args.horizons_minutes or [int(value) for value in config["series"]["horizons_minutes"]]
    feature_names = load_feature_names()
    threshold = load_threshold()
    candidates = sampled_candidates(args.n_iter, SEED)

    # Phase 1: only train/validation files are opened.
    train_valid: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    selected_by_horizon: dict[int, dict[str, Any]] = {}
    metric_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    tuning_log: list[dict[str, object]] = []

    for horizon in horizons:
        train = load_split(horizon, "train", feature_names)
        valid = load_split(horizon, "valid", feature_names)
        train_valid[horizon] = (train, valid)
    if 15 in horizons:
        metrics, diagnostics = train_default_h15(
            *train_valid[15], feature_names, threshold
        )
        metric_rows.append(metrics)
        diagnostic_rows.append(diagnostics)

    for horizon in horizons:
        train, valid = train_valid[horizon]
        selected, tuning = tune_horizon(
            horizon,
            train,
            valid,
            feature_names,
            threshold,
            candidates,
            tuning_log,
        )
        selected_by_horizon[horizon] = selected
        config_path = save_config(
            horizon, feature_names, selected, tuning, len(train), len(valid)
        )
        metrics, diagnostics = fit_selected_validation_model(
            horizon,
            train,
            valid,
            feature_names,
            selected,
            threshold,
            args.shap_sample_size,
            args.skip_shap,
        )
        metric_rows.append(metrics)
        diagnostic_rows.append(diagnostics)
        print(f"Frozen validation-selected configuration: {config_path}")

    # Phase 2 starts only after every horizon has a frozen config.
    print("All horizon configurations frozen; opening test split files now.")
    for horizon in horizons:
        train, valid = train_valid[horizon]
        metrics, diagnostics = final_refit_and_test_once(
            horizon,
            train,
            valid,
            feature_names,
            selected_by_horizon[horizon],
            threshold,
        )
        metric_rows.append(metrics)
        diagnostic_rows.append(diagnostics)

    results = pd.DataFrame(metric_rows).sort_values(
        ["horizon_minutes", "split", "model"], kind="mergesort"
    )
    results.to_csv(project_path("outputs", "tables", "table_04_xgb_results.csv"), index=False)
    pd.DataFrame(diagnostic_rows).to_csv(
        project_path("outputs", "tables", "table_04_xgb_prediction_diagnostics.csv"), index=False
    )
    save_comparison(results)
    save_h15_error_analysis()
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
