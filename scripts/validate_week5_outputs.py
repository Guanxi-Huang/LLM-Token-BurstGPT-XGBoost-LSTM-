"""Independently validate the frozen Week-5 tables, figures, and experiment log."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
MPL_CONFIG_DIR = PROJECT_ROOT / "outputs" / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib.image as mpimg
import numpy as np
import pandas as pd


MODELS = ("Persistence", "Seasonal Naive", "XGBoost", "LSTM")
HORIZONS = (5, 15, 60)


def load_module(alias: str, filename: str) -> ModuleType:
    path = PROJECT_ROOT / "src" / filename
    specification = importlib.util.spec_from_file_location(alias, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[alias] = module
    specification.loader.exec_module(module)
    return module


def require(path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        raise AssertionError(f"Missing or empty Week-5 artifact: {path}")
    return path


def close_to_reported(actual: float, reported: float) -> bool:
    # Final CSVs intentionally contain three significant digits.
    return bool(np.isclose(actual, reported, rtol=0.005, atol=5e-6))


def main() -> None:
    evaluate = load_module("evaluate_week5_validation", "07_evaluate.py")
    threshold = float(
        pd.read_csv(require(TABLE_DIR / "table_01_burst_threshold.csv")).loc[
            0, "p95_threshold"
        ]
    )
    checks: list[dict[str, object]] = []

    predictions = pd.read_csv(
        require(TABLE_DIR / "test_predictions_all_models.csv"),
        parse_dates=["timestamp"],
    )
    expected_columns = [
        "timestamp",
        "horizon",
        "model",
        "y_true",
        "y_pred",
        "actual_burst",
        "predicted_burst",
    ]
    assert predictions.columns.tolist() == expected_columns
    assert len(predictions) == 5_228 * len(HORIZONS) * len(MODELS)
    assert set(predictions["horizon"]) == set(HORIZONS)
    assert set(predictions["model"]) == set(MODELS)
    assert not predictions.duplicated(["timestamp", "horizon", "model"]).any()
    coverage = predictions.groupby(["timestamp", "horizon"]).agg(
        rows=("model", "size"),
        models=("model", "nunique"),
        actual_values=("y_true", "nunique"),
        actual_labels=("actual_burst", "nunique"),
    )
    assert coverage["rows"].eq(4).all() and coverage["models"].eq(4).all()
    assert coverage[["actual_values", "actual_labels"]].eq(1).all().all()
    assert predictions["actual_burst"].eq(predictions["y_true"].ge(threshold)).all()
    assert predictions["predicted_burst"].eq(predictions["y_pred"].ge(threshold)).all()
    checks.append(
        {
            "check": "unified_prediction_grain_and_labels",
            "status": "passed",
            "evidence": f"{len(predictions)} rows; 4 models per timestamp+horizon",
        }
    )

    main_results = pd.read_csv(require(TABLE_DIR / "table_04_main_results.csv"))
    burst_results = pd.read_csv(require(TABLE_DIR / "table_05_burst_results.csv"))
    assert main_results["horizon_minutes"].eq(15).all()
    assert burst_results["horizon_minutes"].eq(15).all()
    assert main_results["model"].tolist() == list(MODELS)
    assert burst_results["model"].tolist() == list(MODELS)
    for model in MODELS:
        part = predictions.loc[
            predictions["horizon"].eq(15) & predictions["model"].eq(model)
        ]
        regression = evaluate.regression_metrics(part["y_true"], part["y_pred"])
        burst = evaluate.burst_metrics(part["y_true"], part["y_pred"], threshold)
        reported_regression = main_results.loc[main_results["model"].eq(model)].iloc[0]
        reported_burst = burst_results.loc[burst_results["model"].eq(model)].iloc[0]
        for metric in ("mae", "rmse", "smape_percent"):
            assert close_to_reported(float(regression[metric]), float(reported_regression[metric]))
        for metric in ("precision", "recall", "f1", "pr_auc"):
            assert close_to_reported(float(burst[metric]), float(reported_burst[metric]))
        for source, target in (
            ("true_positive", "true_positive"),
            ("false_positive", "false_positive"),
            ("false_negative", "false_negative"),
            ("true_negative", "true_negative"),
        ):
            assert int(burst[source]) == int(reported_burst[target])
    checks.append(
        {
            "check": "independent_main_metric_recalculation",
            "status": "passed",
            "evidence": "MAE/RMSE/sMAPE/Precision/Recall/F1/AP/confusion counts agree",
        }
    )

    appendix_main = pd.read_csv(require(TABLE_DIR / "table_04a_appendix_results_h5_h60.csv"))
    appendix_burst = pd.read_csv(require(TABLE_DIR / "table_05a_appendix_burst_results_h5_h60.csv"))
    assert appendix_main["horizon_minutes"].tolist() == [5] * 4 + [60] * 4
    assert appendix_burst["horizon_minutes"].tolist() == [5] * 4 + [60] * 4
    checks.append(
        {
            "check": "main_and_appendix_horizon_separation",
            "status": "passed",
            "evidence": "h15 in main tables; h5/h60 in appendix tables",
        }
    )

    segments = pd.read_csv(require(TABLE_DIR / "table_06_segment_errors.csv"))
    expected_dimensions = {
        "actual_burst",
        "relative_day_phase",
        "relative_week_phase",
        "gpt4_share",
        "api_share",
    }
    assert set(segments["dimension"]) == expected_dimensions
    assert segments["horizon_minutes"].eq(15).all()
    assert not segments["segment"].str.lower().str.contains("weekday|weekend").any()
    composition = segments.loc[segments["dimension"].isin(["gpt4_share", "api_share"])]
    assert composition["target_time_composition_role"].str.contains(
        "diagnostic grouping only", regex=False
    ).all()
    checks.append(
        {
            "check": "predefined_segment_scope_and_leakage_note",
            "status": "passed",
            "evidence": "exactly five dimensions; composition marked diagnostic-only",
        }
    )

    ablation = pd.read_csv(require(TABLE_DIR / "table_06a_xgb_h15_ablation.csv"))
    assert set(ablation["feature_set"]) == {
        "full_features",
        "without_service_structure",
        "lags_plus_relative_calendar",
    }
    feature_counts = (
        ablation.loc[ablation["split"].eq("valid")]
        .set_index("feature_set")["n_features"]
        .to_dict()
    )
    assert feature_counts == {
        "full_features": 36,
        "without_service_structure": 30,
        "lags_plus_relative_calendar": 12,
    }
    assert ablation["validation_selected_feature_set"].eq("full_features").all()
    assert int(ablation["test_prediction_calls_in_week5_ablation"].sum()) == 2
    checks.append(
        {
            "check": "minimal_ablation_boundary",
            "status": "passed",
            "evidence": "3 fixed sets; validation chose full; only 2 new test calls",
        }
    )

    robust = pd.read_csv(require(TABLE_DIR / "table_07_robustness_burstgpt3.csv"))
    assert len(robust) == len(HORIZONS) * len(MODELS)
    assert set(robust["horizon_minutes"]) == set(HORIZONS)
    assert set(robust["model"]) == set(MODELS)
    assert robust["design"].eq("external zero-shot cross-period test").all()
    assert robust["batch3_fit_or_selection_rows"].eq(0).all()
    assert np.isclose(robust["train_1_2_p95_threshold"], threshold, rtol=0.005).all()
    series3 = pd.read_parquet(require(PROJECT_ROOT / "data" / "processed" / "series_5min_burstgpt3.parquet"))
    assert "actual_burst" in series3 and "split" in series3
    assert series3["split"].eq("robustness_zero_shot").all()
    assert series3["actual_burst"].eq(series3["token_load"].ge(threshold)).all()
    checks.append(
        {
            "check": "burstgpt3_zero_shot_design",
            "status": "passed",
            "evidence": "12 model-horizon rows; 0 batch-3 fit/selection rows; main P95 retained",
        }
    )

    figure_names = [
        "fig_11_test_week_forecasts_h15.png",
        "fig_12_burst_window_zoom_h15.png",
        "fig_13_model_mae_f1_h15.png",
        "fig_14_pr_curves_h15.png",
        "fig_15_burst_confusion_matrices_h15.png",
        "fig_16_segment_error_burst_h15.png",
        "fig_17_segment_error_phases_h15.png",
        "fig_18_segment_error_composition_h15.png",
    ]
    for name in figure_names:
        image = mpimg.imread(require(FIGURE_DIR / name))
        assert image.ndim in (2, 3) and image.shape[0] >= 800 and image.shape[1] >= 1200
        assert np.isfinite(image).all()
    checks.append(
        {
            "check": "figure_render_integrity",
            "status": "passed",
            "evidence": "8 readable high-resolution PNGs",
        }
    )

    log = require(PROJECT_ROOT / "docs" / "experiment_log.md").read_text(encoding="utf-8")
    for phrase in (
        "WEEK5_FINAL_EVALUATION_START",
        "WEEK5_ABLATION_ROBUSTNESS_START",
        "测试使用次数",
        "外部零样本跨时期测试",
        "第5周在此停止调参与阈值修改",
    ):
        assert phrase in log
    checks.append(
        {
            "check": "experiment_log_freeze_and_test_use",
            "status": "passed",
            "evidence": "freeze, test-use count, figure conclusions, and robustness design recorded",
        }
    )

    output = pd.DataFrame(checks)
    output.to_csv(TABLE_DIR / "table_08_week5_validation.csv", index=False)
    print(output.to_string(index=False))
    print("\nWEEK-5 VALIDATION: READY TO SHARE WITH THE DOCUMENTED FIVE-BURST CAVEAT")


if __name__ == "__main__":
    main()
