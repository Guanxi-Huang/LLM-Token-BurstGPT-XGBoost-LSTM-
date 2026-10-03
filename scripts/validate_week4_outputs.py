"""Validate the corrected Week-4 LSTM artifacts before reporting them."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
MODELS = ROOT / "models"
HORIZONS = (5, 15, 60)
SPLITS = ("train", "valid", "test")
EPSILON = 1e-8


def require(path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        raise AssertionError(f"Missing or empty artifact: {path}")
    return path


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "smape_percent": float(
            100.0 * np.mean(2.0 * np.abs(error) / (np.abs(actual) + np.abs(predicted) + EPSILON))
        ),
    }


def confusion(actual: np.ndarray, predicted: np.ndarray) -> dict[str, int | float]:
    tn = int(np.sum(~actual & ~predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    tp = int(np.sum(actual & predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def validate_configs_and_models() -> None:
    import tensorflow as tf

    expected_features = [
        "token_load", "request_count", "mean_request_tokens", "mean_response_tokens",
        "gpt4_share", "api_share", "relative_day_sin", "relative_day_cos",
        "relative_week_sin", "relative_week_cos",
    ]
    for horizon in HORIZONS:
        config_path = require(ROOT / "configs" / f"lstm_h{horizon}.yaml")
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        assert config["horizon_minutes"] == horizon
        assert config["lookback"] == 144
        assert config["features"] == expected_features
        assert config["network"]["parameter_count"] == 5537
        assert config["training"]["shuffle"] is False
        assert config["selection"]["formal_runs"] == 1
        assert str(config["selection"]["data"]).startswith("validation only")
        assert config["temporal_contract"]["latest_input_window"].startswith("t-5 minutes")
        feature_scaler = joblib.load(require(MODELS / f"lstm_feature_scaler_h{horizon}.joblib"))
        target_scaler = joblib.load(require(MODELS / f"lstm_target_scaler_h{horizon}.joblib"))
        assert int(feature_scaler.n_features_in_) == 10
        assert int(target_scaler.n_features_in_) == 1
        summary = require(MODELS / f"lstm_h{horizon}_summary.txt").read_text(encoding="utf-8")
        assert "Architecture parameters (model.count_params): 5,537" in summary
        model = tf.keras.models.load_model(require(MODELS / f"lstm_h{horizon}.keras"))
        assert int(model.count_params()) == 5537


def validate_selection_and_histories() -> None:
    tuning = pd.read_csv(require(TABLES / "lstm_tuning_log.csv"))
    assert set(tuning["horizon_minutes"].astype(int)) == set(HORIZONS)
    assert len(tuning) == 3
    assert tuning["selected"].astype(bool).all()
    assert tuning["selection_data"].eq("validation_only").all()
    assert tuning["parameter_count"].eq(5537).all()
    assert tuning["lookback"].eq(144).all()
    assert np.isfinite(tuning[["train_mae_tokens", "valid_mae_tokens", "valid_rmse_tokens", "valid_f1"]]).all().all()
    for row in tuning.itertuples():
        history = pd.read_csv(require(TABLES / f"lstm_history_h{int(row.horizon_minutes)}.csv"))
        assert history["epoch"].iloc[0] == 1
        assert history["epoch"].is_monotonic_increasing
        assert int(row.best_epoch) in set(history["epoch"].astype(int))
        assert len(history) == int(row.epochs_completed)
        assert np.isfinite(history.select_dtypes(include=[np.number])).all().all()


def validate_sequence_audits() -> None:
    audit = pd.read_csv(require(TABLES / "table_05_lstm_sequence_audit.csv"))
    manual = pd.read_csv(require(TABLES / "table_05_lstm_manual_sequence_check.csv"))
    assert len(audit) == 9 and len(manual) == 9
    assert set(zip(audit["horizon_minutes"].astype(int), audit["split"])) == {
        (h, split) for h in HORIZONS for split in SPLITS
    }
    assert audit["lookback_steps"].eq(144).all()
    assert audit["n_features"].eq(10).all()
    assert audit["labels_confined_to_split"].astype(bool).all()
    assert manual["time_alignment_passed"].astype(bool).all()
    input_end = pd.to_datetime(manual["input_end_time"], utc=True)
    origin = pd.to_datetime(manual["forecast_origin_time"], utc=True)
    label = pd.to_datetime(manual["label_time"], utc=True)
    assert (origin - input_end).eq(pd.Timedelta(minutes=5)).all()
    expected_horizon = pd.to_timedelta(manual["horizon_minutes"], unit="m")
    assert (label - origin).eq(expected_horizon).all()


def validate_predictions_and_metrics() -> None:
    results = pd.read_csv(require(TABLES / "table_05_lstm_results.csv"))
    assert len(results) == 3
    assert results["split"].eq("test").all()
    assert results["model"].eq("lstm_final").all()
    assert set(results["horizon_minutes"].astype(int)) == set(HORIZONS)
    assert results["n_observations"].eq(5228).all()
    assert results["negative_predictions_after_clip"].eq(0).all()
    assert results["target_time_aligned"].astype(bool).all()
    threshold_column = "burst_threshold" if "burst_threshold" in results else "threshold"
    for horizon in HORIZONS:
        prediction = pd.read_csv(
            require(TABLES / f"pred_lstm_test_h{horizon}.csv"),
            parse_dates=["feature_time", "input_end_time", "target_time"],
        )
        assert len(prediction) == 5228
        assert prediction["horizon_minutes"].eq(horizon).all()
        assert (prediction["feature_time"] - prediction["input_end_time"]).eq(pd.Timedelta(minutes=5)).all()
        assert (prediction["target_time"] - prediction["feature_time"]).eq(pd.Timedelta(minutes=horizon)).all()
        actual = prediction["actual_token_load"].to_numpy(dtype="float64")
        predicted = prediction["predicted_token_load"].to_numpy(dtype="float64")
        assert np.isfinite(actual).all() and np.isfinite(predicted).all()
        assert np.min(predicted) >= 0
        assert np.unique(predicted).size > 1
        row = results.loc[results["horizon_minutes"].eq(horizon)].iloc[0]
        metrics = regression_metrics(actual, predicted)
        for name, value in metrics.items():
            # Metrics were computed from in-memory float32 predictions before
            # the prediction CSV decimal round-trip.  A 1e-3 absolute token
            # tolerance is still far below any reportable precision.
            assert np.isclose(float(row[name]), value, rtol=0, atol=1e-3), (horizon, name)
        threshold = float(row[threshold_column])
        classification = confusion(actual >= threshold, predicted >= threshold)
        for name, value in classification.items():
            assert np.isclose(float(row[name]), value, rtol=0, atol=1e-12), (horizon, name)
        np.testing.assert_array_equal(prediction["actual_burst"].astype(bool), actual >= threshold)
        np.testing.assert_array_equal(prediction["predicted_burst"].astype(bool), predicted >= threshold)


def validate_training_figures() -> None:
    source = (ROOT / "src" / "06_lstm.py").read_text(encoding="utf-8")
    for prohibited in ("plt.title(", ".set_title(", ".suptitle(", "fig.text("):
        assert prohibited not in source, f"Prohibited title API found: {prohibited}"
    for horizon in HORIZONS:
        path = require(ROOT / "outputs" / "figures" / f"fig_lstm_training_h{horizon}.png")
        with Image.open(path) as image:
            assert image.width >= 2000 and image.height >= 1200
            dpi = image.info.get("dpi")
            assert dpi is not None and min(dpi) >= 299.0, (horizon, dpi)


def main() -> None:
    validate_configs_and_models()
    validate_selection_and_histories()
    validate_sequence_audits()
    validate_predictions_and_metrics()
    validate_training_figures()
    print("Week-4 validation passed: configs, scalers, histories, temporal audits, predictions, metrics, and title-free 300-DPI figures are internally consistent.")


if __name__ == "__main__":
    main()
