"""Validate the common no-leakage timing contract across all model families.

The canonical timestamp is the *left edge* of a 5-minute aggregation window.
For a forecast origin ``t``, model inputs may therefore end no later than the
window stamped ``t-5min``; labels are stamped ``t+h`` and split membership is
assigned by that target timestamp.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
HORIZONS = (5, 15, 60)
SPLITS = ("train", "valid", "test")
FREQUENCY = pd.Timedelta(minutes=5)


def load_numeric_module(alias: str, filename: str):
    path = SRC / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def require(path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        raise AssertionError(f"Missing or empty artifact: {path}")
    return path


def validate_target_time_splits(series: pd.DataFrame) -> None:
    for horizon in HORIZONS:
        previous_end = None
        for split in SPLITS:
            frame = pd.read_parquet(
                require(ROOT / f"data/processed/features_h{horizon}_{split}.parquet"),
                columns=["feature_time", "target_time", "split", f"target_h{horizon}"],
            )
            origin = pd.DatetimeIndex(pd.to_datetime(frame["feature_time"], utc=True))
            target_time = pd.DatetimeIndex(pd.to_datetime(frame["target_time"], utc=True))
            assert (target_time - origin == pd.Timedelta(minutes=horizon)).all()
            assert frame["split"].eq(split).all()
            expected_split = series["split"].reindex(target_time).astype(str).to_numpy()
            np.testing.assert_array_equal(frame["split"].astype(str), expected_split)
            if previous_end is not None:
                assert previous_end < target_time.min()
            previous_end = target_time.max()


def validate_baselines(series: pd.DataFrame) -> None:
    predictions = pd.read_csv(
        require(ROOT / "outputs/tables/pred_baselines_test.csv"),
        parse_dates=["timestamp"],
    )
    target_time = pd.DatetimeIndex(pd.to_datetime(predictions["timestamp"], utc=True))
    for horizon in HORIZONS:
        origin = pd.DatetimeIndex(
            pd.to_datetime(predictions[f"forecast_origin_h{horizon}"], utc=True)
        )
        input_end = pd.DatetimeIndex(
            pd.to_datetime(predictions[f"last_available_window_h{horizon}"], utc=True)
        )
        assert (target_time - origin == pd.Timedelta(minutes=horizon)).all()
        assert (origin - input_end == FREQUENCY).all()
        expected = series["token_load"].reindex(input_end).to_numpy(dtype="float64")
        np.testing.assert_array_equal(predictions[f"pred_persistence_h{horizon}"], expected)


def validate_lstm(series: pd.DataFrame) -> None:
    lstm = load_numeric_module("lstm_temporal_validation", "06_lstm.py")
    for horizon in HORIZONS:
        prediction = pd.read_csv(
            require(ROOT / f"outputs/tables/pred_lstm_test_h{horizon}.csv"),
            parse_dates=["feature_time", "input_end_time", "target_time"],
        )
        origin = pd.DatetimeIndex(pd.to_datetime(prediction["feature_time"], utc=True))
        input_end = pd.DatetimeIndex(pd.to_datetime(prediction["input_end_time"], utc=True))
        target_time = pd.DatetimeIndex(pd.to_datetime(prediction["target_time"], utc=True))
        assert (origin - input_end == FREQUENCY).all()
        assert (target_time - origin == pd.Timedelta(minutes=horizon)).all()

        with require(ROOT / f"configs/lstm_h{horizon}.yaml").open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        assert config["temporal_contract"]["latest_input_window"].startswith("t-5 minutes")
        anchor = lstm.load_split_anchor(horizon, "test")
        scaler_x = joblib.load(require(ROOT / f"models/lstm_feature_scaler_h{horizon}.joblib"))
        scaler_y = joblib.load(require(ROOT / f"models/lstm_target_scaler_h{horizon}.joblib"))
        features = lstm.build_lstm_features(series)
        sequence = lstm.prepare_sequence_set(
            features,
            anchor,
            "test",
            list(config["features"]),
            int(config["lookback"]),
            horizon // 5,
            scaler_x,
            scaler_y,
            fit_scalers=False,
        )
        assert sequence.feature_time.equals(origin)
        assert sequence.input_end_time.equals(input_end)
        np.testing.assert_array_equal(sequence.y_raw, prediction["actual_token_load"])


def validate_training_only_state(series: pd.DataFrame) -> None:
    threshold = float(
        pd.read_csv(require(ROOT / "outputs/tables/table_01_burst_threshold.csv")).loc[
            0, "p95_threshold"
        ]
    )
    expected = float(np.quantile(series.loc[series["split"].eq("train"), "token_load"], 0.95))
    assert np.isclose(threshold, expected)
    for stem in ("xgb_tuning_log.csv", "lstm_tuning_log.csv"):
        tuning = pd.read_csv(require(ROOT / "outputs/tables" / stem))
        assert not any("test" in column.lower() for column in tuning.columns)
    for horizon in HORIZONS:
        with require(ROOT / f"configs/xgb_h{horizon}.yaml").open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        assert str(config["selection"]["data"]).startswith("validation only")


def validate_batch3_if_present() -> None:
    prediction_path = ROOT / "outputs/tables/burstgpt3_predictions_all_models.csv"
    series_path = ROOT / "data/processed/series_5min_burstgpt3.parquet"
    if not prediction_path.exists() or not series_path.exists():
        return
    predictions = pd.read_csv(prediction_path, parse_dates=["timestamp"])
    series = pd.read_parquet(series_path)
    series.index = pd.DatetimeIndex(pd.to_datetime(series.index, utc=True))
    for horizon in HORIZONS:
        part = predictions.loc[
            predictions["horizon"].eq(horizon) & predictions["model"].eq("Persistence")
        ]
        target_time = pd.DatetimeIndex(pd.to_datetime(part["timestamp"], utc=True))
        expected = series["token_load"].reindex(
            target_time - pd.Timedelta(minutes=horizon + 5)
        )
        np.testing.assert_array_equal(part["y_pred"], expected.to_numpy(dtype="float64"))


def main() -> None:
    series = pd.read_parquet(require(ROOT / "data/processed/series_5min.parquet"))
    series.index = pd.DatetimeIndex(pd.to_datetime(series.index, utc=True))
    validate_target_time_splits(series)
    validate_baselines(series)
    validate_lstm(series)
    validate_training_only_state(series)
    validate_batch3_if_present()
    print(
        "Temporal contract passed: target-time splits, completed-window inputs, "
        "training-only threshold/scalers, and validation-only tuning are consistent."
    )


if __name__ == "__main__":
    main()
