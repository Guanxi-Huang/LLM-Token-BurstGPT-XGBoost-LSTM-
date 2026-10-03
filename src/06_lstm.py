"""Train conservative, leakage-safe single-layer LSTM load forecasts.

The implementation deliberately mirrors the XGBoost experiment protocol:

1. target membership and forecast timestamps come from the frozen XGBoost
   split files;
2. feature and target scalers are fitted on training observations only;
3. model selection sees train/validation data only;
4. every selected configuration is written to YAML before any test split is
   opened; and
5. the final refit makes exactly one prediction call on each test split.

The default run is intentionally small: one 32-unit LSTM, a 144-step (12-hour)
lookback, and dropout=0.1.  ``--max-runs`` can expose at most four pre-declared
validation candidates without expanding the network family.
"""

from __future__ import annotations

import argparse
import gc
import io
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / "outputs" / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from utils import ensure_output_directories, project_path, set_global_seed


SEED = 42
FREQUENCY_MINUTES = 5
DEFAULT_LOOKBACK = 144
DEFAULT_UNITS = 32
DEFAULT_DROPOUT = 0.1
MAX_EPOCHS = 100
BATCH_SIZE = 64
PATIENCE = 10
EPSILON = 1e-9
SPLIT_ORDER = ("train", "valid", "test")
RELATIVE_TIME_ORIGIN = pd.Timestamp("1970-01-01", tz="UTC")

MAIN_FEATURES = (
    "token_load",
    "request_count",
    "mean_request_tokens",
    "mean_response_tokens",
    "gpt4_share",
    "api_share",
    "relative_day_sin",
    "relative_day_cos",
    "relative_week_sin",
    "relative_week_cos",
)
HISTORY_LOAD_FEATURES = ("token_load",)

# The order starts with the required conservative baseline.  The remaining
# candidates change one bounded axis at a time; no other architectures are
# reachable from this script.
BOUNDED_CANDIDATES = (
    {"units": 32, "lookback": 144, "dropout": 0.1},
    {"units": 64, "lookback": 144, "dropout": 0.1},
    {"units": 32, "lookback": 288, "dropout": 0.1},
    {"units": 32, "lookback": 144, "dropout": 0.0},
)


@dataclass
class SequenceSet:
    """One target-time split after scaling and window construction."""

    split: str
    X: np.ndarray
    y_scaled: np.ndarray
    y_raw: np.ndarray
    feature_time: pd.DatetimeIndex
    target_time: pd.DatetimeIndex
    input_start_time: pd.DatetimeIndex
    input_end_time: pd.DatetimeIndex
    feature_fit_rows: int = 0


@dataclass
class SelectionResult:
    """Frozen validation-selected architecture and its evidence."""

    spec: dict[str, float | int]
    best_epoch: int
    history: dict[str, list[float]]
    tuning: pd.DataFrame
    parameter_count: int
    train_mae: float
    valid_mae: float
    valid_f1: float
    persistent_generalization_gap: bool


def load_base_config() -> dict[str, Any]:
    with project_path("configs", "base.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def artifact_suffix(feature_set: str) -> str:
    return "" if feature_set == "main" else "_history_load"


def feature_names_for(feature_set: str) -> list[str]:
    if feature_set == "main":
        return list(MAIN_FEATURES)
    if feature_set == "history_load":
        return list(HISTORY_LOAD_FEATURES)
    raise ValueError(f"Unknown feature set: {feature_set}")


def validate_horizon(horizon: int) -> int:
    if horizon <= 0 or horizon % FREQUENCY_MINUTES:
        raise ValueError(f"Horizon must be a positive multiple of 5 minutes: {horizon}")
    return horizon // FREQUENCY_MINUTES


def load_split_anchor(horizon: int, split: str) -> pd.DataFrame:
    """Load only timestamps and labels from one frozen XGBoost split file."""
    if split not in SPLIT_ORDER:
        raise ValueError(f"Unknown split: {split}")
    path = project_path("data", "processed", f"features_h{horizon}_{split}.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}; run src/03_features.py")
    target_name = f"target_h{horizon}"
    frame = pd.read_parquet(
        path,
        columns=["feature_time", "target_time", "split", target_name, "actual_burst"],
    )
    frame["feature_time"] = pd.to_datetime(frame["feature_time"], utc=True)
    frame["target_time"] = pd.to_datetime(frame["target_time"], utc=True)
    frame = frame.sort_values("target_time", kind="mergesort").reset_index(drop=True)
    if frame.empty or not frame["split"].eq(split).all():
        raise AssertionError(f"Invalid or empty {split} anchor for h{horizon}")
    expected_delta = pd.Timedelta(minutes=horizon)
    if not (frame["target_time"] - frame["feature_time"]).eq(expected_delta).all():
        raise AssertionError(f"Feature/target time misalignment in {path}")
    for column in ("feature_time", "target_time"):
        if frame[column].duplicated().any():
            raise AssertionError(f"Duplicate {column} in {path}")
        if not frame[column].diff().dropna().eq(pd.Timedelta(minutes=5)).all():
            raise AssertionError(f"Non-contiguous {column} in {path}")
    target_values = frame[target_name].to_numpy(dtype="float64")
    if not np.isfinite(target_values).all():
        raise AssertionError(f"Non-finite target values in {path}")
    return frame


def load_series_through(cutoff: pd.Timestamp) -> pd.DataFrame:
    """Read only rows at or before ``cutoff`` from the canonical parquet file."""
    path = project_path("data", "processed", "series_5min.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Missing canonical series: {path}; run src/02_build_series.py")
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    frame = pd.read_parquet(path, filters=[("timestamp", "<=", cutoff.to_pydatetime())])
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True), name="timestamp")
    frame = frame.sort_index(kind="mergesort")
    required = {
        "token_load",
        "request_count",
        "mean_request_tokens",
        "mean_response_tokens",
        "gpt4_request_count",
        "api_request_count",
        "split",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Canonical series is missing columns: {sorted(missing)}")
    if frame.empty or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise AssertionError("Canonical series must have a unique increasing index")
    if not frame.index.to_series().diff().dropna().eq(pd.Timedelta(minutes=5)).all():
        raise AssertionError("Canonical series must be a complete 5-minute grid")
    if frame.index.max() != cutoff:
        raise AssertionError(f"Series cutoff mismatch: expected {cutoff}, got {frame.index.max()}")
    return frame


def build_lstm_features(series: pd.DataFrame) -> pd.DataFrame:
    """Create the minimal contemporaneously observed LSTM input matrix."""
    request_count = series["request_count"].to_numpy(dtype="float64")
    denominator = np.where(request_count > 0.0, request_count, np.nan)
    # Preserve the published relative-second phase when a frozen main model is
    # applied to a later batch instead of resetting phase at that batch's first
    # request.  The primary 1/2 series already begins exactly at this origin.
    elapsed_seconds = (series.index - RELATIVE_TIME_ORIGIN).total_seconds().to_numpy()
    day_angle = 2.0 * np.pi * (elapsed_seconds % 86_400.0) / 86_400.0
    week_angle = 2.0 * np.pi * (elapsed_seconds % 604_800.0) / 604_800.0

    features = pd.DataFrame(
        {
            "token_load": series["token_load"].to_numpy(dtype="float64"),
            "request_count": request_count,
            "mean_request_tokens": series["mean_request_tokens"].fillna(0.0).to_numpy(dtype="float64"),
            "mean_response_tokens": series["mean_response_tokens"].fillna(0.0).to_numpy(dtype="float64"),
            "gpt4_share": np.nan_to_num(
                series["gpt4_request_count"].to_numpy(dtype="float64") / denominator,
                nan=0.0,
            ),
            "api_share": np.nan_to_num(
                series["api_request_count"].to_numpy(dtype="float64") / denominator,
                nan=0.0,
            ),
            "relative_day_sin": np.sin(day_angle),
            "relative_day_cos": np.cos(day_angle),
            "relative_week_sin": np.sin(week_angle),
            "relative_week_cos": np.cos(week_angle),
        },
        index=series.index,
    )
    if not np.isfinite(features.to_numpy()).all():
        raise AssertionError("LSTM feature matrix contains non-finite values")
    for share in ("gpt4_share", "api_share"):
        if not features[share].between(0.0, 1.0).all():
            raise AssertionError(f"{share} must be in [0, 1]")
    return features


def make_sequences(
    features: np.ndarray,
    target: np.ndarray,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``[t-lookback+1, ..., t]`` inputs and ``t+horizon`` labels.

    Parameters use rows/steps rather than minutes.  For the canonical 5-minute
    series, ``lookback=144`` means 12 hours and ``horizon=3`` means 15 minutes.
    """
    feature_array = np.asarray(features)
    target_array = np.asarray(target).reshape(-1)
    if feature_array.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {feature_array.shape}")
    if len(feature_array) != len(target_array):
        raise ValueError("features and target must have the same number of rows")
    if lookback <= 0 or horizon <= 0:
        raise ValueError("lookback and horizon must be positive")
    sample_count = len(feature_array) - lookback - horizon + 1
    if sample_count <= 0:
        raise ValueError(
            f"Need at least lookback+horizon rows, got {len(feature_array)} "
            f"for lookback={lookback}, horizon={horizon}"
        )

    # NumPy places the sliding axis last for axis=0; move it to the middle to
    # obtain Keras' (samples, timesteps, features) convention.
    windows = np.lib.stride_tricks.sliding_window_view(
        feature_array, window_shape=lookback, axis=0
    )
    X = np.moveaxis(windows[:sample_count], -1, 1).copy()
    first_label = lookback - 1 + horizon
    y = target_array[first_label : first_label + sample_count].copy()
    expected_shape = (sample_count, lookback, feature_array.shape[1])
    if X.shape != expected_shape or y.shape != (sample_count,):
        raise AssertionError(f"Unexpected sequence shapes: X={X.shape}, y={y.shape}")
    return X, y


def _block_for_anchor(
    full_frame: pd.DataFrame,
    anchor: pd.DataFrame,
    lookback: int,
    horizon_steps: int,
) -> pd.DataFrame:
    first_origin = anchor["feature_time"].iloc[0]
    last_target = anchor["target_time"].iloc[-1]
    # ``feature_time`` is the forecast origin.  The window stamped at that
    # origin has only just started, so the LSTM must end one complete bin back.
    first_input = first_origin - pd.Timedelta(minutes=FREQUENCY_MINUTES * lookback)
    block = full_frame.loc[first_input:last_target]
    expected_rows = len(anchor) + lookback + horizon_steps
    if len(block) != expected_rows:
        raise AssertionError(
            f"Sequence block has {len(block)} rows, expected {expected_rows}; "
            f"range={first_input}..{last_target}"
        )
    return block


def prepare_sequence_set(
    full_frame: pd.DataFrame,
    anchor: pd.DataFrame,
    split: str,
    feature_names: list[str],
    lookback: int,
    horizon_steps: int,
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
    *,
    fit_scalers: bool,
) -> SequenceSet:
    """Scale and window one split while allowing only historical context rows."""
    block = _block_for_anchor(full_frame, anchor, lookback, horizon_steps)
    target_name = next(column for column in anchor if column.startswith("target_h"))
    feature_fit_rows = 0
    if fit_scalers:
        # Fit on unique raw observations used by training windows, not on the
        # duplicated 3D window tensor and never on validation/test rows.
        fit_end = anchor["feature_time"].iloc[-1] - pd.Timedelta(
            minutes=FREQUENCY_MINUTES
        )
        fit_frame = block.loc[:fit_end, feature_names]
        feature_scaler.fit(fit_frame.to_numpy(dtype="float64"))
        target_scaler.fit(anchor[[target_name]].to_numpy(dtype="float64"))
        feature_fit_rows = len(fit_frame)
    elif not hasattr(feature_scaler, "mean_") or not hasattr(target_scaler, "mean_"):
        raise AssertionError("Scalers must be fitted on training data before transform")

    scaled_features = feature_scaler.transform(
        block[feature_names].to_numpy(dtype="float64")
    ).astype("float32")
    scaled_target = target_scaler.transform(
        block[["token_load"]].to_numpy(dtype="float64")
    ).reshape(-1).astype("float32")
    X, y_scaled = make_sequences(
        scaled_features,
        scaled_target,
        lookback=lookback,
        # One extra row separates the last completed input window from the
        # forecast origin; the target remains exactly ``origin + horizon``.
        horizon=horizon_steps + 1,
    )

    first_origin_offset = lookback
    feature_time = block.index[
        first_origin_offset : first_origin_offset + len(X)
    ]
    label_offset = first_origin_offset + horizon_steps
    target_time = block.index[label_offset : label_offset + len(X)]
    input_start_time = block.index[: len(X)]
    input_end_time = block.index[
        first_origin_offset - 1 : first_origin_offset - 1 + len(X)
    ]
    expected_feature_time = pd.DatetimeIndex(anchor["feature_time"])
    expected_target_time = pd.DatetimeIndex(anchor["target_time"])
    if not feature_time.equals(expected_feature_time):
        raise AssertionError(f"{split} sequence origins do not match XGBoost origins")
    if not target_time.equals(expected_target_time):
        raise AssertionError(f"{split} labels do not match XGBoost target times")

    y_raw = block["token_load"].iloc[label_offset : label_offset + len(X)].to_numpy(
        dtype="float64"
    )
    np.testing.assert_array_equal(y_raw, anchor[target_name].to_numpy(dtype="float64"))
    if not (feature_time < target_time).all():
        raise AssertionError(f"{split} contains non-causal feature/target timestamps")
    expected_availability_gap = pd.Timedelta(minutes=FREQUENCY_MINUTES)
    if not np.asarray(feature_time - input_end_time == expected_availability_gap).all():
        raise AssertionError(f"{split} input window is not complete before forecast origin")
    expected_input_delta = pd.Timedelta(minutes=FREQUENCY_MINUTES * (lookback - 1))
    if not np.asarray(input_end_time - input_start_time == expected_input_delta).all():
        raise AssertionError(f"{split} lookback timestamp alignment failed")

    return SequenceSet(
        split=split,
        X=X,
        y_scaled=y_scaled,
        y_raw=y_raw,
        feature_time=feature_time,
        target_time=target_time,
        input_start_time=input_start_time,
        input_end_time=input_end_time,
        feature_fit_rows=feature_fit_rows,
    )


def import_tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for training. Install the pinned project "
            "dependencies with `.venv\\Scripts\\python.exe -m pip install -r requirements.txt`."
        ) from exc
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    return tf


def reset_training_state(tf: Any, seed: int = SEED) -> None:
    tf.keras.backend.clear_session()
    set_global_seed(seed)


def build_model(
    tf: Any,
    lookback: int,
    n_features: int,
    units: int,
    dropout: float,
) -> Any:
    if units not in (32, 64) or dropout not in (0.0, 0.1) or lookback not in (144, 288):
        raise ValueError("Architecture is outside the bounded LSTM search space")
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(lookback, n_features), name="input_history"),
            tf.keras.layers.LSTM(units, dropout=dropout, name="lstm"),
            tf.keras.layers.Dense(1, name="token_load"),
        ],
        name="conservative_lstm",
    )
    model.compile(
        optimizer="adam",
        loss="mae",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(y_true, dtype="float64")
    predicted = np.asarray(y_pred, dtype="float64")
    error = predicted - true
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "smape_percent": float(
            100.0
            * np.mean(2.0 * np.abs(error) / (np.abs(true) + np.abs(predicted) + EPSILON))
        ),
    }


def burst_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float
) -> dict[str, float | int]:
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


def load_threshold() -> float:
    path = project_path("outputs", "tables", "table_01_burst_threshold.csv")
    table = pd.read_csv(path)
    if len(table) != 1 or "p95_threshold" not in table:
        raise ValueError("Burst threshold table must contain exactly one p95_threshold")
    return float(table.loc[0, "p95_threshold"])


def inverse_predictions(
    target_scaler: StandardScaler, predictions_scaled: np.ndarray
) -> np.ndarray:
    predictions = target_scaler.inverse_transform(
        np.asarray(predictions_scaled).reshape(-1, 1)
    ).reshape(-1)
    if not np.isfinite(predictions).all():
        raise AssertionError("Model produced non-finite predictions")
    return np.clip(predictions, 0.0, None)


def candidate_specs(max_runs: int) -> list[dict[str, float | int]]:
    if not 1 <= max_runs <= len(BOUNDED_CANDIDATES):
        raise ValueError("--max-runs must be in [1, 4]")
    return [dict(spec) for spec in BOUNDED_CANDIDATES[:max_runs]]


def train_and_select(
    tf: Any,
    horizon: int,
    series: pd.DataFrame,
    train_anchor: pd.DataFrame,
    valid_anchor: pd.DataFrame,
    feature_names: list[str],
    threshold: float,
    max_runs: int,
    feature_set: str,
) -> SelectionResult:
    """Run the bounded validation search without opening test data."""
    horizon_steps = validate_horizon(horizon)
    suffix = artifact_suffix(feature_set)
    checkpoint_path = project_path("models", f"lstm_h{horizon}{suffix}.keras")
    tuning_path = project_path(
        "outputs", "tables", f"lstm_tuning_log_h{horizon}{suffix}.csv"
    )
    rows: list[dict[str, Any]] = []
    histories: dict[int, dict[str, list[float]]] = {}

    for candidate_index, spec in enumerate(candidate_specs(max_runs), start=1):
        lookback = int(spec["lookback"])
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        train_set = prepare_sequence_set(
            series,
            train_anchor,
            "train",
            feature_names,
            lookback,
            horizon_steps,
            feature_scaler,
            target_scaler,
            fit_scalers=True,
        )
        valid_set = prepare_sequence_set(
            series,
            valid_anchor,
            "valid",
            feature_names,
            lookback,
            horizon_steps,
            feature_scaler,
            target_scaler,
            fit_scalers=False,
        )

        reset_training_state(tf)
        model = build_model(
            tf,
            lookback=lookback,
            n_features=len(feature_names),
            units=int(spec["units"]),
            dropout=float(spec["dropout"]),
        )
        print(
            f"\nLSTM h{horizon} candidate {candidate_index}/{max_runs}: "
            f"units={spec['units']}, lookback={lookback}, dropout={spec['dropout']}"
        )
        model.summary()
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=PATIENCE,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                str(checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
            ),
        ]
        started = time.perf_counter()
        history_object = model.fit(
            train_set.X,
            train_set.y_scaled,
            validation_data=(valid_set.X, valid_set.y_scaled),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            shuffle=False,
            verbose=2,
        )
        elapsed = time.perf_counter() - started
        history = {
            key: [float(value) for value in values]
            for key, values in history_object.history.items()
        }
        if not history.get("val_loss") or not np.isfinite(history["val_loss"]).all():
            raise AssertionError(f"Unstable validation history for h{horizon} candidate {candidate_index}")
        best_epoch = int(np.argmin(history["val_loss"])) + 1

        train_scaled_prediction = model.predict(
            train_set.X, batch_size=BATCH_SIZE, verbose=0
        ).reshape(-1)
        valid_scaled_prediction = model.predict(
            valid_set.X, batch_size=BATCH_SIZE, verbose=0
        ).reshape(-1)
        train_prediction = inverse_predictions(target_scaler, train_scaled_prediction)
        valid_prediction = inverse_predictions(target_scaler, valid_scaled_prediction)
        train_metrics = regression_metrics(train_set.y_raw, train_prediction)
        valid_metrics = regression_metrics(valid_set.y_raw, valid_prediction)
        classification = burst_metrics(valid_set.y_raw, valid_prediction, threshold)
        persistent_gap = bool(
            all(
                valid_loss > train_loss
                for train_loss, valid_loss in zip(history["loss"], history["val_loss"])
            )
        )
        row = {
            "horizon_minutes": horizon,
            "feature_set": feature_set,
            "candidate_index": candidate_index,
            **spec,
            "parameter_count": int(model.count_params()),
            "epochs_completed": len(history["loss"]),
            "best_epoch": best_epoch,
            "best_train_scaled_mae": float(history["loss"][best_epoch - 1]),
            "best_valid_scaled_mae": float(history["val_loss"][best_epoch - 1]),
            "train_mae_tokens": train_metrics["mae"],
            "valid_mae_tokens": valid_metrics["mae"],
            "valid_rmse_tokens": valid_metrics["rmse"],
            "valid_f1": classification["f1"],
            "valid_precision": classification["precision"],
            "valid_recall": classification["recall"],
            "persistent_generalization_gap": persistent_gap,
            "fit_seconds": elapsed,
            "selected": False,
            "selection_data": "validation_only",
        }
        rows.append(row)
        histories[candidate_index] = history
        pd.DataFrame(rows).to_csv(tuning_path, index=False)
        print(
            f"candidate {candidate_index}: best_epoch={best_epoch}, "
            f"train_MAE={train_metrics['mae']:,.2f}, "
            f"valid_MAE={valid_metrics['mae']:,.2f}, F1={classification['f1']:.4f}"
        )

        del model, train_set, valid_set, train_scaled_prediction, valid_scaled_prediction
        gc.collect()

    tuning = pd.DataFrame(rows).sort_values(
        ["valid_mae_tokens", "valid_f1", "units", "lookback", "dropout"],
        ascending=[True, False, True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)
    tuning.loc[0, "selected"] = True
    selected_index = int(tuning.loc[0, "candidate_index"])
    selected_history = histories[selected_index]
    tuning.to_csv(tuning_path, index=False)
    _merge_rows(
        project_path("outputs", "tables", f"lstm_tuning_log{suffix}.csv"),
        tuning.to_dict(orient="records"),
        ["horizon_minutes", "candidate_index"],
    )
    best = tuning.iloc[0]
    return SelectionResult(
        spec={
            "units": int(best["units"]),
            "lookback": int(best["lookback"]),
            "dropout": float(best["dropout"]),
        },
        best_epoch=int(best["best_epoch"]),
        history=selected_history,
        tuning=tuning,
        parameter_count=int(best["parameter_count"]),
        train_mae=float(best["train_mae_tokens"]),
        valid_mae=float(best["valid_mae_tokens"]),
        valid_f1=float(best["valid_f1"]),
        persistent_generalization_gap=bool(best["persistent_generalization_gap"]),
    )


def save_training_history(
    horizon: int,
    history: dict[str, list[float]],
    feature_set: str,
) -> tuple[Path, Path]:
    suffix = artifact_suffix(feature_set)
    table_path = project_path("outputs", "tables", f"lstm_history_h{horizon}{suffix}.csv")
    figure_path = project_path(
        "outputs", "figures", f"fig_lstm_training_h{horizon}{suffix}.png"
    )
    history_frame = pd.DataFrame(
        {
            "epoch": np.arange(1, len(history["loss"]) + 1),
            "train_mae_scaled": history["loss"],
            "valid_mae_scaled": history["val_loss"],
            "train_rmse_scaled": history.get("rmse", [np.nan] * len(history["loss"])),
            "valid_rmse_scaled": history.get(
                "val_rmse", [np.nan] * len(history["loss"])
            ),
        }
    )
    history_frame.to_csv(table_path, index=False)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(
        history_frame["epoch"],
        history_frame["train_mae_scaled"],
        color="#31688E",
        linewidth=2.0,
        label="Train MAE",
    )
    ax.plot(
        history_frame["epoch"],
        history_frame["valid_mae_scaled"],
        color="#D98E04",
        linewidth=2.0,
        linestyle="--",
        label="Validation MAE",
    )
    best_epoch = int(history_frame.loc[history_frame["valid_mae_scaled"].idxmin(), "epoch"])
    ax.axvline(best_epoch, color="#4A4A4A", linewidth=1.2, linestyle=":")
    ax.annotate(
        f"Best epoch: {best_epoch}",
        xy=(best_epoch, history_frame["valid_mae_scaled"].min()),
        xytext=(8, 12),
        textcoords="offset points",
        color="#333333",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (training-target standard deviations)")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return table_path, figure_path


def sequence_audit_rows(horizon: int, sequence_set: SequenceSet) -> list[dict[str, Any]]:
    return [
        {
            "horizon_minutes": horizon,
            "split": sequence_set.split,
            "n_samples": len(sequence_set.X),
            "lookback_steps": int(sequence_set.X.shape[1]),
            "n_features": int(sequence_set.X.shape[2]),
            "X_shape": str(tuple(sequence_set.X.shape)),
            "y_shape": str(tuple(sequence_set.y_scaled.shape)),
            "input_start": sequence_set.input_start_time[0].isoformat(),
            "input_end_first": sequence_set.input_end_time[0].isoformat(),
            "feature_time_start": sequence_set.feature_time[0].isoformat(),
            "feature_time_end": sequence_set.feature_time[-1].isoformat(),
            "target_time_start": sequence_set.target_time[0].isoformat(),
            "target_time_end": sequence_set.target_time[-1].isoformat(),
            "feature_scaler_fit_rows": sequence_set.feature_fit_rows,
            "labels_confined_to_split": True,
        }
    ]


def manual_sequence_check_row(
    horizon: int, sequence_set: SequenceSet, sample_index: int = 0
) -> dict[str, Any]:
    expected_label_time = sequence_set.feature_time[sample_index] + pd.Timedelta(
        minutes=horizon
    )
    actual_label_time = sequence_set.target_time[sample_index]
    return {
        "horizon_minutes": horizon,
        "split": sequence_set.split,
        "sample_index": sample_index,
        "input_start_time": sequence_set.input_start_time[sample_index].isoformat(),
        "input_end_time": sequence_set.input_end_time[sample_index].isoformat(),
        "forecast_origin_time": sequence_set.feature_time[sample_index].isoformat(),
        "label_time": actual_label_time.isoformat(),
        "expected_label_time": expected_label_time.isoformat(),
        "target_token_load": float(sequence_set.y_raw[sample_index]),
        "time_alignment_passed": bool(actual_label_time == expected_label_time),
    }


def _merge_rows(path: Path, new_rows: Iterable[dict[str, Any]], replace_keys: list[str]) -> None:
    new_frame = pd.DataFrame(list(new_rows))
    if path.exists():
        existing = pd.read_csv(path)
        if not existing.empty and set(replace_keys).issubset(existing.columns):
            replacement_keys = set(
                map(tuple, new_frame[replace_keys].astype(str).itertuples(index=False, name=None))
            )
            existing_keys = list(
                map(tuple, existing[replace_keys].astype(str).itertuples(index=False, name=None))
            )
            existing = existing.loc[
                [key not in replacement_keys for key in existing_keys]
            ]
            new_frame = pd.concat([existing, new_frame], ignore_index=True, sort=False)
    new_frame.to_csv(path, index=False)


def save_audits(
    audit_rows: list[dict[str, Any]],
    manual_rows: list[dict[str, Any]],
    feature_set: str,
) -> None:
    suffix = artifact_suffix(feature_set)
    _merge_rows(
        project_path("outputs", "tables", f"table_05_lstm_sequence_audit{suffix}.csv"),
        audit_rows,
        ["horizon_minutes", "split"],
    )
    _merge_rows(
        project_path(
            "outputs", "tables", f"table_05_lstm_manual_sequence_check{suffix}.csv"
        ),
        manual_rows,
        ["horizon_minutes", "split", "sample_index"],
    )


def save_model_summary(model: Any, horizon: int, feature_set: str) -> Path:
    suffix = artifact_suffix(feature_set)
    path = project_path("models", f"lstm_h{horizon}{suffix}_summary.txt")
    buffer = io.StringIO()
    model.summary(print_fn=lambda line: buffer.write(line + "\n"))
    # Keras 3 may show optimizer-slot variables in the post-fit summary's
    # displayed total.  Record the architecture-only count explicitly so the
    # experiment log and summary remain unambiguous.
    text = (
        f"Architecture parameters (model.count_params): {int(model.count_params()):,}\n"
        "Optimizer state is not part of the network parameter count.\n\n"
        + buffer.getvalue()
    )
    print(text)
    path.write_text(text, encoding="utf-8")
    return path


def seasonal_validation_mae(horizon: int) -> float | None:
    path = project_path("outputs", "tables", "table_02_baseline_results.csv")
    if not path.exists():
        return None
    table = pd.read_csv(path)
    match = table.loc[
        table["split"].eq("valid")
        & table["model"].eq("seasonal_naive")
        & table["horizon_minutes"].eq(horizon),
        "mae",
    ]
    return float(match.iloc[0]) if len(match) == 1 else None


def save_frozen_config(
    horizon: int,
    feature_names: list[str],
    feature_set: str,
    selection: SelectionResult,
    scaler_fit_rows: int,
    train_rows: int,
    valid_rows: int,
) -> Path:
    suffix = artifact_suffix(feature_set)
    seasonal_mae = seasonal_validation_mae(horizon)
    selected = selection.spec
    config = {
        "horizon_minutes": horizon,
        "horizon_steps": validate_horizon(horizon),
        "frequency_minutes": FREQUENCY_MINUTES,
        "target_column": f"target_h{horizon}",
        "target_definition": f"token_load at t+{horizon} minutes",
        "split_source": f"data/processed/features_h{horizon}_{{split}}.parquet",
        "split_assignment_basis": "target_time; identical target rows to XGBoost",
        "temporal_contract": {
            "forecast_origin": "feature_time t",
            "latest_input_window": "t-5 minutes (window fully completed before t)",
            "target_time": f"t+{horizon} minutes",
        },
        "feature_set": feature_set,
        "features": feature_names,
        "lookback": int(selected["lookback"]),
        "network": {
            "layers": [
                {"type": "Input", "shape": [int(selected["lookback"]), len(feature_names)]},
                {
                    "type": "LSTM",
                    "units": int(selected["units"]),
                    "dropout": float(selected["dropout"]),
                },
                {"type": "Dense", "units": 1},
            ],
            "parameter_count": selection.parameter_count,
            "optimizer": "adam",
            "loss": "mae",
            "metrics": ["rmse"],
        },
        "training": {
            "max_epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "shuffle": False,
            "early_stopping": {
                "monitor": "val_loss",
                "patience": PATIENCE,
                "restore_best_weights": True,
            },
            "checkpoint": f"models/lstm_h{horizon}{suffix}.keras",
            "best_epoch": selection.best_epoch,
            "final_refit": "train + validation sequences for fixed best_epoch",
        },
        "scaling": {
            "feature_scaler": "StandardScaler",
            "feature_fit": "unique input rows used by training sequences only",
            "feature_fit_rows": scaler_fit_rows,
            "target_scaler": "StandardScaler",
            "target_fit": "training labels only",
            "feature_scaler_path": f"models/lstm_feature_scaler_h{horizon}{suffix}.joblib",
            "target_scaler_path": f"models/lstm_target_scaler_h{horizon}{suffix}.joblib",
        },
        "selection": {
            "data": "validation only; test split unopened until this config was written",
            "primary_metric": "MAE in original Token units",
            "secondary_metric": "burst F1 using training-set P95 threshold",
            "rule": "minimum validation MAE; ties prefer higher F1 then smaller network",
            "formal_runs": len(selection.tuning),
            "maximum_formal_runs": 4,
            "selected_candidate_index": int(selection.tuning.iloc[0]["candidate_index"]),
            "train_mae_tokens": selection.train_mae,
            "valid_mae_tokens": selection.valid_mae,
            "valid_f1": selection.valid_f1,
            "seasonal_naive_valid_mae_tokens": seasonal_mae,
            "outperformed_seasonal_naive": (
                bool(selection.valid_mae < seasonal_mae) if seasonal_mae is not None else None
            ),
            "persistent_generalization_gap": selection.persistent_generalization_gap,
            "stability_action": (
                "kept bounded single-layer model; no search expansion"
                if selection.persistent_generalization_gap
                or (seasonal_mae is not None and selection.valid_mae >= seasonal_mae)
                else "no intervention required"
            ),
            "bounded_candidates": candidate_specs(len(selection.tuning)),
        },
        "fit_rows": {"train": train_rows, "validation": valid_rows},
        "prediction_policy": {
            "inverse_transform": "target scaler",
            "negative_predictions": "clip to zero",
            "burst_threshold": "training token_load P95",
            "test": "one predict call after validation configuration freeze",
        },
        "random_seeds": {
            "python": SEED,
            "numpy": SEED,
            "tensorflow": SEED,
            "python_hash_seed": SEED,
        },
        "forbidden_architectures": ["Transformer", "Attention", "CNN", "stacked LSTM"],
    }
    path = project_path("configs", f"lstm_h{horizon}{suffix}.yaml")
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    return path


def update_experiment_log(
    horizon: int,
    feature_set: str,
    selection: SelectionResult,
    config_path: Path,
) -> None:
    path = project_path("docs", "experiment_log.md")
    suffix = artifact_suffix(feature_set)
    marker_name = f"LSTM_H{horizon}{suffix.upper()}"
    start_marker = f"<!-- {marker_name}_START -->"
    end_marker = f"<!-- {marker_name}_END -->"
    block = (
        f"{start_marker}\n"
        f"### LSTM h{horizon} ({feature_set})\n\n"
        f"- Config: `configs/{config_path.name}`\n"
        f"- Architecture: Input → LSTM({selection.spec['units']}, "
        f"dropout={selection.spec['dropout']}) → Dense(1)\n"
        f"- Lookback: {selection.spec['lookback']} × 5 minutes\n"
        f"- Parameters: {selection.parameter_count:,}\n"
        f"- Best epoch: {selection.best_epoch}\n"
        f"- Train MAE: {selection.train_mae:,.3f} Token\n"
        f"- Validation MAE: {selection.valid_mae:,.3f} Token\n"
        f"- Validation burst F1: {selection.valid_f1:.4f}\n"
        f"- Seed (Python/NumPy/TensorFlow): {SEED}\n"
        f"{end_marker}"
    )
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Experiment log\n"
    if start_marker in existing and end_marker in existing:
        before, remainder = existing.split(start_marker, 1)
        _, after = remainder.split(end_marker, 1)
        updated = before.rstrip() + "\n\n" + block + after
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    path.write_text(updated, encoding="utf-8")


def prepare_selected_sets(
    horizon: int,
    series: pd.DataFrame,
    train_anchor: pd.DataFrame,
    valid_anchor: pd.DataFrame,
    feature_names: list[str],
    selection: SelectionResult,
    feature_set: str,
) -> tuple[SequenceSet, SequenceSet, StandardScaler, StandardScaler]:
    lookback = int(selection.spec["lookback"])
    horizon_steps = validate_horizon(horizon)
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    train_set = prepare_sequence_set(
        series,
        train_anchor,
        "train",
        feature_names,
        lookback,
        horizon_steps,
        feature_scaler,
        target_scaler,
        fit_scalers=True,
    )
    valid_set = prepare_sequence_set(
        series,
        valid_anchor,
        "valid",
        feature_names,
        lookback,
        horizon_steps,
        feature_scaler,
        target_scaler,
        fit_scalers=False,
    )
    suffix = artifact_suffix(feature_set)
    joblib.dump(
        feature_scaler,
        project_path("models", f"lstm_feature_scaler_h{horizon}{suffix}.joblib"),
    )
    joblib.dump(
        target_scaler,
        project_path("models", f"lstm_target_scaler_h{horizon}{suffix}.joblib"),
    )
    return train_set, valid_set, feature_scaler, target_scaler


def final_refit(
    tf: Any,
    horizon: int,
    train_set: SequenceSet,
    valid_set: SequenceSet,
    feature_names: list[str],
    selection: SelectionResult,
    feature_set: str,
) -> Any:
    """Refit the frozen model for the selected epoch count on train+validation."""
    reset_training_state(tf)
    model = build_model(
        tf,
        lookback=int(selection.spec["lookback"]),
        n_features=len(feature_names),
        units=int(selection.spec["units"]),
        dropout=float(selection.spec["dropout"]),
    )
    X_combined = np.concatenate([train_set.X, valid_set.X], axis=0)
    y_combined = np.concatenate([train_set.y_scaled, valid_set.y_scaled], axis=0)
    model.fit(
        X_combined,
        y_combined,
        epochs=selection.best_epoch,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=2,
    )
    suffix = artifact_suffix(feature_set)
    model_path = project_path("models", f"lstm_h{horizon}{suffix}.keras")
    model.save(model_path)
    save_model_summary(model, horizon, feature_set)
    if int(model.count_params()) != selection.parameter_count:
        raise AssertionError("Final model parameter count changed after selection")
    del X_combined, y_combined
    gc.collect()
    return model


def final_test_once(
    model: Any,
    horizon: int,
    series: pd.DataFrame,
    test_anchor: pd.DataFrame,
    feature_names: list[str],
    selection: SelectionResult,
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
    threshold: float,
    feature_set: str,
) -> tuple[SequenceSet, dict[str, Any]]:
    """Construct test sequences and invoke ``predict`` exactly once."""
    test_set = prepare_sequence_set(
        series,
        test_anchor,
        "test",
        feature_names,
        int(selection.spec["lookback"]),
        validate_horizon(horizon),
        feature_scaler,
        target_scaler,
        fit_scalers=False,
    )
    scaled_predictions = model.predict(
        test_set.X, batch_size=BATCH_SIZE, verbose=0
    ).reshape(-1)  # the sole test prediction call
    predictions = inverse_predictions(target_scaler, scaled_predictions)
    regression = regression_metrics(test_set.y_raw, predictions)
    classification = burst_metrics(test_set.y_raw, predictions, threshold)
    output = pd.DataFrame(
        {
            "feature_time": test_set.feature_time,
            "input_end_time": test_set.input_end_time,
            "target_time": test_set.target_time,
            "horizon_minutes": horizon,
            "model_stage": "final",
            "actual_token_load": test_set.y_raw,
            "predicted_token_load": predictions,
            "actual_burst": test_set.y_raw >= threshold,
            "predicted_burst": predictions >= threshold,
        }
    )
    suffix = artifact_suffix(feature_set)
    output.to_csv(
        project_path("outputs", "tables", f"pred_lstm_test_h{horizon}{suffix}.csv"),
        index=False,
    )
    metrics = {
        "split": "test",
        "model": "lstm_final" if feature_set == "main" else "lstm_history_load_ablation",
        "feature_set": feature_set,
        "horizon_minutes": horizon,
        "n_observations": len(test_set.X),
        **regression,
        **classification,
        "burst_threshold": threshold,
        "negative_predictions_after_clip": int((predictions < 0.0).sum()),
        "n_unique_predictions": int(pd.Series(predictions).nunique()),
        "target_time_aligned": True,
    }
    return test_set, metrics


def print_sequence_evidence(horizon: int, sets: Iterable[SequenceSet]) -> None:
    print(f"\nSequence evidence for h{horizon}:")
    for sequence_set in sets:
        check = manual_sequence_check_row(horizon, sequence_set)
        print(
            f"  {sequence_set.split:5s} X={sequence_set.X.shape}, "
            f"y={sequence_set.y_scaled.shape}; origins "
            f"{sequence_set.feature_time[0]}..{sequence_set.feature_time[-1]}; "
            f"labels {sequence_set.target_time[0]}..{sequence_set.target_time[-1]}"
        )
        print(
            "    manual[0]: input_end="
            f"{check['input_end_time']}, origin={check['forecast_origin_time']}, "
            f"label={check['label_time']}, "
            f"target={check['target_token_load']:,.3f}, "
            f"aligned={check['time_alignment_passed']}"
        )


def run_horizons(horizons: list[int], max_runs: int, feature_set: str) -> None:
    ensure_output_directories()
    set_global_seed(SEED)
    tf = import_tensorflow()
    threshold = load_threshold()
    feature_names = feature_names_for(feature_set)

    # Phase 1: load only train/validation anchors and only canonical-series rows
    # through the validation endpoint.  Every config is frozen in this phase.
    frozen: dict[int, dict[str, Any]] = {}
    for horizon in horizons:
        train_anchor = load_split_anchor(horizon, "train")
        valid_anchor = load_split_anchor(horizon, "valid")
        selection_series = load_series_through(valid_anchor["target_time"].iloc[-1])
        feature_frame = build_lstm_features(selection_series)
        # ``token_load`` contributes only through completed windows ending
        # before the forecast origin; future rows in the frame supply labels.
        selection_frame = feature_frame

        selection = train_and_select(
            tf,
            horizon,
            selection_frame,
            train_anchor,
            valid_anchor,
            feature_names,
            threshold,
            max_runs,
            feature_set,
        )
        train_set, valid_set, feature_scaler, target_scaler = prepare_selected_sets(
            horizon,
            selection_frame,
            train_anchor,
            valid_anchor,
            feature_names,
            selection,
            feature_set,
        )
        save_training_history(horizon, selection.history, feature_set)
        config_path = save_frozen_config(
            horizon,
            feature_names,
            feature_set,
            selection,
            train_set.feature_fit_rows,
            len(train_set.X),
            len(valid_set.X),
        )
        update_experiment_log(horizon, feature_set, selection, config_path)
        print(f"Frozen validation-selected configuration: {config_path}")
        frozen[horizon] = {
            "selection": selection,
            "train_anchor": train_anchor,
            "valid_anchor": valid_anchor,
            "train_set": train_set,
            "valid_set": valid_set,
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
        }
        del selection_series, feature_frame, selection_frame
        gc.collect()

    # Phase 2: only now open test anchors/canonical test rows and call predict.
    print("All requested horizon configurations are frozen; opening test data now.")
    metric_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    manual_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        state = frozen[horizon]
        test_anchor = load_split_anchor(horizon, "test")
        full_series = load_series_through(test_anchor["target_time"].iloc[-1])
        full_feature_frame = build_lstm_features(full_series)
        final_model = final_refit(
            tf,
            horizon,
            state["train_set"],
            state["valid_set"],
            feature_names,
            state["selection"],
            feature_set,
        )
        test_set, metrics = final_test_once(
            final_model,
            horizon,
            full_feature_frame,
            test_anchor,
            feature_names,
            state["selection"],
            state["feature_scaler"],
            state["target_scaler"],
            threshold,
            feature_set,
        )
        metric_rows.append(metrics)
        sets = [state["train_set"], state["valid_set"], test_set]
        for sequence_set in sets:
            audit_rows.extend(sequence_audit_rows(horizon, sequence_set))
            manual_rows.append(manual_sequence_check_row(horizon, sequence_set))
        print_sequence_evidence(horizon, sets)
        del full_series, full_feature_frame, final_model, test_set
        gc.collect()

    suffix = artifact_suffix(feature_set)
    _merge_rows(
        project_path("outputs", "tables", f"table_05_lstm_results{suffix}.csv"),
        metric_rows,
        ["horizon_minutes", "split", "model"],
    )
    save_audits(audit_rows, manual_rows, feature_set)
    print("\nFinal LSTM test metrics:")
    print(pd.DataFrame(metric_rows).to_string(index=False))


def finalize_frozen_horizons(horizons: list[int], feature_set: str) -> None:
    ensure_output_directories()
    set_global_seed(SEED)
    tf = import_tensorflow()
    threshold = load_threshold()
    feature_names = feature_names_for(feature_set)
    suffix = artifact_suffix(feature_set)
    metric_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    manual_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        config_path = project_path("configs", f"lstm_h{horizon}{suffix}.yaml")
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        selected = config["selection"]
        network = config["network"]
        selection = SelectionResult(
            spec={
                "units": int(network["layers"][1]["units"]),
                "lookback": int(config["lookback"]),
                "dropout": float(network["layers"][1]["dropout"]),
            },
            best_epoch=int(config["training"]["best_epoch"]),
            history={},
            tuning=pd.DataFrame(),
            parameter_count=int(network["parameter_count"]),
            train_mae=float(selected["train_mae_tokens"]),
            valid_mae=float(selected["valid_mae_tokens"]),
            valid_f1=float(selected["valid_f1"]),
            persistent_generalization_gap=bool(selected["persistent_generalization_gap"]),
        )
        train_anchor = load_split_anchor(horizon, "train")
        valid_anchor = load_split_anchor(horizon, "valid")
        test_anchor = load_split_anchor(horizon, "test")
        full_series = load_series_through(test_anchor["target_time"].iloc[-1])
        feature_frame = build_lstm_features(full_series)
        feature_scaler = joblib.load(
            project_path("models", f"lstm_feature_scaler_h{horizon}{suffix}.joblib")
        )
        target_scaler = joblib.load(
            project_path("models", f"lstm_target_scaler_h{horizon}{suffix}.joblib")
        )
        train_set = prepare_sequence_set(
            feature_frame,
            train_anchor,
            "train",
            feature_names,
            int(selection.spec["lookback"]),
            validate_horizon(horizon),
            feature_scaler,
            target_scaler,
            fit_scalers=False,
        )
        valid_set = prepare_sequence_set(
            feature_frame,
            valid_anchor,
            "valid",
            feature_names,
            int(selection.spec["lookback"]),
            validate_horizon(horizon),
            feature_scaler,
            target_scaler,
            fit_scalers=False,
        )
        model = tf.keras.models.load_model(
            project_path("models", f"lstm_h{horizon}{suffix}.keras")
        )
        if int(model.count_params()) != selection.parameter_count:
            raise AssertionError("Frozen model parameter count differs from YAML")
        save_model_summary(model, horizon, feature_set)
        test_set, metrics = final_test_once(
            model,
            horizon,
            feature_frame,
            test_anchor,
            feature_names,
            selection,
            feature_scaler,
            target_scaler,
            threshold,
            feature_set,
        )
        metric_rows.append(metrics)
        sets = [train_set, valid_set, test_set]
        for sequence_set in sets:
            audit_rows.extend(sequence_audit_rows(horizon, sequence_set))
            manual_rows.append(manual_sequence_check_row(horizon, sequence_set))
        print_sequence_evidence(horizon, sets)
    _merge_rows(
        project_path("outputs", "tables", f"table_05_lstm_results{suffix}.csv"),
        metric_rows,
        ["horizon_minutes", "split", "model"],
    )
    save_audits(audit_rows, manual_rows, feature_set)
    print(pd.DataFrame(metric_rows).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--horizon",
        type=int,
        action="append",
        default=None,
        help="Forecast horizon in minutes; repeat for multiple horizons (default: 5, 15, 60)",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=1,
        help="Number of bounded validation candidates, from 1 to 4 (default: 1)",
    )
    parser.add_argument(
        "--feature-set",
        choices=("main", "history_load"),
        default="main",
        help="Main 10-variable input or token_load-only ablation",
    )
    parser.add_argument("--finalize-frozen", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_base_config()
    horizons = args.horizon or [int(value) for value in config["series"]["horizons_minutes"]]
    if len(set(horizons)) != len(horizons):
        raise ValueError("Duplicate --horizon values are not allowed")
    for horizon in horizons:
        validate_horizon(horizon)
    candidate_specs(args.max_runs)
    if args.finalize_frozen:
        finalize_frozen_horizons(horizons, args.feature_set)
    else:
        run_horizons(horizons, args.max_runs, args.feature_set)


if __name__ == "__main__":
    main()
