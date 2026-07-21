"""Generate leakage-safe persistence and daily seasonal-naive forecasts.

Prediction rows are indexed by the *target* window.  A forecast made at origin
``o`` may use only windows that have fully ended before ``o``.  Because each
timestamp is the left edge of a 5-minute window, the last available load is
``y[o-5min]``.  Thus persistence for target ``o+h`` is ``y[o-5min]`` (that is,
``h/5+1`` rows back from the target), while daily seasonal naive uses the
completed window starting exactly one day before the target.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, confusion_matrix

from utils import ensure_output_directories, project_path


STEPS_PER_DAY = 288
EPSILON = 1e-9


def load_config() -> dict:
    with project_path("configs", "base.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_series() -> pd.DataFrame:
    path = project_path("data", "processed", "series_5min.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Missing canonical series: {path}; run src/02_build_series.py")
    series = pd.read_parquet(path)
    series.index = pd.DatetimeIndex(pd.to_datetime(series.index, utc=True))
    series.index.name = "timestamp"
    required = {"token_load", "split", "actual_burst"}
    missing = required.difference(series.columns)
    if missing:
        raise ValueError(f"Canonical series is missing columns: {sorted(missing)}")
    if not series.index.is_unique or not series.index.is_monotonic_increasing:
        raise ValueError("Canonical series index must be unique and increasing")
    if not series.index.to_series().diff().dropna().eq(pd.Timedelta(minutes=5)).all():
        raise ValueError("Canonical series is not a complete 5-minute grid")
    return series


def load_threshold() -> float:
    path = project_path("outputs", "tables", "table_01_burst_threshold.csv")
    table = pd.read_csv(path)
    if len(table) != 1 or "p95_threshold" not in table:
        raise ValueError(f"Invalid training-threshold table: {path}")
    return float(table.loc[0, "p95_threshold"])


def evaluate_baselines(
    predictions: pd.DataFrame, horizon_minutes: list[int], threshold: float
) -> pd.DataFrame:
    """Compute validation/test metrics needed by all later model stages."""
    rows: list[dict[str, float | int | str | bool]] = []
    for split_name in ("valid", "test"):
        part = predictions.loc[predictions["split"].eq(split_name)]
        y_true = part["actual_token_load"].to_numpy(dtype="float64")
        actual = y_true >= threshold
        for minutes in horizon_minutes:
            for model, column in (
                ("persistence", f"pred_persistence_h{minutes}"),
                ("seasonal_naive", f"pred_seasonal_h{minutes}"),
            ):
                y_pred = part[column].to_numpy(dtype="float64")
                error = y_pred - y_true
                predicted = y_pred >= threshold
                tn, fp, fn, tp = confusion_matrix(
                    actual, predicted, labels=[False, True]
                ).ravel()
                precision = float(tp / (tp + fp)) if tp + fp else 0.0
                recall = float(tp / (tp + fn)) if tp + fn else 0.0
                f1 = (
                    float(2.0 * precision * recall / (precision + recall))
                    if precision + recall
                    else 0.0
                )
                ap_defined = bool(actual.any())
                average_precision = (
                    float(average_precision_score(actual, y_pred))
                    if ap_defined
                    else 0.0
                )
                rows.append(
                    {
                        "split": split_name,
                        "model": model,
                        "horizon_minutes": minutes,
                        "n_observations": len(part),
                        "train_p95_threshold": threshold,
                        "mae": float(np.mean(np.abs(error))),
                        "rmse": float(np.sqrt(np.mean(np.square(error)))),
                        "smape_percent": float(
                            100.0
                            * np.mean(
                                2.0
                                * np.abs(error)
                                / (np.abs(y_true) + np.abs(y_pred) + EPSILON)
                            )
                        ),
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "average_precision": average_precision,
                        "pr_auc": average_precision,
                        "average_precision_defined": ap_defined,
                        "true_negative": int(tn),
                        "false_positive": int(fp),
                        "false_negative": int(fn),
                        "true_positive": int(tp),
                        "actual_burst_windows": int(actual.sum()),
                        "predicted_burst_windows": int(predicted.sum()),
                        "actual_burst_rate": float(actual.mean()),
                        "predicted_burst_rate": float(predicted.mean()),
                    }
                )
    return pd.DataFrame(rows)


def build_predictions(series: pd.DataFrame, horizon_minutes: list[int]) -> pd.DataFrame:
    """Create wide target-aligned forecasts and explicit forecast-origin columns."""
    target = series["token_load"].astype("float64")
    output = pd.DataFrame(index=series.index)
    output["split"] = series["split"].astype("string")
    output["actual_token_load"] = target
    output["actual_burst"] = series["actual_burst"].astype(bool)

    for minutes in horizon_minutes:
        if minutes <= 0 or minutes % 5:
            raise ValueError(f"Horizon must be a positive multiple of 5 minutes: {minutes}")
        steps = minutes // 5
        persistence_name = f"pred_persistence_h{minutes}"
        seasonal_name = f"pred_seasonal_h{minutes}"
        output[f"forecast_origin_h{minutes}"] = output.index - pd.Timedelta(minutes=minutes)
        output[f"last_available_window_h{minutes}"] = (
            output[f"forecast_origin_h{minutes}"] - pd.Timedelta(minutes=5)
        )
        persistence_lag = steps + 1
        output[persistence_name] = target.shift(persistence_lag)
        output[seasonal_name] = target.shift(STEPS_PER_DAY)

        # Alignment gates: all source windows must be fully complete at origin.
        np.testing.assert_array_equal(
            output[persistence_name].iloc[persistence_lag:].to_numpy(),
            target.iloc[:-persistence_lag].to_numpy(),
        )
        np.testing.assert_array_equal(
            output[seasonal_name].iloc[STEPS_PER_DAY:].to_numpy(),
            target.iloc[:-STEPS_PER_DAY].to_numpy(),
        )
        assert (
            output[f"last_available_window_h{minutes}"]
            < output[f"forecast_origin_h{minutes}"]
        ).all()
        assert (
            output.index.to_series() - pd.Timedelta(days=1) + pd.Timedelta(minutes=5)
            <= output[f"forecast_origin_h{minutes}"]
        ).all()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--horizons-minutes",
        type=int,
        nargs="*",
        default=None,
        help="Override configured horizons (default: 5 15 60)",
    )
    args = parser.parse_args()

    ensure_output_directories()
    config = load_config()
    horizons = args.horizons_minutes or [int(value) for value in config["series"]["horizons_minutes"]]
    series = load_series()
    predictions = build_predictions(series, horizons)
    threshold = load_threshold()

    table_dir = project_path("outputs", "tables")
    for split_name in ("valid", "test"):
        split_predictions = predictions.loc[predictions["split"].eq(split_name)].copy()
        forecast_columns = [column for column in split_predictions if column.startswith("pred_")]
        if split_predictions[forecast_columns].isna().any().any():
            raise AssertionError(f"{split_name} contains missing baseline forecasts")
        output_path = table_dir / f"pred_baselines_{split_name}.csv"
        split_predictions.reset_index().to_csv(output_path, index=False)
        print(
            f"Saved {split_name} predictions: {output_path} "
            f"({len(split_predictions):,} target windows)"
        )
    metrics = evaluate_baselines(predictions, horizons, threshold)
    metrics.to_csv(table_dir / "table_02_baseline_results.csv", index=False)
    print(f"Saved baseline metrics: {table_dir / 'table_02_baseline_results.csv'}")


if __name__ == "__main__":
    main()
