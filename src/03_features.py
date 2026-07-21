"""Build leakage-safe features and target-time splits for 5/15/60-minute forecasts.

Every model feature is either known calendar phase at the forecast origin or is
computed from observations ending at least one complete 5-minute window before
the origin.  Split membership is deliberately assigned from ``target_time``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from utils import ensure_output_directories, project_path


FREQUENCY_MINUTES = 5
LAGS = (1, 2, 3, 6, 12, 24, 48, 288)
ROLLING_WINDOWS = (3, 6, 12)
SPLIT_ORDER = ("train", "valid", "test")
RELATIVE_TIME_ORIGIN = pd.Timestamp("1970-01-01", tz="UTC")


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    category: str
    availability: str
    obtained_at_prediction_time: str


def load_config() -> dict:
    with project_path("configs", "base.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_series() -> pd.DataFrame:
    path = project_path("data", "processed", "series_5min.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Missing canonical series: {path}; run src/02_build_series.py")
    series = pd.read_parquet(path).sort_index(kind="mergesort")
    series.index = pd.DatetimeIndex(pd.to_datetime(series.index, utc=True), name="timestamp")
    required = {
        "token_load",
        "request_count",
        "mean_request_tokens",
        "mean_response_tokens",
        "gpt4_request_count",
        "api_request_count",
        "split",
        "actual_burst",
    }
    missing = required.difference(series.columns)
    if missing:
        raise ValueError(f"Canonical series is missing columns: {sorted(missing)}")
    if not series.index.is_unique or not series.index.is_monotonic_increasing:
        raise ValueError("Canonical series index must be unique and increasing")
    expected = pd.date_range(series.index.min(), series.index.max(), freq="5min")
    if not series.index.equals(expected):
        raise ValueError("Canonical series must be a complete 5-minute grid")
    observed_order = tuple(series["split"].drop_duplicates().astype(str))
    if observed_order != SPLIT_ORDER:
        raise ValueError(f"Expected chronological splits {SPLIT_ORDER}, got {observed_order}")
    return series


def _past_rolling_sum(series: pd.Series, window: int) -> pd.Series:
    """Aggregate complete past windows; the shift is the leakage-safety gate."""
    return series.shift(1).rolling(window=window, min_periods=window).sum()


def build_feature_matrix(series: pd.DataFrame) -> tuple[pd.DataFrame, list[FeatureDefinition]]:
    """Create features available at origin t, never values observed in [t, t+h]."""
    features = pd.DataFrame(index=series.index)
    definitions: list[FeatureDefinition] = []
    load = series["token_load"].astype("float64")

    for lag in LAGS:
        name = f"lag_{lag}"
        features[name] = load.shift(lag)
        definitions.append(
            FeatureDefinition(
                name,
                "historical_load",
                "available_at_t",
                f"token_load observed {lag * FREQUENCY_MINUTES} minutes before t",
            )
        )

    for window in ROLLING_WINDOWS:
        # Keep the required expression explicit so code review can verify that
        # the current/future target window never enters a rolling statistic.
        for statistic in ("mean", "std", "max"):
            name = f"rolling_{statistic}_{window}"
            features[name] = load.shift(1).rolling(
                window=window, min_periods=window
            ).agg(statistic)
            definitions.append(
                FeatureDefinition(
                    name,
                    "historical_load",
                    "available_at_t",
                    f"{statistic} of token_load in the {window} completed windows before t",
                )
            )

    request_count = series["request_count"].astype("float64")
    historical_request_tokens = (
        series["mean_request_tokens"].fillna(0.0).astype("float64") * request_count
    )
    historical_response_tokens = (
        series["mean_response_tokens"].fillna(0.0).astype("float64") * request_count
    )
    structure_numerators = {
        "gpt4_share": series["gpt4_request_count"].astype("float64"),
        "api_share": series["api_request_count"].astype("float64"),
        "mean_request_tokens": historical_request_tokens,
        "mean_response_tokens": historical_response_tokens,
    }
    for window in ROLLING_WINDOWS:
        denominator = _past_rolling_sum(request_count, window)
        count_name = f"request_count_rolling_mean_{window}"
        features[count_name] = request_count.shift(1).rolling(
            window=window, min_periods=window
        ).mean()
        definitions.append(
            FeatureDefinition(
                count_name,
                "historical_request_structure",
                "available_at_t",
                f"mean request count in the {window} completed windows before t",
            )
        )
        for prefix, numerator in structure_numerators.items():
            name = f"{prefix}_history_{window}"
            ratio = _past_rolling_sum(numerator, window).div(denominator.replace(0.0, np.nan))
            # A zero denominator means the entire historical window had no
            # requests; zero is an explicit, past-known sentinel in that case.
            features[name] = ratio.fillna(0.0)
            value_kind = "share" if prefix.endswith("share") else "per-request mean"
            definitions.append(
                FeatureDefinition(
                    name,
                    "historical_request_structure",
                    "available_at_t",
                    f"historical {value_kind} aggregated over the {window} completed windows before t",
                )
            )

    # The public Timestamp field is relative seconds and is represented on a
    # synthetic UTC axis anchored at the Unix epoch.  Keeping that same anchor
    # matters when the frozen 1/2 models are applied directly to batch 3.
    elapsed_seconds = (series.index - RELATIVE_TIME_ORIGIN).total_seconds().to_numpy()
    phases = {
        "relative_day": 24 * 60 * 60,
        "relative_week": 7 * 24 * 60 * 60,
    }
    for prefix, period_seconds in phases.items():
        angle = 2.0 * np.pi * (elapsed_seconds % period_seconds) / period_seconds
        for suffix, values in (("sin", np.sin(angle)), ("cos", np.cos(angle))):
            name = f"{prefix}_{suffix}"
            features[name] = values
            definitions.append(
                FeatureDefinition(
                    name,
                    "relative_calendar_phase",
                    "available_at_t",
                    f"deterministic {prefix} phase from relative time at t; not a real weekday/date",
                )
            )

    if features.columns.duplicated().any():
        raise AssertionError("Duplicate feature names")
    if {definition.name for definition in definitions} != set(features.columns):
        raise AssertionError("Feature definitions do not exactly cover the model matrix")
    forbidden_fragments = ("target_", "future", "is_weekend", "weekday")
    offenders = [
        column for column in features if any(fragment in column.lower() for fragment in forbidden_fragments)
    ]
    if offenders:
        raise AssertionError(f"Forbidden future/real-calendar features found: {offenders}")
    return features, definitions


def assign_target_split(series: pd.DataFrame, target_time: pd.Series) -> pd.Series:
    """Map each sample to the split containing its target, never its origin."""
    mapping = series["split"].astype("string")
    values = mapping.reindex(pd.DatetimeIndex(target_time)).array
    return pd.Series(values, index=target_time.index, dtype="string")


def build_horizon_dataset(
    series: pd.DataFrame, features: pd.DataFrame, horizon_minutes: int
) -> pd.DataFrame:
    if horizon_minutes <= 0 or horizon_minutes % FREQUENCY_MINUTES:
        raise ValueError(f"Horizon must be a positive multiple of 5: {horizon_minutes}")
    steps = horizon_minutes // FREQUENCY_MINUTES
    target_name = f"target_h{horizon_minutes}"

    # Required target construction (h15 is exactly token_load.shift(-3)).
    target = series["token_load"].astype("float64").shift(-steps).rename(target_name)
    target_time = pd.Series(
        series.index + pd.Timedelta(minutes=horizon_minutes),
        index=series.index,
        name="target_time",
    )
    split = assign_target_split(series, target_time).rename("split")
    actual_burst = series["actual_burst"].shift(-steps).rename("actual_burst")

    dataset = features.copy()
    dataset.insert(0, "feature_time", series.index)
    dataset.insert(1, "target_time", target_time)
    dataset.insert(2, "split", split)
    dataset.insert(3, target_name, target)
    dataset.insert(4, "actual_burst", actual_burst)
    # Rows missing because of lag/rolling warm-up or future target shifting are
    # removed only after every feature and target has been assembled.
    required_columns = [*features.columns, target_name, "target_time", "split", "actual_burst"]
    dataset = dataset.dropna(subset=required_columns).reset_index(drop=True)
    dataset["actual_burst"] = dataset["actual_burst"].astype(bool)

    expected_delta = pd.Timedelta(minutes=horizon_minutes)
    if not (dataset["target_time"] - dataset["feature_time"]).eq(expected_delta).all():
        raise AssertionError(f"Target alignment failed for h{horizon_minutes}")
    expected_target = series["token_load"].reindex(pd.DatetimeIndex(dataset["target_time"]))
    np.testing.assert_array_equal(dataset[target_name].to_numpy(), expected_target.to_numpy())
    expected_burst = series["actual_burst"].reindex(pd.DatetimeIndex(dataset["target_time"]))
    np.testing.assert_array_equal(dataset["actual_burst"].to_numpy(), expected_burst.to_numpy())

    for previous, following in zip(SPLIT_ORDER, SPLIT_ORDER[1:]):
        previous_target_end = dataset.loc[dataset["split"].eq(previous), "target_time"].max()
        following_target_start = dataset.loc[dataset["split"].eq(following), "target_time"].min()
        if not previous_target_end < following_target_start:
            raise AssertionError(
                f"Target-time leakage between {previous} and {following} for h{horizon_minutes}"
            )
    return dataset


def save_outputs(
    series: pd.DataFrame,
    features: pd.DataFrame,
    definitions: list[FeatureDefinition],
    horizons: list[int],
) -> None:
    summary_rows: list[dict[str, object]] = []
    processed_dir = project_path("data", "processed")
    table_dir = project_path("outputs", "tables")
    feature_names = list(features.columns)

    for horizon in horizons:
        dataset = build_horizon_dataset(series, features, horizon)
        for split_name in SPLIT_ORDER:
            part = dataset.loc[dataset["split"].eq(split_name)].copy()
            if part.empty:
                raise AssertionError(f"Empty {split_name} dataset for h{horizon}")
            output_path = processed_dir / f"features_h{horizon}_{split_name}.parquet"
            part.to_parquet(output_path, compression="zstd", index=False)
            summary_rows.append(
                {
                    "horizon_minutes": horizon,
                    "split": split_name,
                    "split_assignment_basis": "target_time",
                    "n_samples": len(part),
                    "n_features": len(feature_names),
                    "feature_names": ";".join(feature_names),
                    "feature_time_start": part["feature_time"].min().isoformat(),
                    "feature_time_end": part["feature_time"].max().isoformat(),
                    "target_time_start": part["target_time"].min().isoformat(),
                    "target_time_end": part["target_time"].max().isoformat(),
                    "target_column": f"target_h{horizon}",
                    "leakage_check": "passed",
                }
            )

    pd.DataFrame(summary_rows).to_csv(
        table_dir / "table_03_feature_split_summary.csv", index=False
    )
    pd.DataFrame([definition.__dict__ for definition in definitions]).to_csv(
        table_dir / "table_03_feature_availability.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons-minutes", type=int, nargs="*", default=None)
    args = parser.parse_args()

    ensure_output_directories()
    config = load_config()
    horizons = args.horizons_minutes or [int(value) for value in config["series"]["horizons_minutes"]]
    series = load_series()
    features, definitions = build_feature_matrix(series)
    save_outputs(series, features, definitions, horizons)
    print(
        f"Saved {len(features.columns)} leakage-safe features for horizons {horizons}; "
        "all split membership is based on target_time."
    )
    print(project_path("outputs", "tables", "table_03_feature_split_summary.csv"))


if __name__ == "__main__":
    main()
