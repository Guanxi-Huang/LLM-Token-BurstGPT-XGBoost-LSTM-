"""Validate Week-3 feature, split, model, prediction, and explanation artifacts.

This audit loads saved models only to inspect their feature schema; it never
calls ``predict`` and therefore does not create another test evaluation.
"""

from __future__ import annotations

import importlib
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
EXPECTED_CANDIDATES_PER_HORIZON = 12
SEARCH_SPACE = {
    "max_depth": {3, 5, 7},
    "learning_rate": {0.02, 0.05, 0.1},
    "n_estimators": {300, 600, 1000},
    "min_child_weight": {1, 5, 10},
    "subsample": {0.7, 0.9, 1.0},
    "colsample_bytree": {0.7, 0.9, 1.0},
}


def validate_feature_values() -> tuple[list[str], pd.DataFrame]:
    module = importlib.import_module("03_features")
    series = module.load_series()
    recomputed, definitions = module.build_feature_matrix(series)
    feature_names = [definition.name for definition in definitions]
    availability = pd.read_csv(ROOT / "outputs/tables/table_03_feature_availability.csv")
    assert availability["name"].tolist() == feature_names
    assert availability["availability"].eq("available_at_t").all()

    split_summary = pd.read_csv(ROOT / "outputs/tables/table_03_feature_split_summary.csv")
    assert len(split_summary) == len(HORIZONS) * len(SPLITS)
    assert split_summary["leakage_check"].eq("passed").all()
    assert split_summary["split_assignment_basis"].eq("target_time").all()

    for horizon in HORIZONS:
        parts = {
            split: pd.read_parquet(
                ROOT / f"data/processed/features_h{horizon}_{split}.parquet"
            )
            for split in SPLITS
        }
        combined = pd.concat(parts.values(), ignore_index=True)
        combined["feature_time"] = pd.to_datetime(combined["feature_time"], utc=True)
        combined["target_time"] = pd.to_datetime(combined["target_time"], utc=True)
        assert combined["feature_time"].is_monotonic_increasing
        assert combined["feature_time"].is_unique
        assert (combined["target_time"] - combined["feature_time"]).eq(
            pd.Timedelta(minutes=horizon)
        ).all()

        expected_x = recomputed.reindex(pd.DatetimeIndex(combined["feature_time"]))
        np.testing.assert_allclose(
            combined[feature_names].to_numpy(dtype="float64"),
            expected_x[feature_names].to_numpy(dtype="float64"),
            rtol=1e-12,
            atol=1e-12,
        )
        expected_y = series["token_load"].reindex(pd.DatetimeIndex(combined["target_time"]))
        np.testing.assert_array_equal(combined[f"target_h{horizon}"], expected_y)
        expected_split = series["split"].reindex(pd.DatetimeIndex(combined["target_time"]))
        np.testing.assert_array_equal(combined["split"].astype(str), expected_split.astype(str))
        for previous, following in zip(SPLITS, SPLITS[1:]):
            assert parts[previous]["target_time"].max() < parts[following]["target_time"].min()
    return feature_names, series


def validate_tuning_configs_models(feature_names: list[str]) -> None:
    log = pd.read_csv(ROOT / "outputs/tables/xgb_tuning_log.csv")
    assert len(log) == len(HORIZONS) * EXPECTED_CANDIDATES_PER_HORIZON
    assert not any("test" in column.lower() for column in log.columns)
    assert log["selection_data"].eq("validation_only").all()
    for horizon in HORIZONS:
        horizon_log = log.loc[log["horizon_minutes"].eq(horizon)]
        assert len(horizon_log) == EXPECTED_CANDIDATES_PER_HORIZON
        selected_log = horizon_log.loc[horizon_log["selected"].astype(bool)]
        assert len(selected_log) == 1
        assert float(selected_log.iloc[0]["valid_mae"]) == float(horizon_log["valid_mae"].min())

        with (ROOT / f"configs/xgb_h{horizon}.yaml").open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        assert config["features"] == feature_names
        assert config["selection"]["data"].startswith("validation only")
        assert config["fixed_params"] == {
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": -1,
        }
        for key, allowed in SEARCH_SPACE.items():
            assert config["selected_params"][key] in allowed
            assert config["selected_params"][key] == selected_log.iloc[0][key]

        model = joblib.load(ROOT / f"models/xgb_h{horizon}.joblib")
        assert model.feature_names_in_.tolist() == feature_names


def validate_predictions_and_explanations(feature_names: list[str], series: pd.DataFrame) -> None:
    prediction_paths = [ROOT / "outputs/tables/pred_xgb_valid_h15.csv"]
    prediction_paths.extend(
        ROOT / f"outputs/tables/pred_xgb_tuned_valid_h{horizon}.csv" for horizon in HORIZONS
    )
    prediction_paths.extend(
        ROOT / f"outputs/tables/pred_xgb_test_h{horizon}.csv" for horizon in HORIZONS
    )
    for path in prediction_paths:
        frame = pd.read_csv(path, parse_dates=["feature_time", "target_time"])
        frame["feature_time"] = pd.to_datetime(frame["feature_time"], utc=True)
        frame["target_time"] = pd.to_datetime(frame["target_time"], utc=True)
        horizon = int(frame["horizon_minutes"].iloc[0])
        assert frame["horizon_minutes"].eq(horizon).all()
        assert (frame["target_time"] - frame["feature_time"]).eq(
            pd.Timedelta(minutes=horizon)
        ).all()
        expected = series["token_load"].reindex(pd.DatetimeIndex(frame["target_time"]))
        np.testing.assert_array_equal(frame["actual_token_load"], expected)
        assert np.isfinite(frame["predicted_token_load"]).all()
        assert frame["predicted_token_load"].nunique() > 1

    diagnostics = pd.read_csv(ROOT / "outputs/tables/table_04_xgb_prediction_diagnostics.csv")
    assert diagnostics["target_time_aligned"].astype(bool).all()
    assert not diagnostics["is_constant"].astype(bool).any()

    for horizon in HORIZONS:
        for kind in ("feature_importance", "shap_importance"):
            table = pd.read_csv(ROOT / f"outputs/tables/xgb_{kind}_h{horizon}.csv")
            assert set(table["feature"]) == set(feature_names)
            assert table["availability_audit"].eq("historical_or_relative_calendar").all()
        figure_paths = [
            ROOT / f"outputs/figures/fig_xgb_feature_importance_h{horizon}.png",
            ROOT / f"outputs/figures/fig_xgb_shap_summary_h{horizon}.png",
        ]
        figure_paths.extend(
            (ROOT / "outputs/figures").glob(f"fig_xgb_shap_dependence_h{horizon}_*.png")
        )
        assert len(figure_paths) == 5
        assert all(path.exists() and path.stat().st_size > 10_000 for path in figure_paths)

    comparison = pd.read_csv(ROOT / "outputs/tables/table_04_xgb_vs_baselines_h15.csv")
    assert set(comparison["baseline"]) == {"persistence", "seasonal_naive"}
    assert not comparison["xgboost_better"].astype(bool).any()


def main() -> None:
    feature_names, series = validate_feature_values()
    validate_tuning_configs_models(feature_names)
    validate_predictions_and_explanations(feature_names, series)
    print(
        "Week-3 validation passed: 36 features, target-time splits, 36 tuning rows, "
        "3 frozen models, 7 prediction files, and 15 explanation figures."
    )


if __name__ == "__main__":
    main()
