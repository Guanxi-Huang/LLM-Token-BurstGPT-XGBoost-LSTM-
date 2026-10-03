"""Independent acceptance checks for all Week-2 deliverables."""

from __future__ import annotations

from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
SERIES_PATH = PROJECT_ROOT / "data" / "processed" / "series_5min.parquet"
HORIZONS = (5, 15, 60)


def require(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise AssertionError(f"Missing or empty deliverable: {path}")


def main() -> None:
    eda_builder_path = PROJECT_ROOT / "scripts" / "build_eda_notebook.py"
    required_files = [
        SERIES_PATH,
        PROJECT_ROOT / "notebooks" / "02_eda.ipynb",
        TABLE_DIR / "table_01_split_summary.csv",
        TABLE_DIR / "table_01_burst_threshold.csv",
        TABLE_DIR / "table_01_batch_boundary_audit.csv",
        TABLE_DIR / "table_01_manual_window_checks.csv",
        TABLE_DIR / "table_02_baseline_results.csv",
        TABLE_DIR / "pred_baselines_valid.csv",
        TABLE_DIR / "pred_baselines_test.csv",
    ]
    for path in required_files:
        require(path)

    eda_source = eda_builder_path.read_text(encoding="utf-8")
    forbidden_title_calls = ("plt.title(", ".set_title(", ".suptitle(", "fig.text(")
    for call in forbidden_title_calls:
        assert call not in eda_source, f"Formal EDA figures must not use {call}"
    assert ".text(" not in eda_source, (
        "Formal EDA figures must not simulate a title or subtitle with canvas text"
    )

    series = pd.read_parquet(SERIES_PATH)
    series.index = pd.DatetimeIndex(pd.to_datetime(series.index, utc=True))
    assert len(series) == len(pd.date_range(series.index.min(), series.index.max(), freq="5min"))
    assert series.index.is_unique and series.index.is_monotonic_increasing
    assert series.index.to_series().diff().dropna().eq(pd.Timedelta(minutes=5)).all()
    assert series.loc[series["request_count"].eq(0), ["mean_request_tokens", "mean_response_tokens"]].isna().all().all()
    assert set(series["split"].astype(str).unique()) == {"train", "valid", "test"}

    raw_rows = 0
    raw_tokens = 0
    for batch in (1, 2):
        raw = pd.read_parquet(
            PROJECT_ROOT / "data" / "processed" / f"requests_{batch}.parquet",
            columns=["total_tokens"],
        )
        raw_rows += len(raw)
        raw_tokens += int(raw["total_tokens"].sum())
    assert int(series["request_count"].sum()) == raw_rows
    assert int(series["token_load"].sum()) == raw_tokens

    threshold_table = pd.read_csv(TABLE_DIR / "table_01_burst_threshold.csv")
    threshold = float(threshold_table.loc[0, "p95_threshold"])
    recomputed = float(np.quantile(series.loc[series["split"].eq("train"), "token_load"], 0.95))
    assert np.isclose(threshold, recomputed)
    np.testing.assert_array_equal(series["actual_burst"], series["token_load"].ge(threshold))

    split_summary = pd.read_csv(TABLE_DIR / "table_01_split_summary.csv")
    assert split_summary["n_rows"].sum() == len(series)
    assert split_summary["split"].tolist() == ["train", "valid", "test"]
    checks = pd.read_csv(TABLE_DIR / "table_01_manual_window_checks.csv")
    assert len(checks) == 3 and checks["all_metrics_match"].all()

    for split_name in ("valid", "test"):
        predictions = pd.read_csv(
            TABLE_DIR / f"pred_baselines_{split_name}.csv", parse_dates=["timestamp"]
        )
        target_index = pd.DatetimeIndex(pd.to_datetime(predictions["timestamp"], utc=True))
        for minutes in HORIZONS:
            expected_persistence = series["token_load"].reindex(
                target_index - pd.Timedelta(minutes=minutes + 5)
            ).to_numpy(dtype=float)
            expected_seasonal = series["token_load"].reindex(
                target_index - pd.Timedelta(days=1)
            ).to_numpy(dtype=float)
            np.testing.assert_array_equal(
                predictions[f"pred_persistence_h{minutes}"], expected_persistence
            )
            np.testing.assert_array_equal(
                predictions[f"pred_seasonal_h{minutes}"], expected_seasonal
            )
            forecast_origin = pd.DatetimeIndex(
                pd.to_datetime(predictions[f"forecast_origin_h{minutes}"], utc=True)
            )
            last_available = pd.DatetimeIndex(
                pd.to_datetime(predictions[f"last_available_window_h{minutes}"], utc=True)
            )
            assert (last_available == forecast_origin - pd.Timedelta(minutes=5)).all()

    results = pd.read_csv(TABLE_DIR / "table_02_baseline_results.csv")
    assert len(results) == 2 * 2 * len(HORIZONS)
    assert set(results["horizon_minutes"]) == set(HORIZONS)
    confusion_total = results[
        ["true_negative", "false_positive", "false_negative", "true_positive"]
    ].sum(axis=1)
    np.testing.assert_array_equal(confusion_total, results["n_observations"])
    assert np.isfinite(results[["mae", "rmse", "smape_percent", "precision", "recall", "f1", "average_precision"]]).all().all()

    notebook = nbformat.read(PROJECT_ROOT / "notebooks" / "02_eda.ipynb", as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code_cells) >= 10
    assert all(cell.execution_count is not None for cell in code_cells)
    assert not any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )

    figure_names = [
        "fig_01_full_series_trend.png",
        "fig_02_representative_relative_week.png",
        "fig_03_relative_week_phase_heatmap.png",
        "fig_04_token_load_log_distribution.png",
        "fig_05_request_count_vs_token_load.png",
        "fig_06_model_share_over_time.png",
        "fig_07_log_type_share_over_time.png",
        "fig_08_acf_log_load.png",
        "fig_09_pacf_log_load.png",
        "fig_10_baseline_representative_day.png",
    ]
    for name in figure_names:
        path = FIGURE_DIR / name
        require(path)
        with Image.open(path) as image:
            width, height = image.size
            assert width >= 1000 and height >= 500, f"Unexpectedly small figure: {name} {image.size}"
            dpi = image.info.get("dpi")
            assert dpi is not None and min(dpi) >= 299, f"Figure is below 300 DPI: {name} {dpi}"

    print("Week-2 acceptance checks passed:")
    print(f"  canonical grid: {len(series):,} rows; {raw_rows:,} requests; {raw_tokens:,} tokens")
    print(f"  fixed train P95: {threshold:.6f} Token/5min")
    print("  manual windows: 3/3 matched")
    print("  target-aligned baselines: valid/test × 3 horizons passed")
    print("  evaluation rows: 12; every confusion matrix reconciled")
    print(f"  executed notebook: {len(code_cells)} code cells; no errors")
    print("  figures: 10/10 present, title-free, and dimension-checked")


if __name__ == "__main__":
    main()
