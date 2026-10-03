from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_data_quality.ipynb"
EXPECTED_CLEAN_ROWS = {"1": 1_363_384, "2": 3_695_526}
REQUIRED_COLUMNS = {
    "timestamp",
    "timestamp_relative_seconds",
    "model",
    "request_tokens",
    "response_tokens",
    "reported_total_tokens",
    "total_tokens",
    "log_type",
    "source_batch",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_parquet(batch: str) -> None:
    path = PROCESSED_DIR / f"requests_{batch}.parquet"
    require(path.exists(), f"Missing {path}")
    parquet = pq.ParquetFile(path)
    require(parquet.metadata.num_rows == EXPECTED_CLEAN_ROWS[batch], f"Unexpected row count in {path}")
    require(REQUIRED_COLUMNS.issubset(parquet.schema.names), f"Missing required columns in {path}")
    frame = pd.read_parquet(
        path,
        columns=["request_tokens", "response_tokens", "total_tokens", "source_batch"],
    )
    require((frame["request_tokens"] >= 0).all(), f"Negative request tokens in {path}")
    require((frame["response_tokens"] >= 0).all(), f"Negative response tokens in {path}")
    require(
        (frame["request_tokens"] + frame["response_tokens"] == frame["total_tokens"]).all(),
        f"Token totals do not reconcile in {path}",
    )
    require(frame["source_batch"].astype(str).eq(batch).all(), f"Wrong source batch in {path}")


def validate_audit_tables() -> None:
    quality_path = TABLE_DIR / "table_00_data_quality.csv"
    steps_path = TABLE_DIR / "table_00_cleaning_steps.csv"
    require(quality_path.exists(), f"Missing {quality_path}")
    require(steps_path.exists(), f"Missing {steps_path}")
    quality = pd.read_csv(quality_path).set_index("file_id")
    steps = pd.read_csv(steps_path).set_index("file_id")
    require(
        set(quality.index) == {"without_fails_1", "without_fails_2", "without_fails_3", "complete_1"},
        "Unexpected data-quality file set",
    )
    for batch, expected_rows in EXPECTED_CLEAN_ROWS.items():
        file_id = f"without_fails_{batch}"
        require(int(quality.loc[file_id, "clean_rows"]) == expected_rows, f"Audit row mismatch for {file_id}")
    deletion_columns = [column for column in steps.columns if column.startswith("deleted_")]
    require(
        (steps[deletion_columns].sum(axis=1) == steps["raw_rows"] - steps["clean_rows"]).all(),
        "Sequential deletion ledger does not reconcile",
    )
    failure_rate = float(quality.loc["complete_1", "zero_output_token_ratio"])
    require(abs(failure_rate - 25_443 / 1_429_737) < 1e-12, "Complete-1 failure rate mismatch")


def validate_smoke_test() -> None:
    path = PROCESSED_DIR / "smoke_2days_5min.csv"
    require(path.exists(), f"Missing {path}")
    frame = pd.read_csv(path)
    require(len(frame) == 576, "Smoke test must contain 576 five-minute bins")
    require(frame["bin_start_relative_seconds"].diff().dropna().eq(300).all(), "Smoke-test grid is not five-minute regular")
    require(int(frame["request_count"].sum()) == 6_107, "Smoke-test request count mismatch")
    require(
        (frame["request_tokens_sum"] + frame["response_tokens_sum"] == frame["total_tokens"]).all(),
        "Smoke-test token totals do not reconcile",
    )
    require(int(frame["total_tokens"].sum()) == 5_694_880, "Smoke-test total load mismatch")


def validate_figures() -> None:
    names = ["fig_00_request_time_histogram.png", "fig_00_total_tokens_log_histogram.png"]
    for name in names:
        path = FIGURE_DIR / name
        require(path.exists(), f"Missing {path}")
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (0, 0))
            require(min(dpi) >= 299, f"{path} is below 300 DPI: {dpi}")
            require(image.width > 1_000 and image.height > 600, f"{path} resolution is too small")


def validate_notebook() -> None:
    require(NOTEBOOK_PATH.exists(), f"Missing {NOTEBOOK_PATH}")
    notebook = nbformat.reads(NOTEBOOK_PATH.read_text(encoding="utf-8"), as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    require(code_cells, "Data-quality notebook has no code cells")
    require(all(cell.execution_count is not None for cell in code_cells), "Notebook has unexecuted code cells")
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    require(not errors, f"Notebook contains execution errors: {json.dumps(errors, ensure_ascii=False)}")


def main() -> None:
    for batch in EXPECTED_CLEAN_ROWS:
        validate_parquet(batch)
    validate_audit_tables()
    validate_smoke_test()
    validate_figures()
    validate_notebook()
    print("Week 1 validation passed: Parquet, audits, smoke test, figures, and notebook are consistent.")


if __name__ == "__main__":
    main()
