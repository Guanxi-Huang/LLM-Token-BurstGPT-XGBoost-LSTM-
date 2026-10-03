"""Clean and audit request-level BurstGPT v2.0 traces in bounded chunks.

The published ``Timestamp`` field is elapsed time in seconds; the collection
calendar is undisclosed.  ``timestamp`` below is therefore a synthetic UTC
axis anchored at the Unix epoch, while ``timestamp_relative_seconds`` keeps
the source value needed for an unambiguous interpretation.

Running this module writes the two primary cleaned Parquet files, a data-
quality table, two diagnostic figures, a sequential deletion audit, and a
two-day 5-minute smoke-test aggregate.  Raw files under ``Dataset/`` are only
read, never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / "outputs" / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


CHUNK_SIZE = 200_000
RAW_DIR = PROJECT_ROOT / "Dataset"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

WITHOUT_FAILS_FILES = {
    "1": RAW_DIR / "level_5_BurstGPT_v2.0_without_fails_1_Undisclosed_GitHub.csv",
    "2": RAW_DIR / "level_5_BurstGPT_v2.0_without_fails_2_Undisclosed_GitHub.csv",
    "3": RAW_DIR / "level_5_BurstGPT_v2.0_without_fails_3_Undisclosed_GitHub.csv",
}
COMPLETE_1_FILE = (
    RAW_DIR / "level_4_BurstGPT_v2.0_complete_1_Undisclosed_GitHub.csv"
)
PRIMARY_OUTPUTS = {
    "1": PROCESSED_DIR / "requests_1.parquet",
    "2": PROCESSED_DIR / "requests_2.parquet",
}

REQUIRED_SOURCE_COLUMNS = {
    "timestamp",
    "model",
    "request_tokens",
    "response_tokens",
    "total_tokens",
    "log_type",
}
NUMERIC_SOURCE_COLUMNS = (
    "timestamp",
    "request_tokens",
    "response_tokens",
    "total_tokens",
)
CRITICAL_TEXT_COLUMNS = ("model", "log_type")


def normalize_column_name(name: object) -> str:
    """Convert a source heading to lowercase snake_case."""
    value = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())
    return value.strip("_")


def _missing_mask(series: pd.Series) -> pd.Series:
    """Treat nulls and blank strings as missing without changing the source."""
    mask = series.isna()
    if pd.api.types.is_object_dtype(series.dtype) or isinstance(
        series.dtype, pd.StringDtype
    ):
        mask = mask | series.astype("string").str.strip().eq("").fillna(False)
    return mask


def _update_counter(counter: Counter[str], values: pd.Series) -> None:
    labels = values.astype("string").fillna("<MISSING>").str.strip()
    labels = labels.mask(labels.eq(""), "<MISSING>")
    counter.update({str(k): int(v) for k, v in labels.value_counts().items()})


def _stable_duplicate_mask(
    frame: pd.DataFrame, seen_row_hashes: set[int]
) -> pd.Series:
    """Find all-field duplicates within and across chunks.

    ``DataFrame.drop_duplicates`` implements the requested all-field duplicate
    rule inside a chunk.  A stable pandas row hash extends the same rule across
    chunk boundaries without retaining millions of full Python row objects.
    """
    first_occurrences = frame.drop_duplicates(keep="first")
    within_chunk = ~frame.index.to_series().isin(first_occurrences.index)
    row_hashes = pd.util.hash_pandas_object(frame, index=False, categorize=True)
    seen_before = row_hashes.map(lambda value: int(value) in seen_row_hashes)
    duplicate = within_chunk | seen_before
    seen_row_hashes.update(int(value) for value in row_hashes.loc[~duplicate])
    return duplicate


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _json_counter(counter: Counter[str]) -> str:
    return json.dumps(dict(sorted(counter.items())), ensure_ascii=False)


def _prepare_clean_chunk(
    source: pd.DataFrame,
    batch: str,
    duplicate_mask: pd.Series,
) -> tuple[pd.DataFrame, dict[str, int], int]:
    """Apply sequential cleaning rules and return deletion counts."""
    working = source.copy()
    numeric = {
        column: pd.to_numeric(working[column], errors="coerce")
        for column in NUMERIC_SOURCE_COLUMNS
    }

    # BurstGPT publishes relative seconds, not collection dates.  Explicit
    # ``unit='s'`` avoids pandas interpreting the values as nanoseconds.
    parsed_time = pd.to_datetime(
        numeric["timestamp"], unit="s", origin="unix", utc=True, errors="coerce"
    )

    alive = pd.Series(True, index=working.index)
    deletion_counts: dict[str, int] = {}

    bad_timestamp = parsed_time.isna()
    deletion_counts["deleted_unparseable_timestamp"] = int((alive & bad_timestamp).sum())
    alive &= ~bad_timestamp

    bad_tokens = numeric["request_tokens"].isna() | numeric["response_tokens"].isna()
    deletion_counts["deleted_non_numeric_tokens"] = int((alive & bad_tokens).sum())
    alive &= ~bad_tokens

    negative_tokens = (
        (numeric["request_tokens"] < 0)
        | (numeric["response_tokens"] < 0)
        | (numeric["total_tokens"] < 0)
    )
    deletion_counts["deleted_negative_tokens"] = int((alive & negative_tokens).sum())
    alive &= ~negative_tokens

    critical_missing = pd.Series(False, index=working.index)
    for column in CRITICAL_TEXT_COLUMNS:
        critical_missing |= _missing_mask(working[column])
    deletion_counts["deleted_critical_missing"] = int(
        (alive & critical_missing).sum()
    )
    alive &= ~critical_missing

    deletion_counts["deleted_duplicates"] = int((alive & duplicate_mask).sum())
    alive &= ~duplicate_mask

    kept = working.loc[alive].copy()
    request_tokens = numeric["request_tokens"].loc[alive].astype("int64")
    response_tokens = numeric["response_tokens"].loc[alive].astype("int64")
    reported_total = numeric["total_tokens"].loc[alive]
    mismatch_rows = int(
        (
            reported_total.notna()
            & reported_total.ne(request_tokens + response_tokens)
        ).sum()
    )

    cleaned = pd.DataFrame(index=kept.index)
    cleaned["timestamp"] = parsed_time.loc[alive]
    cleaned["timestamp_relative_seconds"] = numeric["timestamp"].loc[alive].astype(
        "float64"
    )
    cleaned["model"] = kept["model"].astype("string").str.strip()
    cleaned["request_tokens"] = request_tokens
    cleaned["response_tokens"] = response_tokens
    cleaned["reported_total_tokens"] = reported_total.astype("Int64")
    cleaned["total_tokens"] = request_tokens + response_tokens
    cleaned["log_type"] = kept["log_type"].astype("string").str.strip()

    if "session_id" in kept.columns:
        cleaned["session_id"] = kept["session_id"].astype("string").str.strip()
    if "elapsed_time" in kept.columns:
        cleaned["elapsed_time"] = pd.to_numeric(
            kept["elapsed_time"], errors="coerce"
        ).astype("Float64")

    cleaned["source_batch"] = batch
    return cleaned.reset_index(drop=True), deletion_counts, mismatch_rows


def clean_and_audit_without_fails(
    batch: str,
    path: Path,
    output_path: Path | None,
    chunksize: int = CHUNK_SIZE,
) -> dict[str, Any]:
    """Clean one without-fails file and calculate its data-quality profile."""
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    raw_rows = 0
    missing_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    log_type_counts: Counter[str] = Counter()
    seen_hashes: set[int] = set()
    raw_duplicate_rows = 0
    zero_output_rows = 0
    output_numeric_rows = 0
    total_mismatch_rows = 0
    clean_rows = 0
    clean_total_tokens_sum = 0
    max_clean_request_tokens = 0
    max_clean_response_tokens = 0
    max_clean_total_tokens = 0
    deletion_totals: Counter[str] = Counter()
    timestamp_min_s = float("inf")
    timestamp_max_s = float("-inf")
    previous_timestamp_s: float | None = None
    max_gap_seconds = 0.0
    gaps_over_one_hour = 0
    non_monotonic_transitions = 0
    source_columns: list[str] | None = None

    writer: pq.ParquetWriter | None = None
    temporary_output = output_path.with_suffix(".tmp.parquet") if output_path else None
    if temporary_output and temporary_output.exists():
        temporary_output.unlink()

    try:
        # ``Session ID`` is sparsely populated and may look numeric in some
        # chunks; force its semantic identifier type to keep cross-chunk
        # duplicate hashes stable and avoid mixed-type inference warnings.
        read_dtypes = {"Session ID": "string"} if batch == "3" else None
        for source_chunk in pd.read_csv(
            path, chunksize=chunksize, dtype=read_dtypes
        ):
            source_chunk.columns = [normalize_column_name(c) for c in source_chunk.columns]
            if source_columns is None:
                source_columns = source_chunk.columns.tolist()
                missing_required = REQUIRED_SOURCE_COLUMNS.difference(source_columns)
                if missing_required:
                    raise ValueError(
                        f"{path.name} is missing required columns: {sorted(missing_required)}"
                    )
            elif source_chunk.columns.tolist() != source_columns:
                raise ValueError(f"Schema changed inside {path.name}")

            raw_rows += len(source_chunk)
            for column in source_columns:
                missing_counts[column] += int(_missing_mask(source_chunk[column]).sum())
            _update_counter(model_counts, source_chunk["model"])
            _update_counter(log_type_counts, source_chunk["log_type"])

            output_numeric = pd.to_numeric(source_chunk["response_tokens"], errors="coerce")
            zero_output_rows += int(output_numeric.eq(0).sum())
            output_numeric_rows += int(output_numeric.notna().sum())

            source_for_duplicate_check = source_chunk[source_columns]
            duplicate_mask = _stable_duplicate_mask(
                source_for_duplicate_check, seen_hashes
            )
            raw_duplicate_rows += int(duplicate_mask.sum())

            timestamp_numeric = pd.to_numeric(source_chunk["timestamp"], errors="coerce")
            valid_timestamp = timestamp_numeric.dropna()
            if not valid_timestamp.empty:
                timestamp_min_s = min(timestamp_min_s, float(valid_timestamp.min()))
                timestamp_max_s = max(timestamp_max_s, float(valid_timestamp.max()))
                values = valid_timestamp.to_numpy(dtype="float64")
                if previous_timestamp_s is not None:
                    values_for_diff = np.concatenate(([previous_timestamp_s], values))
                else:
                    values_for_diff = values
                differences = np.diff(values_for_diff)
                if differences.size:
                    max_gap_seconds = max(max_gap_seconds, float(differences.max()))
                    gaps_over_one_hour += int((differences > 3600).sum())
                    non_monotonic_transitions += int((differences < 0).sum())
                previous_timestamp_s = float(values[-1])

            cleaned, deletion_counts, mismatches = _prepare_clean_chunk(
                source_chunk, batch, duplicate_mask
            )
            deletion_totals.update(deletion_counts)
            total_mismatch_rows += mismatches
            clean_rows += len(cleaned)
            if not cleaned.empty:
                clean_total_tokens_sum += int(cleaned["total_tokens"].sum())
                max_clean_request_tokens = max(
                    max_clean_request_tokens, int(cleaned["request_tokens"].max())
                )
                max_clean_response_tokens = max(
                    max_clean_response_tokens, int(cleaned["response_tokens"].max())
                )
                max_clean_total_tokens = max(
                    max_clean_total_tokens, int(cleaned["total_tokens"].max())
                )

            if output_path is not None and not cleaned.empty:
                table = pa.Table.from_pandas(cleaned, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary_output, table.schema, compression="zstd"
                    )
                writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    if output_path is not None:
        if writer is None or temporary_output is None or not temporary_output.exists():
            raise RuntimeError(f"No clean rows were written for {path.name}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_output, output_path)

    if source_columns is None:
        raise ValueError(f"No rows or header found in {path}")

    deleted_rows = raw_rows - clean_rows
    expected_deleted = sum(deletion_totals.values())
    if deleted_rows != expected_deleted:
        raise AssertionError(
            f"Deletion accounting mismatch for {path.name}: "
            f"{deleted_rows} != {expected_deleted}"
        )

    min_seconds = timestamp_min_s if np.isfinite(timestamp_min_s) else float("nan")
    max_seconds = timestamp_max_s if np.isfinite(timestamp_max_s) else float("nan")
    missing_rates = {
        column: _safe_ratio(missing_counts[column], raw_rows)
        for column in source_columns
    }
    result: dict[str, Any] = {
        "file_id": f"without_fails_{batch}",
        "file_name": path.name,
        "role": "main_experiment" if batch in {"1", "2"} else "robustness_only",
        "source_columns": json.dumps(source_columns, ensure_ascii=False),
        "raw_rows": raw_rows,
        "clean_rows": clean_rows,
        "deleted_rows": deleted_rows,
        "deletion_ratio": _safe_ratio(deleted_rows, raw_rows),
        **dict(deletion_totals),
        "raw_duplicate_rows": raw_duplicate_rows,
        "raw_duplicate_ratio": _safe_ratio(raw_duplicate_rows, raw_rows),
        "zero_output_token_rows": zero_output_rows,
        "zero_output_token_ratio": _safe_ratio(zero_output_rows, output_numeric_rows),
        "reported_total_mismatch_rows_after_cleaning": total_mismatch_rows,
        "clean_total_tokens_sum": clean_total_tokens_sum,
        "max_clean_request_tokens": max_clean_request_tokens,
        "max_clean_response_tokens": max_clean_response_tokens,
        "max_clean_total_tokens": max_clean_total_tokens,
        "timestamp_min_relative_seconds": min_seconds,
        "timestamp_max_relative_seconds": max_seconds,
        "timestamp_min_synthetic_utc": (
            pd.to_datetime(min_seconds, unit="s", origin="unix", utc=True).isoformat()
            if np.isfinite(min_seconds)
            else ""
        ),
        "timestamp_max_synthetic_utc": (
            pd.to_datetime(max_seconds, unit="s", origin="unix", utc=True).isoformat()
            if np.isfinite(max_seconds)
            else ""
        ),
        "duration_days_inclusive": (
            (max_seconds - min_seconds) / 86_400
            if np.isfinite(min_seconds) and np.isfinite(max_seconds)
            else float("nan")
        ),
        "mean_raw_requests_per_hour": (
            _safe_ratio(raw_rows, (max_seconds - min_seconds) / 3600)
            if max_seconds > min_seconds
            else float("nan")
        ),
        "max_inter_request_gap_seconds_in_file_order": max_gap_seconds,
        "gaps_over_one_hour_in_file_order": gaps_over_one_hour,
        "non_monotonic_timestamp_transitions": non_monotonic_transitions,
        "model_frequencies": _json_counter(model_counts),
        "log_type_frequencies": _json_counter(log_type_counts),
        "failure_condition": "not applicable (without_fails release)",
    }
    for column in sorted(set(source_columns) | REQUIRED_SOURCE_COLUMNS):
        result[f"missing_rate__{column}"] = missing_rates.get(column, float("nan"))
    return result


def audit_complete_1_failures(
    path: Path = COMPLETE_1_FILE, chunksize: int = CHUNK_SIZE
) -> dict[str, Any]:
    """Count complete_1 failures; do not mix complete data into training."""
    if not path.exists():
        raise FileNotFoundError(f"Missing failure-audit input file: {path}")

    raw_rows = 0
    numeric_output_rows = 0
    failure_rows = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        chunk.columns = [normalize_column_name(c) for c in chunk.columns]
        if "response_tokens" not in chunk.columns:
            raise ValueError(f"{path.name} has no Response tokens field")
        output_tokens = pd.to_numeric(chunk["response_tokens"], errors="coerce")
        raw_rows += len(chunk)
        numeric_output_rows += int(output_tokens.notna().sum())
        # BurstGPT paper, Section 1 (PDF p.2): failure requests have a zero
        # response length.  No other machine-readable failure flag is present.
        failure_rows += int(output_tokens.eq(0).sum())

    return {
        "file_id": "complete_1",
        "file_name": path.name,
        "role": "failure_audit_only",
        "raw_rows": raw_rows,
        "clean_rows": np.nan,
        "deleted_rows": np.nan,
        "deletion_ratio": np.nan,
        "zero_output_token_rows": failure_rows,
        "zero_output_token_ratio": _safe_ratio(failure_rows, numeric_output_rows),
        "failure_condition": (
            "response_tokens == 0 (BurstGPT paper Section 1, PDF p.2); "
            "no separate failure flag in the CSV"
        ),
    }


def _ordered_union(records: Iterable[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    for record in records:
        for column in record:
            if column not in columns:
                columns.append(column)
    return columns


def save_audit_tables(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Write the requested file-level audit and a compact deletion ledger."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame(records, columns=_ordered_union(records))
    audit.to_csv(TABLE_DIR / "table_00_data_quality.csv", index=False)

    step_columns = [
        "file_id",
        "raw_rows",
        "deleted_unparseable_timestamp",
        "deleted_non_numeric_tokens",
        "deleted_negative_tokens",
        "deleted_critical_missing",
        "deleted_duplicates",
        "clean_rows",
    ]
    available = [column for column in step_columns if column in audit.columns]
    audit.loc[audit["file_id"].str.startswith("without_fails_"), available].to_csv(
        TABLE_DIR / "table_00_cleaning_steps.csv", index=False
    )
    return audit


def create_diagnostic_figures() -> None:
    """Plot main-batch request time and log-token distributions."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"1": "#31688e", "2": "#35b779"}

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for batch, path in PRIMARY_OUTPUTS.items():
        times = pd.read_parquet(path, columns=["timestamp_relative_seconds"])[
            "timestamp_relative_seconds"
        ]
        ax.hist(
            times / 86_400,
            bins=120,
            histtype="step",
            linewidth=1.4,
            label=f"Batch {batch}",
            color=colors[batch],
        )
    ax.set_xlabel("Relative day; collection calendar undisclosed")
    ax.set_ylabel("Request count")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "fig_00_request_time_histogram.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for batch, path in PRIMARY_OUTPUTS.items():
        totals = pd.read_parquet(path, columns=["total_tokens"])["total_tokens"]
        ax.hist(
            np.log1p(totals),
            bins=100,
            histtype="step",
            linewidth=1.4,
            label=f"Batch {batch}",
            color=colors[batch],
        )
    ax.set_xlabel("log(1 + request tokens + response tokens)")
    ax.set_ylabel("Request count")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "fig_00_total_tokens_log_histogram.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def create_two_day_smoke_test() -> dict[str, Any]:
    """Aggregate the first continuous two-day span of batch 1 to 5 minutes."""
    columns = [
        "timestamp_relative_seconds",
        "request_tokens",
        "response_tokens",
        "total_tokens",
    ]
    requests = pd.read_parquet(PRIMARY_OUTPUTS["1"], columns=columns)
    start_s = float(np.floor(requests["timestamp_relative_seconds"].min() / 300) * 300)
    end_s = start_s + 2 * 86_400
    sample = requests.loc[
        requests["timestamp_relative_seconds"].between(
            start_s, end_s, inclusive="left"
        )
    ].copy()
    sample["bin_start_relative_seconds"] = (
        np.floor(sample["timestamp_relative_seconds"] / 300) * 300
    ).astype("int64")

    grouped = sample.groupby("bin_start_relative_seconds", as_index=True).agg(
        request_count=("total_tokens", "size"),
        request_tokens_sum=("request_tokens", "sum"),
        response_tokens_sum=("response_tokens", "sum"),
        total_tokens=("total_tokens", "sum"),
    )
    full_index = pd.Index(
        np.arange(int(start_s), int(end_s), 300),
        name="bin_start_relative_seconds",
    )
    grouped = grouped.reindex(full_index, fill_value=0).reset_index()
    grouped.insert(
        1,
        "bin_start_synthetic_utc",
        pd.to_datetime(
            grouped["bin_start_relative_seconds"],
            unit="s",
            origin="unix",
            utc=True,
        ),
    )

    if len(grouped) != 576:
        raise AssertionError(f"Two-day grid should have 576 bins, got {len(grouped)}")
    if int(grouped["request_count"].sum()) != len(sample):
        raise AssertionError("Smoke-test row counts do not reconcile")
    if not (
        grouped["request_tokens_sum"] + grouped["response_tokens_sum"]
        == grouped["total_tokens"]
    ).all():
        raise AssertionError("Smoke-test token sums do not reconcile")
    if int(grouped["total_tokens"].sum()) != int(sample["total_tokens"].sum()):
        raise AssertionError("Smoke-test aggregate differs from request-level total")

    output = PROCESSED_DIR / "smoke_2days_5min.csv"
    grouped.to_csv(output, index=False)
    return {
        "output": str(output.relative_to(PROJECT_ROOT)),
        "start_relative_seconds": start_s,
        "end_relative_seconds_exclusive": end_s,
        "five_minute_bins": len(grouped),
        "request_rows": len(sample),
        "request_tokens_sum": int(sample["request_tokens"].sum()),
        "response_tokens_sum": int(sample["response_tokens"].sum()),
        "total_tokens_sum": int(sample["total_tokens"].sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunksize", type=int, default=CHUNK_SIZE, help="CSV rows per chunk"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be positive")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for batch, path in WITHOUT_FAILS_FILES.items():
        output_path = PRIMARY_OUTPUTS.get(batch)
        print(f"[clean] {path.name} (batch={batch}, output={output_path})", flush=True)
        record = clean_and_audit_without_fails(
            batch=batch,
            path=path,
            output_path=output_path,
            chunksize=args.chunksize,
        )
        records.append(record)
        print(
            f"[done] batch={batch} raw={record['raw_rows']:,} "
            f"clean={record['clean_rows']:,} deleted={record['deleted_rows']:,}",
            flush=True,
        )

    complete_record = audit_complete_1_failures(chunksize=args.chunksize)
    records.append(complete_record)
    audit = save_audit_tables(records)
    create_diagnostic_figures()
    smoke = create_two_day_smoke_test()

    print("\nData-quality summary:")
    print(
        audit[
            [
                "file_id",
                "raw_rows",
                "clean_rows",
                "deletion_ratio",
                "zero_output_token_ratio",
            ]
        ].to_string(index=False)
    )
    print("\nTwo-day smoke test:")
    print(json.dumps(smoke, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
