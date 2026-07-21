"""Build, audit, and split the canonical complete 5-minute target series.

The public BurstGPT timestamps are relative seconds.  ``src/01_clean.py``
maps them to a synthetic UTC axis solely to make time operations unambiguous;
no date in this module is interpreted as a real collection calendar date.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from utils import ensure_output_directories, project_path


PRIMARY_PARQUETS = ("requests_1.parquet", "requests_2.parquet")
REQUIRED_COLUMNS = {
    "timestamp",
    "request_tokens",
    "response_tokens",
    "total_tokens",
    "model",
    "log_type",
    "source_batch",
}
COUNT_COLUMNS = (
    "token_load",
    "request_count",
    "gpt4_request_count",
    "chatgpt_request_count",
    "api_request_count",
    "conversation_request_count",
    "batch_1_request_count",
    "batch_2_request_count",
)
LOG_START = "<!-- WEEK2_AGGREGATION_CHECK_START -->"
LOG_END = "<!-- WEEK2_AGGREGATION_CHECK_END -->"


def load_config() -> dict:
    with project_path("configs", "base.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_and_sort_requests() -> pd.DataFrame:
    """Read both main batches, preserve their identities, concatenate and sort."""
    frames: list[pd.DataFrame] = []
    for expected_batch, filename in zip(("1", "2"), PRIMARY_PARQUETS, strict=True):
        path = project_path("data", "processed", filename)
        if not path.exists():
            raise FileNotFoundError(f"Missing cleaned input: {path}")
        frame = pd.read_parquet(path)
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{filename} is missing required columns: {sorted(missing)}")
        frame = frame.loc[:, sorted(REQUIRED_COLUMNS)].copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        observed_batches = set(frame["source_batch"].astype("string").dropna().unique())
        if observed_batches != {expected_batch}:
            raise ValueError(
                f"{filename} has source_batch values {observed_batches}; "
                f"expected only {expected_batch!r}"
            )
        frames.append(frame)

    combined = pd.concat(frames, axis=0, ignore_index=True)
    combined.sort_values("timestamp", kind="mergesort", inplace=True, ignore_index=True)
    # Required leakage/order gate: no downstream aggregation runs without this.
    assert combined["timestamp"].is_monotonic_increasing
    assert combined["source_batch"].notna().all()
    return combined


def aggregate_complete_grid(requests: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Aggregate requests on a left-closed grid and explicitly retain zero bins."""
    prepared = requests.copy()
    prepared["gpt4_request_count"] = prepared["model"].eq("GPT-4").astype("int8")
    prepared["chatgpt_request_count"] = prepared["model"].eq("ChatGPT").astype("int8")
    prepared["api_request_count"] = prepared["log_type"].eq("API log").astype("int8")
    prepared["conversation_request_count"] = prepared["log_type"].eq(
        "Conversation log"
    ).astype("int8")
    prepared["batch_1_request_count"] = prepared["source_batch"].eq("1").astype("int8")
    prepared["batch_2_request_count"] = prepared["source_batch"].eq("2").astype("int8")

    aggregated = (
        prepared.set_index("timestamp")
        .resample(frequency, label="left", closed="left", origin="start_day")
        .agg(
            token_load=("total_tokens", "sum"),
            request_count=("total_tokens", "size"),
            mean_request_tokens=("request_tokens", "mean"),
            mean_response_tokens=("response_tokens", "mean"),
            gpt4_request_count=("gpt4_request_count", "sum"),
            chatgpt_request_count=("chatgpt_request_count", "sum"),
            api_request_count=("api_request_count", "sum"),
            conversation_request_count=("conversation_request_count", "sum"),
            batch_1_request_count=("batch_1_request_count", "sum"),
            batch_2_request_count=("batch_2_request_count", "sum"),
        )
    )

    first_bin = requests["timestamp"].min().floor(frequency)
    last_bin = requests["timestamp"].max().floor(frequency)
    complete_index = pd.date_range(first_bin, last_bin, freq=frequency, tz="UTC")
    series = aggregated.reindex(complete_index)
    series.index.name = "timestamp"
    for column in COUNT_COLUMNS:
        series[column] = series[column].fillna(0).astype("int64")

    # Means intentionally remain NaN in no-request bins.  Any later imputation
    # must use past-only information in the feature pipeline.
    zero_windows = series["request_count"].eq(0)
    assert series.loc[zero_windows, ["mean_request_tokens", "mean_response_tokens"]].isna().all().all()

    batch_1 = series["batch_1_request_count"].gt(0)
    batch_2 = series["batch_2_request_count"].gt(0)
    series["source_batch"] = np.select(
        [batch_1 & batch_2, batch_1, batch_2],
        ["1|2", "1", "2"],
        default="none",
    )
    series["is_zero_window"] = zero_windows

    expected_index = pd.date_range(series.index.min(), series.index.max(), freq=frequency)
    assert series.index.equals(expected_index)
    assert series.index.is_unique and series.index.is_monotonic_increasing
    assert int(series["request_count"].sum()) == len(requests)
    assert int(series["token_load"].sum()) == int(requests["total_tokens"].sum())
    assert (series["gpt4_request_count"] + series["chatgpt_request_count"]).equals(
        series["request_count"]
    )
    assert (series["api_request_count"] + series["conversation_request_count"]).equals(
        series["request_count"]
    )
    return series


def add_chronological_splits(
    series: pd.DataFrame, train_share: float, valid_share: float, quantile: float
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """Apply one chronological split and a train-only fixed burst threshold."""
    n_rows = len(series)
    train_end = int(n_rows * train_share)
    valid_end = int(n_rows * (train_share + valid_share))
    if not (0 < train_end < valid_end < n_rows):
        raise ValueError("The configured train/validation/test split is empty or invalid")

    result = series.copy()
    split = np.full(n_rows, "test", dtype=object)
    split[:train_end] = "train"
    split[train_end:valid_end] = "valid"
    result["split"] = pd.Series(split, index=result.index, dtype="string")

    train_load = result.iloc[:train_end]["token_load"].to_numpy(dtype="float64")
    threshold = float(np.quantile(train_load, quantile))
    result["actual_burst"] = result["token_load"].ge(threshold)

    rows: list[dict[str, object]] = []
    for split_name in ("train", "valid", "test"):
        part = result.loc[result["split"].eq(split_name)]
        rows.append(
            {
                "split": split_name,
                "start_timestamp_synthetic_utc": part.index.min().isoformat(),
                "end_timestamp_synthetic_utc": part.index.max().isoformat(),
                "n_rows": len(part),
                "share_of_rows": len(part) / n_rows,
                "zero_windows": int(part["is_zero_window"].sum()),
                "total_requests": int(part["request_count"].sum()),
                "total_token_load": int(part["token_load"].sum()),
                "actual_burst_windows": int(part["actual_burst"].sum()),
                "actual_burst_rate": float(part["actual_burst"].mean()),
                "train_p95_threshold": threshold,
            }
        )
    return result, threshold, pd.DataFrame(rows)


def build_batch_boundary_audit(
    requests: pd.DataFrame, series: pd.DataFrame, frequency: str
) -> pd.DataFrame:
    batch_1 = requests.loc[requests["source_batch"].eq("1"), "timestamp"]
    batch_2 = requests.loc[requests["source_batch"].eq("2"), "timestamp"]
    last_1 = batch_1.max()
    first_2 = batch_2.min()
    between = series.loc[(series.index > last_1.floor(frequency)) & (series.index < first_2.floor(frequency))]
    return pd.DataFrame(
        [
            {
                "previous_batch": "1",
                "next_batch": "2",
                "previous_batch_last_request": last_1.isoformat(),
                "next_batch_first_request": first_2.isoformat(),
                "request_gap_seconds": (first_2 - last_1).total_seconds(),
                "previous_batch_last_window": last_1.floor(frequency).isoformat(),
                "next_batch_first_window": first_2.floor(frequency).isoformat(),
                "complete_windows_between": len(between),
                "zero_windows_between": int(between["is_zero_window"].sum()),
            }
        ]
    )


def manual_spot_checks(
    requests: pd.DataFrame, series: pd.DataFrame, frequency: str, seed: int
) -> pd.DataFrame:
    """Recompute three deterministic random bins directly from request rows."""
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(series), size=3, replace=False))
    delta = pd.Timedelta(frequency)
    rows: list[dict[str, object]] = []
    for position in positions:
        start = series.index[int(position)]
        end = start + delta
        raw = requests.loc[
            requests["timestamp"].ge(start) & requests["timestamp"].lt(end)
        ]
        aggregate = series.loc[start]
        raw_values = {
            "token_load": int(raw["total_tokens"].sum()),
            "request_count": int(len(raw)),
            "mean_request_tokens": float(raw["request_tokens"].mean()),
            "mean_response_tokens": float(raw["response_tokens"].mean()),
            "gpt4_request_count": int(raw["model"].eq("GPT-4").sum()),
            "api_request_count": int(raw["log_type"].eq("API log").sum()),
        }
        aggregate_values = {
            key: float(aggregate[key]) if key.startswith("mean_") else int(aggregate[key])
            for key in raw_values
        }
        matches = all(
            np.isclose(raw_values[key], aggregate_values[key], equal_nan=True)
            for key in raw_values
        )
        if not matches:
            raise AssertionError(f"Manual aggregation check failed for [{start}, {end})")
        rows.append(
            {
                "window_start_synthetic_utc": start.isoformat(),
                "window_end_exclusive_synthetic_utc": end.isoformat(),
                "raw_token_load": raw_values["token_load"],
                "series_token_load": aggregate_values["token_load"],
                "raw_request_count": raw_values["request_count"],
                "series_request_count": aggregate_values["request_count"],
                "raw_mean_request_tokens": raw_values["mean_request_tokens"],
                "series_mean_request_tokens": aggregate_values["mean_request_tokens"],
                "raw_mean_response_tokens": raw_values["mean_response_tokens"],
                "series_mean_response_tokens": aggregate_values["mean_response_tokens"],
                "raw_gpt4_request_count": raw_values["gpt4_request_count"],
                "series_gpt4_request_count": aggregate_values["gpt4_request_count"],
                "raw_api_request_count": raw_values["api_request_count"],
                "series_api_request_count": aggregate_values["api_request_count"],
                "all_metrics_match": matches,
            }
        )
    return pd.DataFrame(rows)


def _format_number(value: object) -> str:
    if pd.isna(value):
        return "NaN"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def update_experiment_log(
    checks: pd.DataFrame,
    boundary: pd.DataFrame,
    threshold: float,
    split_summary: pd.DataFrame,
) -> None:
    """Insert an idempotent, human-readable audit block in the experiment log."""
    columns = [
        "window_start_synthetic_utc",
        "window_end_exclusive_synthetic_utc",
        "raw_token_load",
        "series_token_load",
        "raw_request_count",
        "series_request_count",
        "raw_mean_request_tokens",
        "series_mean_request_tokens",
        "raw_mean_response_tokens",
        "series_mean_response_tokens",
        "raw_gpt4_request_count",
        "series_gpt4_request_count",
        "raw_api_request_count",
        "series_api_request_count",
        "all_metrics_match",
    ]
    headers = [
        "窗口起点（合成UTC）",
        "窗口终点（不含）",
        "原始Token",
        "聚合Token",
        "原始请求数",
        "聚合请求数",
        "原始平均输入",
        "聚合平均输入",
        "原始平均输出",
        "聚合平均输出",
        "原始GPT-4数",
        "聚合GPT-4数",
        "原始API数",
        "聚合API数",
        "一致",
    ]
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for _, row in checks.iterrows():
        table_lines.append(
            "| " + " | ".join(_format_number(row[column]) for column in columns) + " |"
        )

    boundary_row = boundary.iloc[0]
    split_text = "；".join(
        f"{row.split}={int(row.n_rows):,}行"
        for row in split_summary.itertuples(index=False)
    )
    block = "\n".join(
        [
            LOG_START,
            "",
            "## 第2周：5分钟目标序列、人工核验与固定突发阈值",
            "",
            f"- 正式运行日期：{date.today().isoformat()}；命令：`& .\\.venv\\Scripts\\python.exe scripts/run_week2_pipeline.py`；配置：`configs/base.yaml`（seed=42）。",
            "- 窗口语义：左闭右开 `[start, start+5min)`；公开时间仅为相对秒，表中的1970日期只是合成UTC轴。",
            f"- 切分：{split_text}。训练集P95固定阈值为 `{threshold:.6f}` Token/5min，三段统一按 `token_load >= threshold` 标记突发。",
            f"- 批次边界：1号最后请求到2号第一请求间隔 `{boundary_row['request_gap_seconds']:.0f}` 秒；两者所在窗口之间有 `{int(boundary_row['complete_windows_between'])}` 个完整窗口，其中 `{int(boundary_row['zero_windows_between'])}` 个为零负载。",
            "- 以下三个窗口由seed=42从完整网格无放回抽样，再回到两个请求级Parquet用布尔条件逐项重算；均值在零请求窗保持NaN。",
            "",
            *table_lines,
            "",
            "结论：三个窗口的Token总量、请求数、输入/输出Token均值、GPT-4请求数和API请求数全部一致；窗口边界与零窗构造通过人工复算。",
            "",
            LOG_END,
        ]
    )
    path = project_path("docs", "experiment_log.md")
    existing = path.read_text(encoding="utf-8")
    if LOG_START in existing and LOG_END in existing:
        prefix, remainder = existing.split(LOG_START, maxsplit=1)
        _, suffix = remainder.split(LOG_END, maxsplit=1)
        updated = prefix.rstrip() + "\n\n" + block + suffix
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    path.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    seed = int(args.seed if args.seed is not None else config["project"]["seed"])
    frequency = str(config["series"]["frequency"])
    train_share = float(config["split"]["train"])
    valid_share = float(config["split"]["validation"])
    quantile = float(config["burst"]["train_quantile"])
    if not np.isclose(train_share + valid_share + float(config["split"]["test"]), 1.0):
        raise ValueError("Configured split shares must sum to 1")

    ensure_output_directories()
    requests = load_and_sort_requests()
    series = aggregate_complete_grid(requests, frequency)
    series, threshold, split_summary = add_chronological_splits(
        series, train_share, valid_share, quantile
    )
    boundary = build_batch_boundary_audit(requests, series, frequency)
    checks = manual_spot_checks(requests, series, frequency, seed)

    series_path = project_path("data", "processed", "series_5min.parquet")
    series.to_parquet(series_path, compression="zstd", index=True)
    table_dir = project_path("outputs", "tables")
    split_summary.to_csv(table_dir / "table_01_split_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "definition": "training token_load quantile",
                "quantile": quantile,
                "comparison": "token_load >= threshold",
                "p95_threshold": threshold,
                "train_rows": int((series["split"] == "train").sum()),
                "time_semantics": "relative seconds on a synthetic UTC axis",
            }
        ]
    ).to_csv(table_dir / "table_01_burst_threshold.csv", index=False)
    boundary.to_csv(table_dir / "table_01_batch_boundary_audit.csv", index=False)
    checks.to_csv(table_dir / "table_01_manual_window_checks.csv", index=False)
    update_experiment_log(checks, boundary, threshold, split_summary)

    print(f"Saved canonical series: {series_path}")
    print(f"Rows: {len(series):,}; zero windows: {int(series['is_zero_window'].sum()):,}")
    print(f"Training-only P95 threshold: {threshold:.6f} Token/5min")
    print(split_summary.to_string(index=False))
    print("Manual checks: 3/3 windows matched request-level recomputation")


if __name__ == "__main__":
    main()
